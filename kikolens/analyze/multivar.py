"""Series temporales con varias variables. `granger` pregunta si el pasado de una variable ayuda a predecir otra (causalidad de
Granger: predecir, no causar). `forecast_with_drivers` prevé una serie usando también sus variables explicativas y comprueba, con
pruebas sobre el pasado, si de verdad mejoran frente a usar solo la propia serie."""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from kikolens.analyze import timeseries

MIN_ROWS = 40


def _frame(df: pd.DataFrame, date: str, columns: list, agg: str = "mean") -> pd.DataFrame:
    """Tabla regular (una fila por periodo) con las columnas pedidas, huecos interpolados."""
    if date not in df.columns:
        raise ValueError(f"Column '{date}' not found.")
    for c in columns:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"'{c}' must be numeric.")
    parts = [timeseries.prepare_series(df, date, c, agg=agg).rename(c) for c in columns]
    out = pd.concat(parts, axis=1).dropna()
    if len(out) < MIN_ROWS:
        raise ValueError(f"At least {MIN_ROWS} periods are needed; there are {len(out)}.")
    return out


def _lags(frame: pd.DataFrame, columns: list, lags: int) -> pd.DataFrame:
    return pd.concat({f"{c}_lag{k}": frame[c].shift(k) for c in columns for k in range(1, lags + 1)}, axis=1)


def granger(df: pd.DataFrame, date: str, cause: str, effect: str, max_lag: int = 7, agg: str = "mean", difference: bool = True) -> dict:
    """¿Mejora el pasado de `cause` la predicción de `effect` más allá del propio pasado de `effect`? F-test para cada retardo.
    Por defecto se trabaja con las diferencias (cambios de un periodo a otro) para no confundir tendencias compartidas con relación."""
    if cause == effect:
        raise ValueError("Choose two different columns.")
    if not 1 <= max_lag <= 30:
        raise ValueError("max_lag must be between 1 and 30.")
    frame = _frame(df, date, [cause, effect], agg)
    if difference:
        frame = frame.diff().dropna()
    rows = []
    for lag in range(1, max_lag + 1):
        y = frame[effect].iloc[lag:].to_numpy()
        own = np.column_stack([frame[effect].shift(k).iloc[lag:] for k in range(1, lag + 1)])
        other = np.column_stack([frame[cause].shift(k).iloc[lag:] for k in range(1, lag + 1)])
        restricted = np.column_stack([np.ones(len(y)), own])
        full = np.column_stack([restricted, other])
        if len(y) <= full.shape[1] + 5:
            break
        rss0 = float(np.sum((y - restricted @ np.linalg.lstsq(restricted, y, rcond=None)[0]) ** 2))
        rss1 = float(np.sum((y - full @ np.linalg.lstsq(full, y, rcond=None)[0]) ** 2))
        dof = len(y) - full.shape[1]
        f = ((rss0 - rss1) / lag) / (rss1 / dof) if rss1 > 0 else np.nan
        rows.append({"lag": lag, "F": float(f), "p-value": float(stats.f.sf(f, lag, dof)), "improvement %": float((1 - rss1 / rss0) * 100)})
    if not rows:
        raise ValueError("There are too few periods for this many lags.")
    table = pd.DataFrame(rows)
    table["adjusted p"] = np.minimum(1.0, table["p-value"] * len(table))          # Bonferroni: se miran varios retardos
    best = table.loc[table["adjusted p"].idxmin()]
    significant = bool(best["adjusted p"] < 0.05)
    fig = go.Figure(go.Bar(x=table["lag"], y=-np.log10(table["adjusted p"].clip(lower=1e-12)), name="evidence"))
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", annotation_text="p = 0.05")
    fig.update_layout(title=f"Does the past of {cause} help predict {effect}?", xaxis_title="Lag (periods)", yaxis_title="-log10 of adjusted p")
    return {"table": table, "significant": significant, "best_lag": int(best["lag"]), "figure": fig,
            "summary": (f"Yes: the last {int(best['lag'])} period(s) of {cause} improve the prediction of {effect} by {best['improvement %']:.1f}% "
                        f"(adjusted p = {best['adjusted p']:.3g}). This is predictive, not proof that {cause} causes {effect}."
                        if significant else
                        f"No clear evidence that the past of {cause} helps predict {effect} (best adjusted p = {best['adjusted p']:.3g}).")}


def forecast_with_drivers(df: pd.DataFrame, date: str, target: str, drivers: list, horizon: int = 12, lags: int = 7, agg: str = "mean",
                          scenario: dict | None = None, folds: int = 4) -> dict:
    """Previsión recursiva con retardos del objetivo y de las variables explicativas. Las explicativas futuras se suponen iguales a su
    último valor, o `scenario` (columna -> factor, por ejemplo {"price": 1.1} para un precio un 10 % más alto)."""
    if not drivers:
        raise ValueError("Choose at least one driver column.")
    if target in drivers:
        raise ValueError("The target cannot be one of its own drivers.")
    if horizon < 1 or horizon > 120:
        raise ValueError("horizon must be between 1 and 120.")
    scenario = scenario or {}
    unknown = [c for c in scenario if c not in drivers]
    if unknown:
        raise ValueError(f"The scenario mentions columns that are not drivers: {unknown}.")
    frame = _frame(df, date, [target] + list(drivers), agg)
    lags = max(1, min(lags, len(frame) // 6))

    def design(data: pd.DataFrame, cols: list) -> pd.DataFrame:
        out = _lags(data, cols, lags)
        idx = data.index
        out["dow"], out["month"] = idx.dayofweek, idx.month
        return out

    def fit_predict(train: pd.DataFrame, cols: list):
        X = design(train, cols).iloc[lags:]
        y = train[target].iloc[lags:]
        model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        return model

    # pruebas sobre el pasado: error a un paso, con y sin variables explicativas
    n, size = len(frame), max(8, len(frame) // (folds + 2))
    errors = {"drivers": [], "alone": []}
    for k in range(folds):
        cut = n - size * (folds - k)
        train, test = frame.iloc[:cut], frame.iloc[cut:cut + size]
        for label, cols in (("drivers", [target] + list(drivers)), ("alone", [target])):
            model = fit_predict(train, cols)
            joined = pd.concat([train, test])
            pred = model.predict(design(joined, cols).iloc[len(train):])
            errors[label].append(np.abs(test[target].to_numpy() - pred))
    mae = {k: float(np.mean(np.concatenate(v))) for k, v in errors.items()}
    paired = np.concatenate(errors["alone"]) - np.concatenate(errors["drivers"])
    p = float(stats.wilcoxon(paired).pvalue) if np.any(paired != 0) else 1.0
    helps = mae["drivers"] < mae["alone"] and p < 0.05

    model = fit_predict(frame, [target] + list(drivers))
    history = frame.copy()
    freq = pd.infer_freq(frame.index) or timeseries.infer_frequency(frame.index)
    future_index = pd.date_range(frame.index[-1], periods=horizon + 1, freq=freq)[1:]
    last = frame[list(drivers)].iloc[-1]
    forecasts = []
    for stamp in future_index:
        row = {target: np.nan, **{d: float(last[d]) * float(scenario.get(d, 1.0)) for d in drivers}}
        history = pd.concat([history, pd.DataFrame(row, index=[stamp])])
        features = design(history, [target] + list(drivers)).iloc[[-1]]
        value = float(model.predict(features)[0])
        history.loc[stamp, target] = value
        forecasts.append(value)
    resid = np.concatenate(errors["drivers"])
    hi = float(np.quantile(resid, 0.9))                      # error absoluto que no se superó el 90 % de las veces
    out = pd.DataFrame({"forecast": forecasts, "lower": np.array(forecasts) - hi, "upper": np.array(forecasts) + hi}, index=future_index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index[-120:], y=frame[target].iloc[-120:], name="Observed"))
    fig.add_trace(go.Scatter(x=out.index, y=out["forecast"], name="Forecast", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=list(out.index) + list(out.index)[::-1], y=list(out["upper"]) + list(out["lower"])[::-1], fill="toself",
                             opacity=0.15, line=dict(width=0), name="Range", hoverinfo="skip"))
    fig.update_layout(title=f"{target} with {', '.join(drivers)}" + (f" (scenario: {scenario})" if scenario else ""))
    return {"forecast": out, "mae_with_drivers": mae["drivers"], "mae_alone": mae["alone"], "drivers_help": helps, "p_value": p,
            "figure": fig, "scenario": scenario,
            "summary": (f"Using {', '.join(drivers)} cuts the one-step error from {mae['alone']:.4g} to {mae['drivers']:.4g} "
                        f"({(1 - mae['drivers'] / mae['alone']):.0%} better, p = {p:.3g}). " if helps else
                        f"The drivers do not clearly improve on the series alone (error {mae['drivers']:.4g} with them against {mae['alone']:.4g}, p = {p:.3g}). ")
                       + "Future values of the drivers are assumed " + ("as in the scenario." if scenario else "equal to the last observed ones.")}
