"""Previsión con varios modelos y validación con origen móvil, previsión por grupo, eventos y detección de cambios.
Sin dependencias nuevas: Holt-Winters aditivo amortiguado (scipy), un modelo de retardos con gradient boosting
(sklearn) y los modelos básicos. El mejor modelo se elige por su error en datos que no vio."""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor

from kikolens.analyze import timeseries as ts

MIN_POINTS = 16
MAX_GROUPS = 20


# ── Modelos: cada uno devuelve `horizon` valores futuros ─────────────────────
def _future_index(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    freq = index.freq or pd.tseries.frequencies.to_offset(ts.infer_frequency(index))
    return pd.date_range(index[-1], periods=horizon + 1, freq=freq)[1:]


def naive(values, index, horizon, period, events=None):
    return np.repeat(values[-1], horizon)


def seasonal_naive(values, index, horizon, period, events=None):
    if not period or period < 2 or len(values) < period:
        return naive(values, index, horizon, period)
    cycle = values[-period:]
    return np.array([cycle[i % period] for i in range(horizon)])


def trend_seasonality(values, index, horizon, period, events=None):
    return ts._fit_predict(values, horizon, period)[0]


def _hw_run(y, params, m, seasonal, horizon=0):
    alpha, beta, gamma, phi = params
    n = len(y)
    if seasonal:
        level = y[:m].mean()
        trend = (y[m:2 * m].mean() - y[:m].mean()) / m
        season = list(y[:m] - level)
    else:
        level, trend, season = y[0], (y[min(n - 1, 3)] - y[0]) / max(min(n - 1, 3), 1), []
    sse, fitted = 0.0, np.empty(n)
    for t in range(n):
        s_old = season[t % m] if seasonal else 0.0
        pred = level + phi * trend + s_old
        fitted[t] = pred
        sse += (y[t] - pred) ** 2
        new_level = alpha * (y[t] - s_old) + (1 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        if seasonal:
            season[t % m] = gamma * (y[t] - level - phi * trend) + (1 - gamma) * s_old
        level = new_level
    if not horizon:
        return sse, fitted
    damp = np.cumsum(phi ** np.arange(1, horizon + 1))
    out = level + damp * trend
    if seasonal:
        out = out + np.array([season[(n + h) % m] for h in range(horizon)])
    return sse, out


def holt_winters(values, index, horizon, period, events=None):
    """Holt-Winters aditivo con tendencia amortiguada; ajusta sus parámetros minimizando el error a un paso."""
    y = np.asarray(values, dtype=float)
    seasonal = bool(period and period >= 2 and len(y) >= 2 * period)
    m = int(period) if seasonal else 1
    start = np.array([0.3, 0.05, 0.1, 0.97])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = minimize(lambda p: _hw_run(y, p, m, seasonal)[0], start, method="L-BFGS-B",
                       bounds=[(0.001, 0.999), (0.0, 0.5), (0.0, 0.999), (0.8, 0.999)], options={"maxiter": 60})
    return _hw_run(y, fit.x, m, seasonal, horizon)[1]


def _calendar(index: pd.DatetimeIndex, events) -> np.ndarray:
    cols = [index.dayofweek.to_numpy(), index.month.to_numpy(), index.dayofyear.to_numpy()]
    flag = np.zeros(len(index))
    if events is not None:
        flag = pd.Series(index).dt.normalize().isin(pd.to_datetime(list(events)).normalize()).to_numpy().astype(float)
    return np.column_stack(cols + [flag])


def lag_boosting(values, index, horizon, period, events=None):
    """Gradient boosting sobre retardos, medias móviles, calendario y eventos, tras quitar la tendencia lineal.
    Predice de forma recursiva, usando sus propias previsiones como retardos."""
    y = np.asarray(values, dtype=float)
    n = len(y)
    lags = sorted({1, 2, 3, 7, 14, int(period or 0)} - {0})
    lags = [lag for lag in lags if lag < n // 2]
    if not lags:
        return naive(values, index, horizon, period)
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (intercept + slope * t)
    future_idx = _future_index(index, horizon)
    cal_all = _calendar(index.append(future_idx), events)
    top = max(lags)

    def features(series, pos):
        row = [series[pos - lag] for lag in lags]
        recent = series[max(0, pos - 7):pos]
        return row + [recent.mean(), recent.std() if len(recent) > 1 else 0.0] + list(cal_all[pos])

    X = np.array([features(resid, i) for i in range(top, n)])
    model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=4, min_samples_leaf=4, random_state=0)
    model.fit(X, resid[top:])
    extended = list(resid)
    for h in range(horizon):
        extended.append(float(model.predict(np.array([features(np.array(extended), n + h)]))[0]))
    steps = np.arange(n, n + horizon, dtype=float)
    return intercept + slope * steps + np.array(extended[n:])


MODELS = {"Naive": naive, "Seasonal naive": seasonal_naive, "Trend + seasonality": trend_seasonality,
          "Holt-Winters": holt_winters, "Lag boosting": lag_boosting}


# ── Validación con origen móvil ──────────────────────────────────────────────
def backtest(series: pd.Series, horizon: int, models=None, folds: int = 3, period="auto", events=None) -> dict:
    """Entrena con el pasado y predice el tramo siguiente, varias veces. Devuelve la tabla de errores y los residuos."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    models = list(models or MODELS)
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. Choose from {list(MODELS)}.")
    n = len(series)
    if n < MIN_POINTS:
        raise ValueError(f"At least {MIN_POINTS} periods are needed to compare models.")
    period = ts.guess_period(series) if period == "auto" else period
    h = min(horizon, max(2, n // 4))
    folds = max(1, min(int(folds), (n - max(8, 2 * (period or 0))) // h))
    values = series.to_numpy(dtype=float)
    scale = np.mean(np.abs(np.diff(values, n=1))) or 1.0
    rows, residuals = [], {m: [] for m in models}
    for model in models:
        errors = []
        for k in range(folds):
            cut = n - h * (folds - k)
            train, test = values[:cut], values[cut:cut + h]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = MODELS[model](train, series.index[:cut], len(test), period, events)
            except (ValueError, np.linalg.LinAlgError):
                errors = None
                break
            errors.append(test - pred)
        if not errors:
            rows.append({"model": model, "MAE": np.nan, "RMSE": np.nan, "MASE": np.nan, "status": "failed"})
            continue
        flat = np.concatenate(errors)
        residuals[model] = errors
        rows.append({"model": model, "MAE": float(np.mean(np.abs(flat))), "RMSE": float(np.sqrt(np.mean(flat ** 2))),
                     "MASE": float(np.mean(np.abs(flat)) / scale), "status": "ok"})
    board = pd.DataFrame(rows).sort_values("MAE", na_position="last", ignore_index=True)
    return {"leaderboard": board, "residuals": residuals, "folds": folds, "horizon_tested": h, "period": period}


def forecast_models(series: pd.Series, horizon: int = 12, models=None, folds: int = 3, events=None,
                    coverage: float = 0.9) -> dict:
    """Compara los modelos con origen móvil, elige el de menor error y lo ajusta con toda la serie. El intervalo
    sale de los errores reales que tuvo ese modelo al predecir datos que no había visto."""
    if not 0.5 <= coverage < 1:
        raise ValueError("coverage must be between 0.5 and 1.")
    test = backtest(series, horizon, models, folds, events=events)
    board = test["leaderboard"]
    ok = board[board["status"] == "ok"]
    if ok.empty:
        raise ValueError("No model could be fitted to this series.")
    best = ok.iloc[0]["model"]
    period = test["period"]
    values = series.to_numpy(dtype=float)
    future_idx = _future_index(series.index, horizon)
    forecasts = {}
    for name in ok["model"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecasts[name] = pd.Series(MODELS[name](values, series.index, horizon, period, events), index=future_idx)

    errs = [np.asarray(e) for e in test["residuals"][best]]
    lo_q, hi_q = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    lows, highs = [], []
    for step in range(horizon):
        at_step = np.array([e[min(step, len(e) - 1)] for e in errs])
        pool = np.concatenate([at_step, np.concatenate(errs)]) if len(at_step) < 6 else at_step
        lows.append(np.quantile(pool, lo_q))
        highs.append(np.quantile(pool, hi_q))
    # los errores cuantiles no deben estrecharse con el horizonte: se imponen los máximos acumulados
    lows, highs = np.minimum.accumulate(lows), np.maximum.accumulate(highs)
    best_f = forecasts[best]
    frame = pd.DataFrame({"forecast": best_f, "lower": best_f + lows, "upper": best_f + highs})
    naive_row = board[board["model"] == "Naive"]
    beats = bool(len(naive_row) and naive_row.iloc[0]["status"] == "ok" and ok.iloc[0]["MAE"] < naive_row.iloc[0]["MAE"])
    return {"forecast": frame, "best": best, "leaderboard": board, "all_forecasts": forecasts, "period": period,
            "folds": test["folds"], "horizon_tested": test["horizon_tested"], "beats_naive": beats, "coverage": coverage,
            "summary": (f"{best} was the most accurate over {test['folds']} rolling test(s) of {test['horizon_tested']} "
                        f"period(s) (MAE {ok.iloc[0]['MAE']:.4g})."
                        + ("" if best == "Naive" else
                           f" It {'beats' if beats else 'does not beat'} repeating the last value.")
                        + f" The range covers about {coverage:.0%} of what happened in those tests.")}


def forecast_figure(series: pd.Series, result: dict, last: int | None = 200):
    import plotly.graph_objects as go

    shown = series.iloc[-last:] if last else series
    fc = result["forecast"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shown.index, y=shown, name="Observed", mode="lines"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["lower"], mode="lines", line=dict(width=0), fill="tonexty", name="Range",
                             hoverinfo="skip", opacity=0.3))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], name=f"{result['best']} forecast", mode="lines", line=dict(dash="dash")))
    for name, f in result["all_forecasts"].items():
        if name != result["best"]:
            fig.add_trace(go.Scatter(x=f.index, y=f, name=name, mode="lines", visible="legendonly", line=dict(width=1)))
    fig.update_layout(title="Forecast with the best model")
    return fig


# ── Cambios de nivel o de pendiente ──────────────────────────────────────────
def _line_sse(x, y) -> float:
    if len(y) < 3:
        return 0.0
    coef = np.polyfit(x, y, 1)
    return float(np.sum((y - np.polyval(coef, x)) ** 2))


def detect_changepoints(series: pd.Series, max_points: int = 4, min_size: int | None = None, sensitivity: float = 1.0) -> list:
    """Puntos donde la serie cambia de nivel o de pendiente (segmentación binaria con rectas por tramos). Cada
    corte debe reducir el error más que una penalidad basada en el ruido, así que una serie sin cambios no devuelve nada."""
    if not 1 <= max_points <= 10:
        raise ValueError("max_points must be between 1 and 10.")
    n = len(series)
    if n < 20:
        raise ValueError("At least 20 periods are needed to look for changes.")
    y = series.to_numpy(dtype=float)
    min_size = int(min_size or max(5, n // 12))
    noise = np.median(np.abs(np.diff(y, n=2))) / 0.6745 / np.sqrt(6) if n > 3 else float(np.std(y))
    noise = noise or float(np.std(y)) or 1.0
    penalty = float(sensitivity) * 6 * noise ** 2 * np.log(n)
    x = np.arange(n, dtype=float)
    segments, found = [(0, n)], []
    for _ in range(max_points):
        best = None
        for lo, hi in segments:
            if hi - lo < 2 * min_size:
                continue
            whole = _line_sse(x[lo:hi], y[lo:hi])
            for cut in range(lo + min_size, hi - min_size + 1, max(1, (hi - lo) // 150)):
                gain = whole - _line_sse(x[lo:cut], y[lo:cut]) - _line_sse(x[cut:hi], y[cut:hi])
                if best is None or gain > best[0]:
                    best = (gain, cut, lo, hi)
        if best is None or best[0] <= penalty:
            break
        _, cut, lo, hi = best
        segments.remove((lo, hi))
        segments += [(lo, cut), (cut, hi)]
        found.append(cut)
    out = []
    for cut in sorted(found):
        lo = max([c for c in [0] + found if c < cut and c > 0] + [0])
        hi = min([c for c in found + [n] if c > cut] + [n])
        before, after = y[lo:cut], y[cut:hi]
        slope_before, slope_after = np.polyfit(np.arange(len(before)), before, 1)[0], np.polyfit(np.arange(len(after)), after, 1)[0]
        level_jump = float(after[:3].mean() - before[-3:].mean())
        kind = "level shift" if abs(level_jump) > 3 * noise else "trend change"
        out.append({"date": series.index[cut], "kind": kind, "level change": level_jump,
                    "slope before": float(slope_before), "slope after": float(slope_after)})
    return out


def changepoint_figure(series: pd.Series, points: list):
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=series.index, y=series, mode="lines", name="Observed"))
    for p in points:
        fig.add_shape(type="line", x0=p["date"], x1=p["date"], y0=0, y1=1, yref="paper", line=dict(dash="dash"))
        fig.add_annotation(x=p["date"], y=1, yref="paper", text=p["kind"], showarrow=False, yanchor="bottom")
    fig.update_layout(title="Detected changes", showlegend=False)
    return fig


# ── Por grupo ────────────────────────────────────────────────────────────────
def forecast_by_group(df: pd.DataFrame, date_col: str, value_col: str, group_col: str, horizon: int = 12,
                      freq: str | None = None, agg: str = "sum", models=None, events=None) -> dict:
    """Una previsión por cada valor de `group_col` (los MAX_GROUPS grupos con más actividad)."""
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found.")
    groups = df[group_col].dropna().astype(str).value_counts()
    if len(groups) < 2:
        raise ValueError("The group column needs at least two values.")
    parts, summary, skipped = [], [], []
    for g in groups.index[:MAX_GROUPS]:
        sub = df[df[group_col].astype(str) == g]
        try:
            series = ts.prepare_series(sub, date_col, value_col, freq=freq, agg=agg)
            res = forecast_models(series, horizon, models, folds=2, events=events)
        except ValueError as e:
            skipped.append({"group": g, "reason": str(e)})
            continue
        fc = res["forecast"].copy()
        fc.insert(0, "group", g)
        parts.append(fc)
        board = res["leaderboard"]
        summary.append({"group": g, "periods": len(series), "best model": res["best"],
                        "MAE": float(board[board["model"] == res["best"]]["MAE"].iloc[0]),
                        "next period": float(fc["forecast"].iloc[0]), "last observed": float(series.iloc[-1])})
    if not parts:
        raise ValueError("No group had enough data to forecast.")
    return {"forecast": pd.concat(parts).rename_axis("date").reset_index(), "summary": pd.DataFrame(summary),
            "skipped": pd.DataFrame(skipped, columns=["group", "reason"]),
            "truncated": len(groups) > MAX_GROUPS}
