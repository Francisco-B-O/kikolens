"""Efecto de una intervención. `causal_impact` compara lo ocurrido después con lo que habría pasado sin ella
(estimado con series de control o con la propia tendencia anterior) y `diff_in_diff` compara el cambio de un grupo
tratado con el de uno de control. Ninguno prueba causalidad por sí solo: dependen de que el control sea comparable."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import RidgeCV

from kikolens.analyze import forecasting


def _split(series: pd.Series, intervention):
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("The series needs a date index.")
    series = series.sort_index().astype(float)
    when = pd.Timestamp(intervention)
    pre, post = series[series.index < when], series[series.index >= when]
    if len(pre) < 20:
        raise ValueError("At least 20 periods before the intervention are needed.")
    if len(post) < 3:
        raise ValueError("At least 3 periods after the intervention are needed.")
    return pre, post, when


def _placebo_errors(pre: pd.Series, ctrl: pd.DataFrame, n_post: int) -> np.ndarray:
    """Error medio al "predecir" ventanas del periodo previo con datos anteriores a cada una: mide cuánto se equivoca el
    método al extrapolar, que es lo que importa y los residuos dentro de la muestra no muestran."""
    start = max(20, len(pre) // 2)
    errors = []
    for k in range(start, len(pre) - n_post + 1, max(1, n_post // 8)):
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(ctrl.iloc[:k], pre.iloc[:k])
        errors.append(float((pre.iloc[k:k + n_post] - model.predict(ctrl.iloc[k:k + n_post])).mean()))
    return np.asarray(errors)


def causal_impact(series: pd.Series, intervention, control: pd.DataFrame | None = None, coverage: float = 0.9) -> dict:
    """Efecto estimado por periodo y acumulado, con intervalo y una probabilidad de que sea casualidad. La incertidumbre
    sale de lo que el mismo método se equivocó al predecir periodos anteriores en los que no pasó nada."""
    if not 0.5 <= coverage < 1:
        raise ValueError("coverage must be between 0.5 and 1.")
    pre, post, when = _split(series, intervention)
    n_post = len(post)
    if control is not None and control.shape[1]:
        ctrl = control.reindex(series.index).astype(float).interpolate(limit_direction="both")
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(ctrl.loc[pre.index], pre)
        expected = pd.Series(model.predict(ctrl.loc[post.index]), index=post.index)
        fit_resid = (pre - model.predict(ctrl.loc[pre.index])).to_numpy()
        placebo = _placebo_errors(pre, ctrl.loc[pre.index], n_post)
        method = f"a regression on {control.shape[1]} control series"
        band = np.quantile(fit_resid - fit_resid.mean(), [(1 - coverage) / 2, 1 - (1 - coverage) / 2])
        # la tendencia compartida con los controles puede cambiar: se suma el error típico de extrapolar
        rms = float(np.sqrt(np.mean(placebo ** 2))) if len(placebo) >= 3 else float(fit_resid.std() / np.sqrt(n_post) * 2)
        independent = (len(pre) - max(20, len(pre) // 2)) // n_post + 1     # ventanas que no se solapan
    else:
        if n_post > len(pre) // 2:
            raise ValueError("Without control series, the period after the intervention must be at most half the period before.")
        result = forecasting.forecast_models(pre, n_post, coverage=coverage)
        expected = result["forecast"]["forecast"].set_axis(post.index)
        folds = [np.asarray(e) for e in forecasting.backtest(pre, n_post)["residuals"][result["best"]]]
        method = f"the {result['best']} forecast of the earlier trend"
        band = np.array([float((result["forecast"]["lower"] - result["forecast"]["forecast"]).min()),
                         float((result["forecast"]["upper"] - result["forecast"]["forecast"]).max())])
        window = np.array([f.mean() for f in folds])
        pooled = np.concatenate(folds)
        # el error medio de una ventana: el desvío de los errores entre sí más el sesgo típico entre ventanas
        rms = float(np.sqrt(pooled.var() / max(n_post, 1) + np.mean(window ** 2)))
        independent = len(folds)
    effect = post - expected
    average = float(effect.mean())
    rms = max(rms, 1e-12)
    # con pocas ventanas de referencia el error típico está mal estimado: la distribución t lo reconoce ensanchando el intervalo
    dof = max(2, independent)
    z = stats.t.ppf(0.5 + coverage / 2, dof)
    low, high = average - z * rms, average + z * rms
    p = float(2 * stats.t.sf(abs(average) / rms, dof))
    relative = average / float(expected.mean()) if expected.mean() else np.nan
    detected = bool(low > 0 or high < 0)
    frame = pd.DataFrame({"observed": post, "expected": expected, "effect": effect,
                          "lower": expected + band[0], "upper": expected + band[1]})
    return {"table": frame, "average_effect": average, "total_effect": float(effect.sum()), "relative_effect": relative,
            "interval": (float(low), float(high)), "p_value": p, "detected": detected, "method": method,
            "figure": _figure(pre, frame, when),
            "summary": (f"After {when.date()}, the series averaged {post.mean():.4g} against {expected.mean():.4g} expected "
                        f"from {method}: an effect of {average:+.4g} per period ({relative:+.1%}), "
                        f"{int(coverage * 100)}% range {low:+.4g} to {high:+.4g}. "
                        + ("This is unlikely to be chance " if detected else "This could easily be chance ")
                        + f"(p = {p:.3g}). It only counts as an effect of the intervention if nothing else changed then.")}


def _figure(pre: pd.Series, frame: pd.DataFrame, when) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pre.index, y=pre.values, name="Before", line=dict(color="#6a655c")))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["observed"], name="After (observed)"))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["expected"], name="Expected without it", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=list(frame.index) + list(frame.index)[::-1],
                             y=list(frame["upper"]) + list(frame["lower"])[::-1], fill="toself", opacity=0.15,
                             line=dict(width=0), name="Range", hoverinfo="skip"))
    fig.add_vline(x=when, line_dash="dot")
    fig.update_layout(title="What happened against what was expected")
    return fig


def diff_in_diff(df: pd.DataFrame, outcome: str, treated: str, post: str, coverage: float = 0.95) -> dict:
    """Diferencia en diferencias con dos grupos y dos momentos. `treated` y `post` son columnas 0/1 (o sí/no)."""
    for col in (outcome, treated, post):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")

    def flag(series):
        s = series.astype(str).str.strip().str.lower()
        out = s.map({"1": 1, "1.0": 1, "true": 1, "yes": 1, "y": 1, "0": 0, "0.0": 0, "false": 0, "no": 0, "n": 0})
        if out.isna().any():
            raise ValueError(f"'{series.name}' must have only two values: 0/1 or yes/no.")
        return out.astype(float)

    data = pd.DataFrame({"y": pd.to_numeric(df[outcome], errors="coerce"), "t": flag(df[treated]), "p": flag(df[post])}).dropna()
    cells = data.groupby(["t", "p"])["y"].agg(["mean", "size"])
    if len(cells) < 4 or (cells["size"] < 3).any():
        raise ValueError("Each of the four combinations (treated or not, before or after) needs at least 3 rows.")
    X = np.column_stack([np.ones(len(data)), data["t"], data["p"], data["t"] * data["p"]])
    y = data["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - 4
    cov = np.linalg.inv(X.T @ X) * (resid @ resid / dof)
    se = float(np.sqrt(cov[3, 3]))
    effect = float(beta[3])
    crit = stats.t.ppf(0.5 + coverage / 2, dof)
    p = float(2 * stats.t.sf(abs(effect / se), dof)) if se > 0 else 0.0
    means = cells["mean"].unstack()
    return {"effect": effect, "interval": (effect - crit * se, effect + crit * se), "p_value": p, "means": means,
            "summary": (f"Treated rows changed {means.loc[1.0, 1.0] - means.loc[1.0, 0.0]:+.4g} while the others changed "
                        f"{means.loc[0.0, 1.0] - means.loc[0.0, 0.0]:+.4g}: an estimated effect of {effect:+.4g} "
                        f"({int(coverage * 100)}% range {effect - crit * se:+.4g} to {effect + crit * se:+.4g}, p = {p:.3g}). "
                        "This assumes both groups would have moved alike without the treatment.")}
