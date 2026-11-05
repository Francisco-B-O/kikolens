"""Modelo de mezcla de marketing y elasticidad de precio.

  * `marketing_mix`: cuánto aporta cada canal a las ventas, con el efecto que se arrastra en el tiempo (adstock), la saturación (cada euro extra rinde menos),
    tendencia, estacionalidad y variables de control. Los coeficientes de los canales no pueden ser negativos y los parámetros se eligen probando
    sobre el final de la serie, no sobre lo ya visto.
  * `optimize_budget`: cómo repartir un presupuesto entre canales con las curvas de respuesta del modelo.
  * `price_elasticity`: cuánto cae la demanda al subir el precio (log-log, con errores robustos), por grupo, con el precio que maximiza ingresos o margen
    y, si hay un instrumento, corrigiendo el sesgo de que el precio suele fijarse mirando la demanda.
Es observacional: sirve para orientar y para decidir qué probar con un experimento, no sustituye al experimento."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import minimize, nnls

from kikolens.analyze import timeseries as ts

DECAYS = (0.0, 0.2, 0.4, 0.6, 0.8)
HALVES = (0.25, 0.5, 1.0, 2.0, 4.0)                  # múltiplos del gasto medio donde el canal llega a la mitad de su efecto
SHAPES = (1.0, 2.0)
MIN_PERIODS = 30
HOLDOUT = 0.2


# ── Transformaciones ─────────────────────────────────────────────────────────
def adstock(x, decay: float) -> np.ndarray:
    """El efecto de un gasto se arrastra: cada periodo suma lo nuevo más `decay` de lo acumulado."""
    if not 0 <= decay < 1:
        raise ValueError("The carry-over (decay) must be at least 0 and below 1.")
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


def saturation(x, half: float, shape: float = 1.0) -> np.ndarray:
    """Curva de Hill: sube de 0 a 1, llega a 0,5 cuando `x == half` y se aplana a partir de ahí."""
    x = np.maximum(np.asarray(x, dtype=float), 0)
    if half <= 0:
        raise ValueError("The half-saturation point must be positive.")
    return x ** shape / (x ** shape + half ** shape)


def _seasonal_terms(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Tendencia y ondas de Fourier de la estacionalidad que cabe en la serie."""
    n = len(index)
    cols = {"trend": np.linspace(-1, 1, n)}
    freq = ts.infer_frequency(index)
    periods = {"D": [(7, 2), (365.25, 3)], "W": [(52.18, 3)], "MS": [(12, 3)], "QS": [(4, 1)]}.get(freq, [])
    t = np.arange(n)
    for period, order in periods:
        if n < 2 * period:
            continue
        for k in range(1, order + 1):
            cols[f"season {period:g} sin{k}"] = np.sin(2 * np.pi * k * t / period)
            cols[f"season {period:g} cos{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=index)


def _design(spend: pd.DataFrame, params: dict) -> np.ndarray:
    return np.column_stack([saturation(adstock(spend[c].to_numpy(), params[c]["decay"]), params[c]["half"], params[c]["shape"]) for c in spend.columns])


def _fit(channels: np.ndarray, free: np.ndarray, y: np.ndarray) -> tuple:
    """Mínimos cuadrados con los canales ≥ 0 y el resto libre (cada columna libre entra como ±)."""
    A = np.column_stack([channels, free, -free])
    coef, _ = nnls(A, y, maxiter=20000)
    k, m = channels.shape[1], free.shape[1]
    return coef[:k], coef[k:k + m] - coef[k + m:]


def _mape(y, pred):
    return float(np.mean(np.abs(y - pred) / np.maximum(np.abs(y), 1e-9)))


# ── Modelo de mezcla ─────────────────────────────────────────────────────────
def marketing_mix(df: pd.DataFrame, date: str, target: str, channels: list, controls: list | None = None, seasonality: bool = True) -> dict:
    """Contribución de cada canal a `target`. Las filas se agregan por periodo (suma del objetivo y del gasto, media de los controles)."""
    controls = [c for c in (controls or []) if c not in channels and c != target]
    if not channels:
        raise ValueError("Give at least one channel (a column of spend or effort).")
    for c in [target, *channels, *controls]:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"'{c}' must be a numeric column.")
    if date not in df.columns:
        raise ValueError(f"Column '{date}' not found.")
    data = df[[date, target, *channels, *controls]].copy()
    data[date] = pd.to_datetime(data[date], errors="coerce")
    data = data.dropna(subset=[date]).sort_values(date)
    freq = ts.infer_frequency(pd.DatetimeIndex(data[date]))
    grouped = data.set_index(date).resample(freq).agg({**{c: "sum" for c in [target, *channels]}, **{c: "mean" for c in controls}})
    grouped[controls] = grouped[controls].interpolate(limit_direction="both") if controls else grouped[controls]
    grouped = grouped.fillna(0).dropna()
    if len(grouped) < MIN_PERIODS:
        raise ValueError(f"At least {MIN_PERIODS} periods are needed to separate channels from trend and season (found {len(grouped)}).")
    spend = grouped[channels].astype(float)
    if (spend < 0).any().any():
        raise ValueError("Spend cannot be negative.")
    dead = [c for c in channels if spend[c].sum() <= 0]
    if dead:
        raise ValueError(f"Channel(s) with no spend at all: {', '.join(dead)}.")
    y = grouped[target].to_numpy(dtype=float)
    pieces = [pd.Series(1.0, index=grouped.index, name="baseline")]
    if seasonality:
        pieces.append(_seasonal_terms(grouped.index))
    base_columns = sum(1 if isinstance(p, pd.Series) else p.shape[1] for p in pieces)
    if controls:
        pieces.append((grouped[controls] - grouped[controls].mean()) / grouped[controls].std(ddof=0).replace(0, 1))
    free = pd.concat(pieces, axis=1)
    free_values = free.to_numpy(dtype=float)
    cut = int(len(y) * (1 - HOLDOUT))
    means = {c: float(spend[c][spend[c] > 0].mean()) for c in channels}
    params = {c: {"decay": 0.3, "half": means[c], "shape": 1.0} for c in channels}

    def holdout_error(candidate):
        X = _design(spend, candidate)
        coef, base = _fit(X[:cut], free_values[:cut], y[:cut])
        return _mape(y[cut:], X[cut:] @ coef + free_values[cut:] @ base)

    best = holdout_error(params)
    for _ in range(2):                                              # dos pasadas: cada canal se ajusta con los demás ya ajustados
        for c in channels:
            for decay in DECAYS:
                for half in HALVES:
                    for shape in SHAPES:
                        trial = {**params, c: {"decay": decay, "half": half * means[c], "shape": shape}}
                        err = holdout_error(trial)
                        if err < best - 1e-9:
                            best, params = err, trial
    X = _design(spend, params)
    coef, base = _fit(X, free_values, y)
    fitted = X @ coef + free_values @ base
    resid = y - fitted
    r2 = float(1 - (resid ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    parts = pd.DataFrame(X * coef, index=grouped.index, columns=channels)
    parts["baseline, trend and season"] = free_values[:, :base_columns] @ base[:base_columns]
    if controls:
        parts["controls"] = free_values[:, base_columns:] @ base[base_columns:]
    total_spend = spend.sum()
    contribution = parts[channels].sum()
    steady = {}
    for c in channels:
        p = params[c]
        effective = lambda s, p=p, k=channels.index(c): coef[k] * float(saturation(s / (1 - p["decay"]), p["half"], p["shape"]))        # noqa: E731
        current = float(spend[c].mean())
        marginal = (effective(current * 1.01) - effective(current)) / max(current * 0.01, 1e-9)
        steady[c] = {"fn": effective, "current": current, "marginal": marginal}
    table = pd.DataFrame({
        "channel": channels, "spend": total_spend.values, "contribution": contribution.values, "share of outcome": contribution.values / max(y.sum(), 1e-12),
        "return per unit": contribution.values / np.maximum(total_spend.values, 1e-12), "marginal return": [steady[c]["marginal"] for c in channels],
        "carry-over": [params[c]["decay"] for c in channels], "half-saturation": [params[c]["half"] for c in channels]})
    active = [i for i, c in enumerate(coef) if c > 0]
    t_values = dict.fromkeys(channels, 0.0)
    if active:                                                    # error estándar aproximado de los canales activos con la matriz de diseño final (HC1)
        D = np.column_stack([X[:, active], free_values])
        beta_d, se_d, _ = _ols(D, y)
        for pos, i in enumerate(active):
            t_values[channels[i]] = float(beta_d[pos] / se_d[pos]) if se_d[pos] > 0 else 0.0
    table["t statistic"] = [t_values[c] for c in table["channel"]]
    table["distinguishable from zero"] = table["t statistic"] > 2
    table["saturated"] = [bool(spend[c].mean() / (1 - params[c]["decay"]) > 2 * params[c]["half"]) for c in channels]
    table = table.sort_values("return per unit", ascending=False).reset_index(drop=True)
    notes = []
    holdout = holdout_error(params)
    if holdout > 0.25:
        notes.append(f"On the last {HOLDOUT:.0%} of the periods the model is off by {holdout:.0%} on average: treat the split between channels as rough.")
    corr = spend.corr().abs().to_numpy().copy()                  # pandas 3 devuelve arrays de solo lectura
    np.fill_diagonal(corr, 0)
    if corr.size and corr.max() > 0.8:
        i, j = np.unravel_index(corr.argmax(), corr.shape)
        notes.append(f"'{channels[i]}' and '{channels[j]}' move together (correlation {corr[i, j]:.2f}): the model cannot tell their effects apart reliably. "
                     "Vary one of them on purpose to find out.")
    flat = [c for c in channels if spend[c].std() < 0.05 * max(spend[c].mean(), 1e-9)]
    if flat:
        notes.append(f"{', '.join(flat)} barely changed over time, so its effect cannot be measured from this data.")
    weak = [c for c in channels if not bool(table.set_index("channel").loc[c, "distinguishable from zero"])]
    if weak:
        notes.append(f"{', '.join(weak)} cannot be told apart from zero effect (t statistic below 2): its share is as likely noise as a real contribution.")
    zero = [c for c in channels if coef[channels.index(c)] == 0 and c not in weak]
    if zero:
        notes.append(f"{', '.join(zero)} gets no contribution: either it adds nothing visible, or other channels or the season explain the same ups and downs.")
    if len(grouped) < 104 and freq == "W":
        notes.append("Less than two years of weekly data: the yearly season cannot be separated well from the channels.")
    best_row = table.iloc[0]
    out = {"table": table, "parts": parts, "fitted": pd.Series(fitted, index=grouped.index), "actual": pd.Series(y, index=grouped.index), "r2": r2, "holdout_error": holdout,
           "params": params, "coefficients": dict(zip(channels, coef)), "baseline_share": float(parts["baseline, trend and season"].sum() / max(y.sum(), 1e-12)),
           "frequency": freq, "_steady": steady, "channels": channels, "notes": notes,
           "figure": _decomposition_figure(parts), "curves_figure": _curves_figure(steady, params, channels), "fit_figure": _fit_figure(y, fitted, grouped.index, cut),
           "summary": (f"Channels explain {1 - out_baseline(parts, y):.0%} of the outcome and the baseline the rest; the best return per unit is '{best_row['channel']}' "
                       f"({best_row['return per unit']:.3g}). The model explains {r2:.0%} of the variation and is off by {holdout:.0%} on the last periods.")}
    return out


def out_baseline(parts: pd.DataFrame, y: np.ndarray) -> float:
    base = parts["baseline, trend and season"].sum() + (parts["controls"].sum() if "controls" in parts else 0.0)
    return float(base / max(y.sum(), 1e-12))


def _decomposition_figure(parts: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for c in parts.columns:
        fig.add_trace(go.Scatter(x=parts.index, y=parts[c], stackgroup="one", name=str(c), mode="lines"))
    fig.update_layout(title="What each part contributes to the outcome over time", yaxis_title="Contribution")
    return fig


def _curves_figure(steady: dict, params: dict, channels: list) -> go.Figure:
    fig = go.Figure()
    for c in channels:
        top = steady[c]["current"] * 3
        grid = np.linspace(0, max(top, 1e-9), 60)
        fig.add_trace(go.Scatter(x=grid, y=[steady[c]["fn"](s) for s in grid], name=str(c), mode="lines"))
        fig.add_trace(go.Scatter(x=[steady[c]["current"]], y=[steady[c]["fn"](steady[c]["current"])], mode="markers", showlegend=False, marker=dict(size=9)))
    fig.update_layout(title="Response curves: outcome per period at a steady spend (dots mark today)", xaxis_title="Spend per period", yaxis_title="Outcome from the channel")
    return fig


def _fit_figure(y, fitted, index, cut) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index, y=y, name="Actual"))
    fig.add_trace(go.Scatter(x=index, y=fitted, name="Model", line=dict(dash="dash")))
    fig.add_vrect(x0=index[cut], x1=index[-1], opacity=0.07, line_width=0, annotation_text="used to check")
    fig.update_layout(title="Actual against the model")
    return fig


# ── Presupuesto ──────────────────────────────────────────────────────────────
def optimize_budget(model: dict, total: float | None = None, low: float = 0.5, high: float = 2.0) -> dict:
    """Reparte `total` (por defecto el gasto actual por periodo) entre los canales para maximizar el resultado según sus curvas de respuesta.
    Cada canal puede ir de `low` a `high` veces su gasto actual, para no extrapolar lejos de lo observado."""
    if "_steady" not in model:
        raise ValueError("Pass the result of marketing_mix.")
    if not 0 <= low <= 1 <= high:
        raise ValueError("Bounds must satisfy 0 <= low <= 1 <= high.")
    channels, steady = model["channels"], model["_steady"]
    current = np.array([steady[c]["current"] for c in channels])
    budget = float(total) if total is not None else float(current.sum())
    if budget <= 0:
        raise ValueError("The budget must be positive.")
    lo, hi = current * low, current * high
    if lo.sum() > budget + 1e-9 or hi.sum() < budget - 1e-9:
        raise ValueError(f"A budget of {budget:,.4g} cannot be split within {low:g}x to {high:g}x of each channel's spend "
                         f"(it must be between {lo.sum():,.4g} and {hi.sum():,.4g}).")

    def response(s):
        return sum(steady[c]["fn"](v) for c, v in zip(channels, s))

    scale = budget / len(channels)
    best = None
    rng = np.random.default_rng(0)
    starts = [np.clip(current * budget / current.sum(), lo, hi)] + [lo + rng.dirichlet(np.ones(len(channels))) * (budget - lo.sum()) for _ in range(5)]
    for x0 in starts:
        x0 = np.clip(x0, lo, hi)
        res = minimize(lambda s: -response(s) / scale, x0, method="SLSQP", bounds=list(zip(lo, hi)),
                       constraints=[{"type": "eq", "fun": lambda s: (s.sum() - budget) / scale}], options={"maxiter": 300, "ftol": 1e-12})
        if abs(res.x.sum() - budget) < 1e-6 * max(budget, 1) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise ValueError("The search did not find a valid split.")
    proposed = np.clip(best.x, lo, hi)
    now, later = response(current * budget / current.sum()), response(proposed)
    table = pd.DataFrame({"channel": channels, "today": current, "scaled today": current * budget / current.sum(), "proposed": proposed})
    table["change"] = table["proposed"] / table["scaled today"].replace(0, np.nan) - 1
    gain = later - now
    fig = go.Figure([go.Bar(x=channels, y=table["scaled today"], name="Today"), go.Bar(x=channels, y=table["proposed"], name="Proposed")])
    fig.update_layout(barmode="group", title="Spend per period: today and proposed", yaxis_title="Spend")
    return {"table": table, "gain": float(gain), "relative_gain": float(gain / max(now, 1e-12)), "outcome_now": float(now), "outcome_proposed": float(later), "budget": budget, "figure": fig,
            "summary": (f"Moving spend as proposed changes the outcome from the channels by {gain:+,.4g} per period ({gain / max(now, 1e-12):+.1%}) with the same budget. "
                        "This extrapolates the fitted curves; test a shift in one channel before moving everything.")}


# ── Elasticidad de precio ────────────────────────────────────────────────────
def _ols(X: np.ndarray, y: np.ndarray) -> tuple:
    """Coeficientes y errores estándar robustos a heterocedasticidad (HC1)."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    cov = bread @ meat @ bread * len(y) / max(len(y) - X.shape[1], 1)
    return beta, np.sqrt(np.diag(cov)), resid


def _elasticity(frame: pd.DataFrame, quantity: str, price: str, controls: list, instrument: list | None) -> dict:
    y, p = np.log(frame[quantity].to_numpy(float)), np.log(frame[price].to_numpy(float))
    ctrl = frame[controls].to_numpy(float) if controls else np.empty((len(frame), 0))
    X = np.column_stack([np.ones(len(y)), p, ctrl])
    beta, se, _ = _ols(X, y)
    out = {"elasticity": float(beta[1]), "se": float(se[1]), "n": len(y), "method": "ols"}
    if instrument:
        Z = np.column_stack([np.ones(len(y)), ctrl, frame[instrument].to_numpy(float)])
        first = np.linalg.lstsq(Z, p, rcond=None)[0]
        p_hat = Z @ first
        X2 = np.column_stack([np.ones(len(y)), p_hat, ctrl])
        b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
        resid = y - X @ b2
        bread = np.linalg.pinv(X2.T @ X2)
        meat = (X2 * resid[:, None]).T @ (X2 * resid[:, None])
        cov = bread @ meat @ bread * len(y) / max(len(y) - X2.shape[1], 1)
        restricted = np.column_stack([np.ones(len(y)), ctrl])
        rss_full = np.sum((p - p_hat) ** 2)
        rss_restricted = np.sum((p - restricted @ np.linalg.lstsq(restricted, p, rcond=None)[0]) ** 2)
        q = len(instrument)
        out.update({"elasticity": float(b2[1]), "se": float(np.sqrt(cov[1, 1])), "ols": float(beta[1]), "method": "2sls",
                    "first_stage_f": float(((rss_restricted - rss_full) / q) / (rss_full / max(len(y) - Z.shape[1], 1)))})
    return out


def price_elasticity(df: pd.DataFrame, quantity: str, price: str, controls: list | None = None, group: str | None = None, instrument: list | None = None,
                     unit_cost: float | None = None, date: str | None = None) -> dict:
    """Elasticidad-precio de la demanda con un modelo log-log: un 1 % más de precio cambia la cantidad en `elasticity` %. Con `group` calcula una por grupo;
    con `unit_cost` busca el precio que maximiza el margen (con elasticidad constante, el margen se maximiza en coste·e/(1+e) si e < -1)."""
    controls = [c for c in (controls or []) if c not in (quantity, price)]
    instrument = list(instrument or [])
    cols = [quantity, price, *controls, *instrument] + ([group] if group else [])
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    for c in [quantity, price, *controls, *instrument]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"'{c}' must be a numeric column.")
    frame = df[cols].dropna()
    frame = frame[(frame[quantity] > 0) & (frame[price] > 0)]
    if len(frame) < 30:
        raise ValueError("At least 30 rows with a positive price and quantity are needed.")
    if frame[price].nunique() < 5 or frame[price].std() / frame[price].mean() < 0.02:
        raise ValueError("The price barely changes in this data, so its effect cannot be estimated.")
    main = _elasticity(frame, quantity, price, controls, instrument or None)
    z = stats.norm.ppf(0.975)
    e, se = main["elasticity"], main["se"]
    interval = (e - z * se, e + z * se)
    groups = None
    if group:
        rows = []
        for key, part in frame.groupby(group, observed=True):
            if len(part) >= 30 and part[price].nunique() >= 5:
                try:
                    r = _elasticity(part, quantity, price, controls, instrument or None)
                except np.linalg.LinAlgError:
                    continue
                rows.append({group: key, "elasticity": r["elasticity"], "low": r["elasticity"] - z * r["se"], "high": r["elasticity"] + z * r["se"], "rows": r["n"]})
        groups = pd.DataFrame(rows).sort_values("elasticity").reset_index(drop=True) if rows else pd.DataFrame(columns=[group, "elasticity", "low", "high", "rows"])
    notes = []
    if e > 0 and not instrument:
        notes.append("The estimate is positive, which usually means price was raised when demand was strong (not that higher prices sell more). "
                     "Give an instrument (a cost or a tax that moves price but not demand) or use a price test.")
    elif e > 0:
        notes.append("The estimate is still positive with the instrument: check that it really moves the price and affects demand only through it.")
    if instrument and main.get("first_stage_f", 99) < 10:
        notes.append(f"Weak instrument (first-stage F = {main['first_stage_f']:.1f}): the estimate is unreliable.")
    if instrument and "ols" in main and abs(main["ols"] - e) > 2 * se:
        notes.append(f"Ordinary regression gave {main['ols']:+.2f}: price is not independent of demand here, and the instrument changes the answer a lot.")
    if interval[0] < 0 < interval[1]:
        notes.append("The range includes zero: the data cannot tell whether price changes demand.")
    if not controls:
        notes.append("No controls were given: seasons, promotions and competitors that move with price can distort the estimate.")
    reference_price, reference_quantity = float(frame[price].mean()), float(frame[quantity].mean())
    changes = np.linspace(-0.3, 0.3, 61)
    quantity_curve = reference_quantity * (1 + changes) ** e
    cost = float(unit_cost) if unit_cost is not None else 0.0
    revenue = reference_price * (1 + changes) * quantity_curve
    margin = (reference_price * (1 + changes) - cost) * quantity_curve
    best_price = None
    if unit_cost is not None and e < -1:
        best_price = float(cost * e / (1 + e))
    elif unit_cost is not None:
        notes.append("Demand is not elastic enough (elasticity above -1) for a margin-maximising price to exist in this model: it would keep rising.")
    objective = margin if unit_cost is not None else revenue
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=changes, y=objective, mode="lines", name="Margin" if unit_cost is not None else "Revenue"))
    fig.add_vline(x=0, line_dash="dot")
    fig.update_layout(title=f"{'Margin' if unit_cost is not None else 'Revenue'} if the price changed (elasticity held constant)", xaxis_title="Change in price", xaxis_tickformat=".0%",
                      yaxis_title="Per period")
    verdict = "elastic: raising the price lowers revenue" if e < -1 else "inelastic: raising the price raises revenue but sells less" if e < 0 else "positive: see the warning"
    return {"elasticity": e, "se": se, "interval": interval, "n": main["n"], "method": main["method"], "groups": groups, "first_stage_f": main.get("first_stage_f"),
            "ols": main.get("ols", e), "reference": (reference_price, reference_quantity), "optimal_price": best_price, "curve": pd.DataFrame({"change": changes, "quantity": quantity_curve,
            "revenue": revenue, "margin": margin}), "notes": notes, "figure": fig,
            "summary": (f"A 1% higher price changes the quantity by {e:+.2f}% (95% range {interval[0]:+.2f} to {interval[1]:+.2f}): demand is {verdict}."
                        + (f" With a unit cost of {cost:g}, the margin-maximising price is about {best_price:,.4g} (today's average is {reference_price:,.4g})." if best_price else ""))}
