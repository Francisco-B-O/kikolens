"""Pruebas A/B secuenciales y bandidos. Una prueba clásica solo vale si se mira una vez, al final; mirar cada día y parar cuando sale
«significativo» dispara los falsos positivos. El test de razón de verosimilitud secuencial con mezcla (mSPRT) es válido aunque se
mire en cada observación: su p-valor «siempre válido» nunca baja de lo que permite el nivel alfa por haber mirado mucho.
También simula un bandido de Thompson, que reparte más tráfico a lo que va ganando mientras aprende."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


def msprt(control, treatment, alpha: float = 0.05, tau: float | None = None, min_pairs: int = 20) -> dict:
    """mSPRT sobre dos flujos de observaciones en orden de llegada. Se emparejan la k-ésima del control y la k-ésima del
    tratamiento. `tau` es la desviación típica del efecto que se espera (por defecto, un 20 % de la desviación de la diferencia entre pares); la validez no depende de ella, solo la rapidez con que se detecta."""
    a = np.asarray(pd.to_numeric(pd.Series(control), errors="coerce").dropna(), float)
    b = np.asarray(pd.to_numeric(pd.Series(treatment), errors="coerce").dropna(), float)
    n_total = min(len(a), len(b))
    if n_total < min_pairs:
        raise ValueError(f"At least {min_pairs} observations per group are needed; the smaller group has {n_total}.")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5.")
    a, b = a[:n_total], b[:n_total]
    diff = b - a
    n = np.arange(1, n_total + 1)
    running = np.cumsum(diff) / n
    # varianza por observación estimada con lo visto hasta cada momento (con n-1); no se decide nada antes de `min_pairs`
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.where(n > 1, (np.cumsum(diff ** 2) - n * running ** 2) / np.maximum(n - 1, 1), 1.0)
    var = np.maximum(var, 1e-12)
    scale = float(np.std(diff, ddof=1)) or 1.0
    tau2 = (tau if tau is not None else 0.2 * scale) ** 2      # efecto típico esperado: un 20 % de la desviación de la diferencia
    log_lam = 0.5 * np.log(var / (var + n * tau2)) + (n ** 2 * tau2 * running ** 2) / (2 * var * (var + n * tau2))
    log_lam = np.where(n >= min_pairs, log_lam, 0.0)
    always_valid = np.minimum.accumulate(np.minimum(1.0, np.exp(-log_lam)))
    crossed = np.where(log_lam >= np.log(1 / alpha))[0]
    stop = int(crossed[0]) + 1 if len(crossed) else None
    se = np.sqrt(var / n)
    z = stats.norm.ppf(1 - alpha / 2)
    # intervalo de confianza siempre válido: los theta que el mismo test no rechazaría; es más ancho que el clásico, a cambio de poder mirar siempre
    width = np.sqrt(2 * var * (var + n * tau2) / (n ** 2 * tau2) * (np.log(1 / alpha) + 0.5 * np.log((var + n * tau2) / var)))
    final = int(stop or n_total) - 1
    result = {"n": n_total, "stopped_at": stop, "decision": "treatment better" if stop and running[final] > 0 else
              "control better" if stop else "keep collecting", "difference": float(running[final]),
              "interval": (float(running[final] - width[final]), float(running[final] + width[final])),
              "p_value": float(always_valid[final]), "log_lambda": log_lam, "p_path": always_valid, "means": running, "alpha": alpha,
              "classic_p": float(2 * stats.norm.sf(abs(running[-1] / se[-1]))),
              "figure": sequential_figure(n, log_lam, alpha, stop)}
    result["summary"] = _summary(result, z)
    return result


def _summary(r: dict, z: float) -> str:
    if r["stopped_at"]:
        return (f"The test can stop after {r['stopped_at']:,} observations per group: {r['decision']} "
                f"(difference {r['difference']:+.4g}, always-valid range {r['interval'][0]:+.4g} to {r['interval'][1]:+.4g}).")
    return (f"After {r['n']:,} observations per group the evidence is not enough to stop (always-valid p = {r['p_value']:.3f}). "
            f"Keep collecting; looking at the data again costs nothing in this test.")


def sequential_figure(n: np.ndarray, log_lam: np.ndarray, alpha: float, stop) -> go.Figure:
    fig = go.Figure(go.Scatter(x=n, y=log_lam, mode="lines", name="Evidence for a difference"))
    fig.add_hline(y=float(np.log(1 / alpha)), line_dash="dash", annotation_text=f"Enough to stop (alpha {alpha})")
    if stop:
        fig.add_vline(x=stop, line_dash="dot")
    fig.update_layout(title="Evidence builds up as data arrive (log of the likelihood ratio)", xaxis_title="Observations per group",
                      yaxis_title="Evidence")
    return fig


def from_frame(df: pd.DataFrame, group: str, outcome: str, control, order: str | None = None, **kwargs) -> dict:
    """mSPRT desde una tabla: `order` es la columna (fecha) que da el orden de llegada; si no, el de las filas."""
    for c in (group, outcome) + ((order,) if order else ()):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    data = df.sort_values(order, kind="stable") if order else df
    data = data[[group, outcome]].dropna()
    data[group] = data[group].astype(str)
    levels = sorted(data[group].unique())
    if str(control) not in levels:
        raise ValueError(f"'{control}' is not a value of '{group}'.")
    others = [v for v in levels if v != str(control)]
    if len(others) != 1:
        raise ValueError("The group column must have exactly two values.")
    outcome_values = data[outcome]
    if not pd.api.types.is_numeric_dtype(outcome_values):
        raise ValueError(f"'{outcome}' must be numeric (use 0/1 for conversions).")
    return msprt(outcome_values[data[group] == str(control)], outcome_values[data[group] == others[0]], **kwargs)


# ── Planificación por simulación ─────────────────────────────────────────────
def planning(baseline: float, effect: float, alpha: float = 0.05, runs: int = 200, max_n: int = 20000, seed: int = 0) -> dict:
    """Cuántas observaciones por grupo necesita de media el test secuencial frente a uno de tamaño fijo, simulando conversiones."""
    from kikolens.analyze.stattests import sample_size_proportions
    if not 0 < baseline < 1 or not 0 < baseline + effect < 1 or effect == 0:
        raise ValueError("The baseline rate and the rate with the effect must both be between 0 and 1, and the effect cannot be 0.")
    rng = np.random.default_rng(seed)
    fixed = sample_size_proportions(baseline, abs(effect), alpha)
    stops, false_alarms = [], 0
    scale = np.sqrt(2 * baseline * (1 - baseline))
    for _ in range(runs):
        a = rng.random(max_n) < baseline
        b = rng.random(max_n) < baseline + effect
        r = msprt(a, b, alpha, tau=abs(effect) if abs(effect) > 0 else scale)
        stops.append(r["stopped_at"] if r["stopped_at"] else max_n)
    for _ in range(runs):                         # sin efecto real: ¿con qué frecuencia se declara uno por error?
        a = rng.random(5000) < baseline
        b = rng.random(5000) < baseline
        false_alarms += msprt(a, b, alpha, tau=abs(effect))["stopped_at"] is not None
    stops = np.array(stops)
    return {"fixed_n": int(fixed), "median_stop": int(np.median(stops)), "p90_stop": int(np.quantile(stops, 0.9)),
            "share_stopped": float(np.mean(stops < max_n)), "false_positive_rate": false_alarms / runs,
            "summary": (f"A fixed-size test needs {fixed:,} per group. The sequential test stopped after a median of {int(np.median(stops)):,} "
                        f"(90% of runs by {int(np.quantile(stops, 0.9)):,}). With no real effect it raised a false alarm in "
                        f"{false_alarms / runs:.1%} of runs, at alpha {alpha:.0%}, even though it was checked after every observation.")}


# ── Bandido de Thompson ──────────────────────────────────────────────────────
def thompson(rates, n: int = 5000, seed: int = 0, batch: int = 50) -> dict:
    """Simula repartir n visitantes entre variantes con las tasas de conversión reales `rates`, con Thompson sampling frente al reparto fijo."""
    rates = np.asarray(rates, float)
    if len(rates) < 2 or len(rates) > 8 or ((rates <= 0) | (rates >= 1)).any():
        raise ValueError("Give between 2 and 8 conversion rates, each strictly between 0 and 1.")
    if n < 200:
        raise ValueError("Simulate at least 200 visitors.")
    rng = np.random.default_rng(seed)
    k = len(rates)
    wins, losses = np.ones(k), np.ones(k)
    shown, converted, share = np.zeros(k, int), 0, []
    for start in range(0, n, batch):
        size = min(batch, n - start)
        choice = np.argmax(rng.beta(wins, losses, size=(size, k)), axis=1)
        hits = rng.random(size) < rates[choice]
        for arm in range(k):
            mask = choice == arm
            shown[arm] += mask.sum()
            wins[arm] += hits[mask].sum()
            losses[arm] += (~hits[mask]).sum()
        converted += int(hits.sum())
        share.append(shown / shown.sum())
    fixed = float(rates.mean() * n)
    best = float(rates.max() * n)
    fig = go.Figure()
    for arm in range(k):
        fig.add_trace(go.Scatter(x=np.arange(1, len(share) + 1) * batch, y=[s[arm] for s in share], mode="lines", stackgroup="one",
                                 name=f"Variant {arm + 1} ({rates[arm]:.1%})"))
    fig.update_layout(title="Share of visitors each variant received as the bandit learned", xaxis_title="Visitors", yaxis_tickformat=".0%")
    return {"conversions": converted, "fixed_split": fixed, "perfect": best, "extra_conversions": converted - fixed,
            "regret": best - converted, "share": pd.DataFrame(share, columns=[f"Variant {i + 1}" for i in range(k)]), "final_allocation": shown / shown.sum(),
            "figure": fig,
            "summary": (f"The bandit got {converted:,.0f} conversions against {fixed:,.0f} for an even split ({converted - fixed:+,.0f}); "
                        f"always showing the best variant would have given {best:,.0f}. It gave the best one {shown[int(np.argmax(rates))] / shown.sum():.0%} of the visitors.")}
