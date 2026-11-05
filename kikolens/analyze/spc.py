"""Control estadístico de procesos: ¿está estable el proceso o ha cambiado algo?

  * `control_chart`: cartas de individuales (I-MR), medias de subgrupos (X-S) y de atributos (p, u, c), con las reglas clásicas de señal (Western Electric / Nelson).
    Los límites se calculan con un tramo inicial («fase I») que se puede depurar, y se congelan para vigilar el resto.
  * `capability`: Cp, Cpk, Pp, Ppk y el porcentaje que se saldría de especificación.
  * `cusum` y `ewma`: detectan cambios pequeños y sostenidos que una carta de Shewhart tarda mucho en ver.
Los límites de control describen lo que el proceso hace, no lo que se quiere que haga; los de especificación son los segundos."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import special, stats

MIN_POINTS = 12
D2 = 1.1283791670955126                     # d2 para rangos móviles de 2 (2 / sqrt(pi))
CHARTS = ("individuals", "xbar", "p", "u", "c")
DEFAULT_RULES = (1, 4)
# Probabilidad por punto de que cada regla salte en un proceso estable y normal (simulada); la 4 depende de cuántos seguidos se pidan.
FALSE_ALARM = {1: 0.0027, 2: 0.0019, 3: 0.0040, 5: 0.0027, 6: 0.0042}


def c4(n: int) -> float:
    """Factor que corrige el sesgo de la desviación típica muestral en subgrupos de tamaño n."""
    return math.sqrt(2.0 / (n - 1)) * math.exp(special.gammaln(n / 2) - special.gammaln((n - 1) / 2))


# ── Reglas ───────────────────────────────────────────────────────────────────
def rule_violations(z: np.ndarray, sustained: int = 8) -> pd.DataFrame:
    """Reglas de señal sobre las distancias al centro medidas en desviaciones (`z`): 1 punto a más de 3; 2 de 3 a más de 2 del mismo lado; 4 de 5 a más de 1 del mismo lado;
    `sustained` seguidos del mismo lado; 6 seguidos subiendo o bajando; 14 alternando."""
    rows = []
    n = len(z)
    side = np.sign(z)
    for i in range(n):
        if abs(z[i]) > 3:
            rows.append((i, 1, "A point is beyond the 3-sigma limit"))
        if i >= 2:
            w = z[i - 2: i + 1]
            for s in (1, -1):
                if np.sum(s * w > 2) >= 2 and s * z[i] > 2:
                    rows.append((i, 2, "2 of 3 points are beyond 2 sigma on the same side"))
        if i >= 4:
            w = z[i - 4: i + 1]
            for s in (1, -1):
                if np.sum(s * w > 1) >= 4 and s * z[i] > 1:
                    rows.append((i, 3, "4 of 5 points are beyond 1 sigma on the same side"))
        if i >= sustained - 1:
            w = side[i - sustained + 1: i + 1]
            if (w > 0).all() or (w < 0).all():
                rows.append((i, 4, f"{sustained} points in a row on the same side of the centre"))
        if i >= 5:
            d = np.diff(z[i - 5: i + 1])
            if (d > 0).all() or (d < 0).all():
                rows.append((i, 5, "6 points in a row going up or down"))
        if i >= 13:
            d = np.diff(z[i - 13: i + 1])
            if (np.sign(d[1:]) * np.sign(d[:-1]) < 0).all():
                rows.append((i, 6, "14 points in a row alternating up and down"))
    return pd.DataFrame(rows, columns=["position", "rule", "meaning"]).drop_duplicates(["position", "rule"], ignore_index=True)


# ── Cartas ───────────────────────────────────────────────────────────────────
def _limits_individuals(x: np.ndarray) -> tuple:
    mr = np.abs(np.diff(x))
    sigma = float(mr.mean() / D2) if len(mr) else float(np.std(x, ddof=1))
    return float(x.mean()), sigma


def control_chart(df: pd.DataFrame, value: str, chart: str = "individuals", order: str | None = None, subgroup: str | None = None, size: str | None = None,
                  baseline: int | None = None, clean_baseline: bool = True, rules: tuple = DEFAULT_RULES, sustained: int = 9) -> dict:
    """Carta de control. `individuals`: un valor por punto. `xbar`: medias de los subgrupos de `subgroup`. `p`: `value` son los defectuosos y `size` los examinados.
    `u`: `value` son los defectos y `size` las unidades. `c`: `value` son los defectos por unidad de tamaño fijo. `baseline` = cuántos puntos iniciales fijan los límites.
    `rules` son las reglas de señal activas (1: un punto fuera de 3 sigma; 2: 2 de 3 fuera de 2 sigma; 3: 4 de 5 fuera de 1 sigma; 4: `sustained` seguidos del mismo lado;
    5: 6 seguidos subiendo o bajando; 6: 14 alternando). Cuantas más reglas, antes se ve un cambio y más falsas alarmas hay."""
    if chart not in CHARTS:
        raise ValueError(f"chart must be one of {', '.join(CHARTS)}.")
    rules = tuple(rules)
    if not rules or not set(rules) <= set(range(1, 7)):
        raise ValueError("rules must be numbers from 1 to 6.")
    if not 6 <= sustained <= 12:
        raise ValueError("sustained must be between 6 and 12.")
    for c in (value, order, subgroup, size):
        if c and c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    if not pd.api.types.is_numeric_dtype(df[value]):
        raise ValueError(f"'{value}' must be numeric.")
    data = df.copy()
    if order:
        data = data.sort_values(order)
    data = data.dropna(subset=[value] + ([size] if size else []))
    labels = None
    if chart == "xbar":
        if not subgroup:
            raise ValueError("A chart of subgroup means needs the column that identifies each subgroup.")
        grouped = data.groupby(subgroup, sort=order is None)[value]
        stat = pd.DataFrame({"mean": grouped.mean(), "std": grouped.std(ddof=1), "n": grouped.size()}).dropna()
        stat = stat[stat["n"] >= 2]
        if len(stat) < MIN_POINTS:
            raise ValueError(f"At least {MIN_POINTS} subgroups of 2 or more values are needed (found {len(stat)}).")
        y, n_i, labels = stat["mean"].to_numpy(float), stat["n"].to_numpy(float), list(stat.index)
        s_values = stat["std"].to_numpy(float)
    else:
        if chart in ("p", "u") and not size:
            raise ValueError(f"The '{chart}' chart needs a column with how many were examined (size=).")
        y = data[value].to_numpy(float)
        n_i = data[size].to_numpy(float) if size else np.ones(len(y))
        s_values = None
        if size and (n_i <= 0).any():
            raise ValueError("The sizes must be positive.")
        if chart in ("p", "u", "c") and (y < 0).any():
            raise ValueError("Counts cannot be negative.")
        if chart == "p" and (y > n_i).any():
            raise ValueError("A proportion chart cannot have more defective than examined.")
    counts = y.copy()
    if chart in ("p", "u"):
        y = counts / n_i                                              # se dibuja la proporción (o los defectos por unidad), no el recuento
    if len(y) < MIN_POINTS:
        raise ValueError(f"At least {MIN_POINTS} points are needed (found {len(y)}).")
    base = np.ones(len(y), bool)
    if baseline:
        if not MIN_POINTS <= baseline <= len(y):
            raise ValueError(f"baseline must be between {MIN_POINTS} and the number of points ({len(y)}).")
        base[baseline:] = False
    removed = []
    skew0 = float(stats.skew(y[base])) if chart == "individuals" and base.sum() >= 20 else 0.0

    def compute(mask: np.ndarray) -> tuple:
        if chart == "individuals":
            centre, sigma = _limits_individuals(y[mask])
            return np.full(len(y), centre), np.full(len(y), sigma)
        if chart == "xbar":
            sbar = float(np.average(s_values[mask], weights=n_i[mask] - 1))
            nn = int(round(n_i[mask].mean()))
            centre = float(np.average(y[mask], weights=n_i[mask]))
            sigma_within = sbar / c4(max(nn, 2))
            return np.full(len(y), centre), sigma_within / np.sqrt(n_i)
        if chart == "p":
            p = float(counts[mask].sum() / n_i[mask].sum())
            return np.full(len(y), p), np.sqrt(p * (1 - p) / n_i)
        if chart == "u":
            u = float(counts[mask].sum() / n_i[mask].sum())
            return np.full(len(y), u), np.sqrt(u / n_i)
        c = float(y[mask].mean())
        return np.full(len(y), c), np.full(len(y), math.sqrt(c))

    centre, sigma = compute(base)
    if clean_baseline:
        for _ in range(5):                                            # fase I: se quitan los puntos fuera de límites y se recalcula
            out = base & (np.abs(y - centre) > 3 * sigma)
            if not out.any() or (base & ~out).sum() < MIN_POINTS:
                break
            removed += list(np.where(out)[0])
            base = base & ~out
            centre, sigma = compute(base)
    z = (y - centre) / np.where(sigma > 0, sigma, np.nan)
    z = np.nan_to_num(z)
    lcl, ucl = centre - 3 * sigma, centre + 3 * sigma
    if chart in ("p", "u", "c"):
        lcl = np.maximum(lcl, 0)
    if chart == "p":
        ucl = np.minimum(ucl, 1)
    signals = rule_violations(z, sustained)
    signals = signals[signals["rule"].isin(rules)].reset_index(drop=True)
    signals["value"] = y[signals["position"]] if len(signals) else []
    signals["limit_phase"] = np.where(signals["position"] < (baseline or len(y)), "baseline", "monitoring") if len(signals) else []
    flagged = np.zeros(len(y), bool)
    if len(signals):
        flagged[signals["position"].to_numpy()] = True
    x_axis = labels if labels is not None else (list(data[order]) if order else list(range(1, len(y) + 1)))
    table = pd.DataFrame({"point": x_axis, "value": y, "centre": centre, "lower limit": lcl, "upper limit": ucl, "in baseline": base, "signal": flagged})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=ucl, mode="lines", line=dict(dash="dash", color="crimson"), name="Upper limit"))
    fig.add_trace(go.Scatter(x=x_axis, y=lcl, mode="lines", line=dict(dash="dash", color="crimson"), name="Lower limit"))
    fig.add_trace(go.Scatter(x=x_axis, y=centre, mode="lines", line=dict(color="gray"), name="Centre"))
    fig.add_trace(go.Scatter(x=x_axis, y=y, mode="lines+markers", name=value, marker=dict(color=np.where(flagged, "crimson", "#2F6FB3"), size=7), line=dict(color="#2F6FB3", width=1)))
    if baseline:
        fig.add_vrect(x0=x_axis[0], x1=x_axis[baseline - 1], opacity=0.07, line_width=0, annotation_text="sets the limits")
    names = {"individuals": "Individual values", "xbar": "Means of subgroups", "p": "Proportion", "u": "Defects per unit", "c": "Defects"}
    fig.update_layout(title=f"{names[chart]} control chart", yaxis_title=value)
    after = signals[signals["limit_phase"] == "monitoring"] if baseline and len(signals) else signals
    first = int(after["position"].min()) if len(after) else None
    notes = []
    if removed:
        notes.append(f"{len(removed)} baseline point(s) outside the limits were set aside to compute the limits (positions {', '.join(str(i + 1) for i in removed[:8])}): they look like special causes.")
    if chart == "individuals":
        mr = np.abs(np.diff(y))
        mr_limit = 3.267 * float(np.abs(np.diff(y[base])).mean()) if base.sum() > 2 else None
        if mr_limit is not None and (mr > mr_limit).any():
            notes.append(f"{int((mr > mr_limit).sum())} point-to-point jump(s) are larger than the moving-range limit: sudden changes of level.")
    if chart in ("p", "u") and len(n_i) and n_i.max() > 3 * n_i.min():
        notes.append("The sample sizes vary a lot, so the limits change from point to point (that is correct).")
    if chart == "p" and centre[0] * n_i.min() < 5 and centre[0] > 0:
        notes.append("Some samples are so small that fewer than 5 defects are expected: the limits are only approximate.")
    if abs(skew0) > 1.5:
        notes.append("The values are strongly skewed, so the limits (which assume a symmetric spread) will signal too often on one side; consider a transformation.")
    per_point = sum(2 * 0.5 ** sustained if r == 4 else FALSE_ALARM[r] for r in rules)
    expected = per_point * len(y)
    notes.append(f"Active rules: {', '.join(map(str, rules))}. In a perfectly stable process they would give a false alarm on about {per_point:.1%} of the points "
                 f"(around {expected:.1f} in these {len(y)}), so a lone signal is a reason to look, not proof of a change.")
    verdict = ("in statistical control: no signals" if not len(signals) else f"NOT in control: {len(signals)} signal(s), the first at point {first + 1}")
    return {"table": table, "signals": signals, "centre": float(centre[0]), "sigma": float(np.nanmean(sigma)), "chart": chart, "in_control": not len(signals), "removed": removed,
            "notes": notes, "figure": fig, "summary": f"The process is {verdict}. Centre {centre[0]:.4g}, limits {np.nanmean(lcl):.4g} to {np.nanmean(ucl):.4g}."}


# ── Capacidad ────────────────────────────────────────────────────────────────
def capability(values, lsl: float | None = None, usl: float | None = None, target: float | None = None) -> dict:
    """Índices de capacidad frente a los límites de especificación. Cp/Cpk usan la variación a corto plazo (rangos móviles); Pp/Ppk, la desviación de todo el conjunto."""
    if lsl is None and usl is None:
        raise ValueError("Give at least one specification limit (lsl or usl).")
    if lsl is not None and usl is not None and lsl >= usl:
        raise ValueError("The lower specification limit must be below the upper one.")
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(x) < 30:
        raise ValueError(f"At least 30 values are needed for a capability study (found {len(x)}).")
    mean, overall = float(x.mean()), float(x.std(ddof=1))
    within = float(np.abs(np.diff(x)).mean() / D2)
    if overall == 0:
        raise ValueError("The values do not vary.")

    def indices(sigma):
        cp = (usl - lsl) / (6 * sigma) if lsl is not None and usl is not None else None
        up = (usl - mean) / (3 * sigma) if usl is not None else None
        low = (mean - lsl) / (3 * sigma) if lsl is not None else None
        cpk = min(v for v in (up, low) if v is not None)
        return cp, cpk

    cp, cpk = indices(within)
    pp, ppk = indices(overall)
    below = float((x < lsl).mean()) if lsl is not None else 0.0
    above = float((x > usl).mean()) if usl is not None else 0.0
    expected_low = float(stats.norm.cdf((lsl - mean) / overall)) if lsl is not None else 0.0
    expected_high = float(stats.norm.sf((usl - mean) / overall)) if usl is not None else 0.0
    normal_p = float(stats.shapiro(x[:5000]).pvalue)
    cpm = None
    if target is not None and lsl is not None and usl is not None:
        cpm = (usl - lsl) / (6 * math.sqrt(overall ** 2 + (mean - target) ** 2))
    band = "capable" if cpk >= 1.33 else "marginal" if cpk >= 1.0 else "not capable"
    notes = []
    if normal_p < 0.01:
        notes.append(f"The values do not look normal (Shapiro-Wilk p = {normal_p:.3g}), so the percentages out of specification that come from the indices may be wrong; the observed percentages are reliable.")
    if cp is not None and cp - cpk > 0.3:
        notes.append(f"The process is capable in spread (Cp {cp:.2f}) but off-centre (Cpk {cpk:.2f}): centring it would give most of the improvement.")
    if ppk < cpk - 0.25:
        notes.append("Overall variation is clearly larger than short-term variation: the process drifts over time, so control it first (see the control chart).")
    return {"cp": cp, "cpk": cpk, "pp": pp, "ppk": ppk, "cpm": cpm, "mean": mean, "within_sigma": within, "overall_sigma": overall, "observed_out": below + above, "expected_out": expected_low + expected_high,
            "ppm_expected": (expected_low + expected_high) * 1e6, "band": band, "normal_p": normal_p, "n": len(x), "notes": notes,
            "table": pd.DataFrame([{"index": "Cp", "value": cp}, {"index": "Cpk", "value": cpk}, {"index": "Pp", "value": pp}, {"index": "Ppk", "value": ppk}] + ([{"index": "Cpm", "value": cpm}] if cpm else [])),
            "figure": _capability_figure(x, lsl, usl, target),
            "summary": (f"Cpk {cpk:.2f} ({band}); Ppk {ppk:.2f}. {above + below:.2%} of the values are outside the specification now"
                        f", and about {(expected_low + expected_high) * 1e6:,.0f} per million would be if the spread is normal.")}


def _capability_figure(x: np.ndarray, lsl, usl, target) -> go.Figure:
    fig = go.Figure(go.Histogram(x=x, histnorm="probability density", nbinsx=40, name="Values"))
    grid = np.linspace(x.min() - x.std(), x.max() + x.std(), 200)
    fig.add_trace(go.Scatter(x=grid, y=stats.norm.pdf(grid, x.mean(), x.std(ddof=1)), name="Normal with the same mean and spread"))
    for value, label in ((lsl, "Lower spec"), (usl, "Upper spec"), (target, "Target")):
        if value is not None:
            fig.add_vline(x=value, line_dash="dash", annotation_text=label)
    fig.update_layout(title="Values against the specification", showlegend=True)
    return fig


# ── Cambios pequeños ─────────────────────────────────────────────────────────
def _baseline(x: np.ndarray, target, sigma, baseline) -> tuple:
    n0 = baseline or max(MIN_POINTS, len(x) // 3)
    ref = x[:n0]
    mu = float(ref.mean()) if target is None else float(target)
    s = float(np.abs(np.diff(ref)).mean() / D2) if sigma is None else float(sigma)
    if s <= 0:
        raise ValueError("The standard deviation is zero: the baseline does not vary.")
    return mu, s, n0


def cusum(values, target: float | None = None, sigma: float | None = None, k: float = 0.5, h: float = 5.0, baseline: int | None = None) -> dict:
    """CUSUM tabular: acumula las desviaciones respecto al objetivo (menos una holgura `k` desviaciones) y avisa al pasar `h` desviaciones. Ve cambios de ~1 desviación mucho antes que Shewhart."""
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(x) < MIN_POINTS:
        raise ValueError(f"At least {MIN_POINTS} values are needed (found {len(x)}).")
    if k <= 0 or h <= 0:
        raise ValueError("k and h must be positive.")
    mu, s, n0 = _baseline(x, target, sigma, baseline)
    up, down = np.zeros(len(x)), np.zeros(len(x))
    for i, v in enumerate(x):
        z = (v - mu) / s
        up[i] = max(0.0, (up[i - 1] if i else 0.0) + z - k)
        down[i] = max(0.0, (down[i - 1] if i else 0.0) - z - k)
    signal = (up > h) | (down > h)
    first = int(np.argmax(signal)) if signal.any() else None
    start = None
    if first is not None:                                            # el cambio empezó donde la suma que cruzó el límite salió de cero por última vez
        series = up if up[first] > h else down
        zeros = np.where(series[:first + 1] == 0)[0]
        start = int(zeros[-1] + 1) if len(zeros) else 0
    fig = go.Figure([go.Scatter(y=up, name="Upward shift"), go.Scatter(y=-down, name="Downward shift")])
    fig.add_hline(y=h, line_dash="dash", line_color="crimson")
    fig.add_hline(y=-h, line_dash="dash", line_color="crimson")
    fig.update_layout(title="CUSUM (in standard deviations)", xaxis_title="Point")
    return {"upper": up, "lower": down, "signal_at": first, "shift_started": start, "target": mu, "sigma": s, "k": k, "h": h, "baseline": n0, "figure": fig,
            "notes": ([] if first is None else [f"Direction: {'up' if up[first] > h else 'down'}."]),
            "summary": ("No sustained shift detected." if first is None else
                        f"A sustained shift {'up' if up[first] > h else 'down'} is signalled at point {first + 1}; it started around point {start + 1}.")}


def ewma(values, lam: float = 0.2, width: float = 3.0, target: float | None = None, sigma: float | None = None, baseline: int | None = None) -> dict:
    """Media móvil ponderada exponencialmente: cada punto pesa `lam` y el pasado el resto. Límites que se ensanchan al principio."""
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(x) < MIN_POINTS:
        raise ValueError(f"At least {MIN_POINTS} values are needed (found {len(x)}).")
    if not 0 < lam <= 1 or width <= 0:
        raise ValueError("lam must be in (0, 1] and width positive.")
    mu, s, n0 = _baseline(x, target, sigma, baseline)
    z = np.zeros(len(x))
    prev = mu
    for i, v in enumerate(x):
        prev = lam * v + (1 - lam) * prev
        z[i] = prev
    i = np.arange(1, len(x) + 1)
    spread = width * s * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i)))
    out = np.abs(z - mu) > spread
    first = int(np.argmax(out)) if out.any() else None
    fig = go.Figure([go.Scatter(y=z, name="EWMA"), go.Scatter(y=mu + spread, line=dict(dash="dash", color="crimson"), name="Upper limit"),
                     go.Scatter(y=mu - spread, line=dict(dash="dash", color="crimson"), name="Lower limit"), go.Scatter(y=np.full(len(x), mu), line=dict(color="gray"), name="Target")])
    fig.update_layout(title=f"EWMA (weight {lam:g})", xaxis_title="Point")
    return {"ewma": z, "lower": mu - spread, "upper": mu + spread, "signal_at": first, "target": mu, "sigma": s, "figure": fig, "notes": [], "baseline": n0,
            "summary": "No drift detected." if first is None else f"The average has drifted {'up' if z[first] > mu else 'down'}: signalled at point {first + 1}."}
