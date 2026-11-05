"""Análisis completo de un experimento A/B: antes de fiarse del resultado, comprobar que el experimento está bien hecho.

  * `srm`: ¿llegó a cada grupo la proporción de gente que se había planeado? Si no, algo falló al repartir o al registrar y el resultado no es fiable.
  * `cuped`: reduce el ruido usando lo que cada persona hacía antes del experimento, para ver efectos más pequeños con la misma gente.
  * `effect_over_time`: ¿se mantiene el efecto o se apaga (novedad) o crece (aprendizaje)?
  * `segment_effects`: el efecto en cada segmento, con la corrección por mirar muchos a la vez.
  * `analyze_experiment`: todo junto con una conclusión que se niega a dar un ganador si el reparto está roto.
Sirve con resultados numéricos y con 0/1 (conversión). Con más de dos grupos compara cada uno con el de control y corrige por comparaciones múltiples (Holm)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from kikolens.analyze.stattests import benjamini_hochberg

SRM_ALPHA = 0.001
MIN_PER_GROUP = 20


def _groups(df: pd.DataFrame, group: str, outcome: str, control, pre: str | None = None, date: str | None = None) -> tuple:
    for c in (group, outcome) + tuple(x for x in (pre, date) if x):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError(f"The outcome '{outcome}' must be numeric (use 0 and 1 for conversions).")
    if pre and not pd.api.types.is_numeric_dtype(df[pre]):
        raise ValueError(f"The pre-experiment column '{pre}' must be numeric.")
    data = df.dropna(subset=[group, outcome] + ([pre] if pre else []) + ([date] if date else [])).copy()
    data[group] = data[group].astype(str)
    control = str(control)
    arms = sorted(data[group].unique())
    if control not in arms:
        raise ValueError(f"The control group '{control}' is not a value of '{group}'. Groups: {', '.join(arms)}.")
    if len(arms) < 2:
        raise ValueError("The experiment needs at least two groups.")
    small = [a for a in arms if (data[group] == a).sum() < MIN_PER_GROUP]
    if small:
        raise ValueError(f"Group(s) {', '.join(small)} have fewer than {MIN_PER_GROUP} rows with data.")
    return data, control, [a for a in arms if a != control]


def _binary(s: pd.Series) -> bool:
    return set(pd.unique(s.dropna())) <= {0, 1, 0.0, 1.0, True, False}


def _diff(a: np.ndarray, b: np.ndarray, binary: bool) -> dict:
    """Diferencia b - a con su intervalo del 95 % y su p-valor (z para proporciones, Welch para medias)."""
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    if binary:
        se = np.sqrt(ma * (1 - ma) / na + mb * (1 - mb) / nb)
        pooled = (a.sum() + b.sum()) / (na + nb)
        se0 = np.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
        z = (mb - ma) / se0 if se0 > 0 else 0.0
        p = float(2 * stats.norm.sf(abs(z)))
        crit = stats.norm.ppf(0.975)
    else:
        va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
        se = np.sqrt(va + vb)
        dof = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1)) if se > 0 else na + nb - 2
        t = (mb - ma) / se if se > 0 else 0.0
        p = float(2 * stats.t.sf(abs(t), dof))
        crit = stats.t.ppf(0.975, dof)
    return {"control": float(ma), "treatment": float(mb), "lift": float(mb - ma), "relative": float((mb - ma) / ma) if ma else float("nan"),
            "low": float(mb - ma - crit * se), "high": float(mb - ma + crit * se), "se": float(se), "p_value": p, "n_control": na, "n_treatment": nb}


# ── Reparto ──────────────────────────────────────────────────────────────────
def srm(df: pd.DataFrame, group: str, expected: dict | None = None, alpha: float = SRM_ALPHA) -> dict:
    """Prueba de desajuste de la muestra: chi-cuadrado entre lo observado y el reparto planeado (por defecto, a partes iguales)."""
    if group not in df.columns:
        raise ValueError(f"Column '{group}' not found.")
    counts = df[group].dropna().astype(str).value_counts().sort_index()
    if len(counts) < 2:
        raise ValueError("The experiment needs at least two groups.")
    if expected:
        shares = {str(k): float(v) for k, v in expected.items()}
        missing = set(counts.index) ^ set(shares)
        if missing:
            raise ValueError(f"The expected split must list exactly the groups found ({', '.join(counts.index)}).")
        total = sum(shares.values())
        if total <= 0 or any(v < 0 for v in shares.values()):
            raise ValueError("The expected shares must be positive.")
        plan = pd.Series({k: v / total for k, v in shares.items()}).reindex(counts.index)
    else:
        plan = pd.Series(1.0 / len(counts), index=counts.index)
    n = int(counts.sum())
    exp = plan * n
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    p = float(stats.chi2.sf(chi2, len(counts) - 1))
    table = pd.DataFrame({"group": counts.index, "rows": counts.values, "share": counts.values / n, "expected share": plan.values, "expected rows": exp.values,
                          "difference": counts.values - exp.values})
    broken = p < alpha
    notes = []
    if broken:
        notes.append("The split is not what was planned. Common causes: a bug that drops users from one group, a redirect or slow page that loses people, bots filtered unevenly, or "
                     "assignment happening after something that depends on the variant. Fix this before reading any result.")
    elif p < 0.05:
        notes.append("The split is slightly off (p below 0.05 but above the usual alarm level of 0.001): keep an eye on it.")
    return {"table": table, "chi2": chi2, "p_value": p, "broken": bool(broken), "rows": n, "notes": notes,
            "summary": (f"The split of {n:,} rows is " + (f"NOT what was planned (p = {p:.2g}): the result cannot be trusted." if broken else f"consistent with the plan (p = {p:.2g}).")),
            "figure": go.Figure([go.Bar(x=table["group"], y=table["share"], name="Observed"), go.Bar(x=table["group"], y=table["expected share"], name="Planned")]).update_layout(
                barmode="group", title="Share of rows by group", yaxis_tickformat=".0%")}


# ── CUPED ────────────────────────────────────────────────────────────────────
def cuped(df: pd.DataFrame, group: str, outcome: str, pre: str, control) -> dict:
    """Ajusta el resultado con lo que cada fila hacía antes del experimento (`pre`): `Y' = Y - theta (X - media de X)`. Mismo efecto, menos ruido."""
    data, control, arms = _groups(df, group, outcome, control, pre)
    binary = _binary(data[outcome])
    y, x = data[outcome].to_numpy(float), data[pre].to_numpy(float)
    if np.std(x) == 0:
        raise ValueError(f"The pre-experiment column '{pre}' is constant, so it cannot reduce noise.")
    theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    adjusted = y - theta * (x - x.mean())
    data = data.assign(_adjusted=adjusted)
    corr = float(np.corrcoef(y, x)[0, 1])
    rows = []
    for arm in arms:
        a, b = data[data[group] == control], data[data[group] == arm]
        raw = _diff(a[outcome].to_numpy(float), b[outcome].to_numpy(float), binary)
        adj = _diff(a["_adjusted"].to_numpy(float), b["_adjusted"].to_numpy(float), False)
        balance = stats.ttest_ind(a[pre], b[pre], equal_var=False).pvalue
        rows.append({"group": arm, "raw lift": raw["lift"], "raw low": raw["low"], "raw high": raw["high"], "raw p": raw["p_value"], "adjusted lift": adj["lift"], "adjusted low": adj["low"],
                     "adjusted high": adj["high"], "adjusted p": adj["p_value"], "interval narrower by": 1 - (adj["high"] - adj["low"]) / (raw["high"] - raw["low"]) if raw["high"] > raw["low"] else 0.0,
                     "pre-period balance p": float(balance)})
    table = pd.DataFrame(rows)
    reduction = 1 - float(np.var(adjusted, ddof=1) / np.var(y, ddof=1))
    notes = []
    if abs(corr) < 0.3:
        notes.append(f"The pre-experiment column correlates only {corr:+.2f} with the outcome, so it removes little noise ({reduction:.0%}). A column closer to the outcome (the same measure before the test) works best.")
    if (table["pre-period balance p"] < 0.01).any():
        notes.append("The pre-experiment values differ between groups more than chance explains: check how people were assigned. CUPED corrects for it, but the imbalance itself is a warning.")
    return {"table": table, "theta": theta, "correlation": corr, "variance_reduction": reduction, "sample_saving": reduction, "notes": notes,
            "summary": (f"Using '{pre}' (correlation {corr:+.2f}) removes {reduction:.0%} of the variance: the same data gives intervals about {1 - np.sqrt(1 - reduction):.0%} narrower, "
                        f"equivalent to {reduction:.0%} fewer rows needed for the same precision.")}


# ── En el tiempo y por segmento ──────────────────────────────────────────────
def effect_over_time(df: pd.DataFrame, group: str, outcome: str, control, date: str, treatment=None, freq: str = "W") -> dict:
    """Efecto por periodo, con su intervalo, y si la tendencia sube o baja (los efectos de novedad se apagan con el tiempo)."""
    data, control, arms = _groups(df, group, outcome, control, None, date)
    arm = str(treatment) if treatment is not None else arms[0]
    if arm not in arms:
        raise ValueError(f"'{arm}' is not a treatment group. Groups: {', '.join(arms)}.")
    data[date] = pd.to_datetime(data[date], errors="coerce")
    data = data.dropna(subset=[date])
    data = data[data[group].isin([control, arm])]
    binary = _binary(data[outcome])
    period = data[date].dt.to_period(freq)
    rows = []
    for key, part in data.groupby(period):
        a, b = part[part[group] == control][outcome].to_numpy(float), part[part[group] == arm][outcome].to_numpy(float)
        if len(a) >= 10 and len(b) >= 10:
            d = _diff(a, b, binary)
            rows.append({"period": key.to_timestamp(), "lift": d["lift"], "low": d["low"], "high": d["high"], "control rows": len(a), "treatment rows": len(b)})
    if len(rows) < 3:
        raise ValueError(f"At least 3 periods with 10 rows per group are needed (found {len(rows)}); use a shorter period.")
    table = pd.DataFrame(rows)
    k = np.arange(len(table))
    slope = stats.linregress(k, table["lift"])
    cumulative = _diff(data[data[group] == control][outcome].to_numpy(float), data[data[group] == arm][outcome].to_numpy(float), binary)
    notes = []
    if slope.pvalue < 0.05:
        direction = "shrinking" if np.sign(slope.slope) != np.sign(table["lift"].mean()) else "growing"
        notes.append(f"The effect is {direction} over time (trend p = {slope.pvalue:.3f}): " + (
            "a novelty effect that fades is common; wait for it to settle before deciding." if direction == "shrinking" else "users may be learning the change; the final effect may be larger."))
    if len(table) < 5:
        notes.append("Few periods: trends over time are hard to tell from noise.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table["period"], y=table["high"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=table["period"], y=table["low"], mode="lines", line=dict(width=0), fill="tonexty", name="95% range", opacity=0.3))
    fig.add_trace(go.Scatter(x=table["period"], y=table["lift"], mode="lines+markers", name=f"{arm} minus {control}"))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title="Effect by period", yaxis_title="Difference")
    return {"table": table, "overall": cumulative, "trend_p": float(slope.pvalue), "trend_slope": float(slope.slope), "notes": notes, "figure": fig, "treatment": arm,
            "summary": f"Overall {cumulative['lift']:+.4g} ({cumulative['low']:+.4g} to {cumulative['high']:+.4g}); the weekly effect ranges from {table['lift'].min():+.4g} to {table['lift'].max():+.4g}"
                       + (f", with a clear trend (p = {slope.pvalue:.3f})." if slope.pvalue < 0.05 else ", with no clear trend.")}


def segment_effects(df: pd.DataFrame, group: str, outcome: str, control, segments: list, treatment=None, min_rows: int = 30) -> dict:
    """El efecto dentro de cada valor de cada segmento y si difiere del efecto en el resto de las filas. Mirar muchos segmentos da falsos positivos, así que
    se muestra la p corregida (Benjamini-Hochberg)."""
    data, control, arms = _groups(df, group, outcome, control)
    arm = str(treatment) if treatment is not None else arms[0]
    data = data[data[group].isin([control, arm])]
    binary = _binary(data[outcome])
    overall = _diff(data[data[group] == control][outcome].to_numpy(float), data[data[group] == arm][outcome].to_numpy(float), binary)
    rows = []
    for seg in segments:
        if seg not in data.columns:
            raise ValueError(f"Column '{seg}' not found.")
        if data[seg].nunique() > 30:
            raise ValueError(f"'{seg}' has more than 30 values; group it first.")
        labels = data[seg].astype(str)
        for value, part in data.groupby(labels):
            a, b = part[part[group] == control][outcome].to_numpy(float), part[part[group] == arm][outcome].to_numpy(float)
            rest = data[labels != value]
            ra, rb = rest[rest[group] == control][outcome].to_numpy(float), rest[rest[group] == arm][outcome].to_numpy(float)
            if len(a) >= min_rows and len(b) >= min_rows and len(ra) >= min_rows and len(rb) >= min_rows:
                d, r = _diff(a, b, binary), _diff(ra, rb, binary)
                gap = d["lift"] - r["lift"]
                z = gap / np.sqrt(d["se"] ** 2 + r["se"] ** 2) if d["se"] or r["se"] else 0.0
                rows.append({"segment": seg, "value": value, "rows": len(a) + len(b), "lift": d["lift"], "low": d["low"], "high": d["high"], "p (vs no effect)": d["p_value"],
                             "lift in the rest": r["lift"], "gap": gap, "p": float(2 * stats.norm.sf(abs(z)))})
    if not rows:
        raise ValueError(f"No segment has {min_rows} rows in both groups.")
    table = pd.DataFrame(rows)
    table["p adjusted"] = benjamini_hochberg(table["p"].to_numpy())
    table["differs from the rest"] = (table["p adjusted"] < 0.05)
    table = table.sort_values("p adjusted", ignore_index=True)       # `p` y `p adjusted` comparan cada segmento con el resto de filas, no con cero
    notes = []
    hits = table[table["differs from the rest"]]
    if len(hits) == 0:
        notes.append(f"No segment stands out once the {len(table)} comparisons are accounted for: the differences you see are what chance alone would give.")
    else:
        notes.append(f"{len(hits)} segment(s) stand out after correcting for {len(table)} comparisons. Treat them as leads to test, not as proven: they were found by looking.")
    return {"table": table, "overall": overall, "treatment": arm, "notes": notes,
            "summary": f"Overall {overall['lift']:+.4g}; " + (f"the segment that stands out most is {hits.iloc[0]['segment']} = {hits.iloc[0]['value']} ({hits.iloc[0]['lift']:+.4g})."
                                                              if len(hits) else "no segment is clearly different from the rest.")}


# ── Conjunto ─────────────────────────────────────────────────────────────────
def analyze_experiment(df: pd.DataFrame, group: str, outcome: str, control, pre: str | None = None, date: str | None = None, segments: list | None = None,
                       expected: dict | None = None) -> dict:
    """Comprueba el reparto, calcula el efecto de cada grupo frente al control (con CUPED si das `pre`) y, si hay fecha y segmentos, mira la estabilidad y la heterogeneidad."""
    data, control, arms = _groups(df, group, outcome, control, pre, date)
    binary = _binary(data[outcome])
    check = srm(data, group, expected)
    y = data[outcome].to_numpy(float)
    work, theta = data, None
    if pre:
        x = data[pre].to_numpy(float)
        if np.std(x) > 0:
            theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
            work = data.assign(_y=y - theta * (x - x.mean()))
    use = "_y" if theta is not None else outcome
    rows = []
    for arm in arms:
        a, b = work[work[group] == control][use].to_numpy(float), work[work[group] == arm][use].to_numpy(float)
        d = _diff(a, b, binary and use == outcome)
        raw = _diff(data[data[group] == control][outcome].to_numpy(float), data[data[group] == arm][outcome].to_numpy(float), binary)
        sd = float(data[outcome].std(ddof=1))
        mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * np.sqrt(2 * sd ** 2 / min(raw["n_control"], raw["n_treatment"])) if sd > 0 else float("nan")
        rows.append({"group": arm, "control mean": raw["control"], "mean": raw["treatment"], "lift": d["lift"], "relative": d["lift"] / raw["control"] if raw["control"] else float("nan"),
                     "low": d["low"], "high": d["high"], "p": d["p_value"], "rows": raw["n_treatment"], "smallest effect detectable": float(mde)})
    table = pd.DataFrame(rows)
    if len(table) > 1:
        order = np.argsort(table["p"].to_numpy())
        m = len(table)
        holm = np.empty(m)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * table["p"].iloc[i])
            holm[i] = min(1.0, running)
        table["p adjusted"] = holm
    p_col = "p adjusted" if "p adjusted" in table.columns else "p"
    notes = list(check["notes"])
    out = {"table": table, "srm": check, "control": control, "binary": binary, "cuped_theta": theta, "notes": notes}
    if pre:
        out["cuped"] = cuped(data, group, outcome, pre, control)
        notes += out["cuped"]["notes"]
    if date:
        try:
            out["over_time"] = effect_over_time(data, group, outcome, control, date)
            notes += out["over_time"]["notes"]
        except ValueError as e:
            notes.append(f"Effect over time was not computed: {e}")
    if segments:
        try:
            out["segments"] = segment_effects(data, group, outcome, control, segments)
            notes += out["segments"]["notes"]
        except ValueError as e:
            notes.append(f"Segments were not computed: {e}")
    best = table.sort_values("lift", ascending=False).iloc[0]
    significant = bool(best[p_col] < 0.05)
    small = significant is False and best["smallest effect detectable"] > 0 and abs(best["lift"]) < best["smallest effect detectable"]
    if check["broken"]:
        verdict = "Do not decide from this experiment: the split of users is broken, so the groups may not be comparable."
    elif significant and best["lift"] > 0:
        verdict = f"{best['group']} beats {control}: {best['lift']:+.4g} ({best['low']:+.4g} to {best['high']:+.4g}), p = {best[p_col]:.3g}."
    elif significant:
        verdict = f"{best['group']} is worse than {control}: {best['lift']:+.4g} ({best['low']:+.4g} to {best['high']:+.4g}), p = {best[p_col]:.3g}."
    elif small:
        verdict = (f"No clear difference, and this experiment could only detect effects of about {best['smallest effect detectable']:.3g} or more: a smaller real effect would not show. "
                   "It is not evidence that there is no effect.")
    else:
        verdict = "No clear difference between the groups."
    out.update({"verdict": verdict, "summary": verdict + (" " + check["summary"] if not check["broken"] else ""),
                "figure": _forest(table, control)})
    return out


def _forest(table: pd.DataFrame, control) -> go.Figure:
    fig = go.Figure(go.Scatter(x=table["lift"], y=table["group"], mode="markers", error_x=dict(type="data", symmetric=False, array=table["high"] - table["lift"], arrayminus=table["lift"] - table["low"])))
    fig.add_vline(x=0, line_dash="dot")
    fig.update_layout(title=f"Difference from {control} with its 95% range", xaxis_title="Difference", yaxis_title="Group")
    return fig
