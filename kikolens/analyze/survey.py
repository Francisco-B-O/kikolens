"""Análisis de encuestas: escalas de acuerdo (Likert), fiabilidad, NPS, tablas cruzadas y qué pesa más en la satisfacción global.

  * `likert_summary`: por pregunta, media, % de respuestas favorables y desfavorables y distribución; acepta números o textos («Totalmente de acuerdo»).
  * `reliability`: alfa de Cronbach de un grupo de preguntas que miden lo mismo, qué pasa si se quita cada una y cuáles están al revés.
  * `nps`: promotores, pasivos y detractores, con intervalo y comparación entre grupos.
  * `crosstab`: tabla cruzada con residuos que dicen qué celdas son más (o menos) de lo esperado.
  * `key_drivers`: importancia de cada pregunta para la valoración global (pesos relativos de Johnson, que reparten bien el efecto aunque las preguntas se parezcan) frente a lo bien que puntúan.
  * `compare_groups`: qué preguntas puntúan distinto entre grupos, con pruebas para datos ordinales y corrección por mirar muchas.
  * `response_quality`: quién contesta siempre lo mismo y quién contesta en blanco.
Las puntuaciones de una escala son ordinales: la media sirve para comparar, pero el porcentaje favorable cuenta mejor lo que piensa la gente."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from kikolens.analyze.stattests import benjamini_hochberg

SCALES = {
    "agreement": ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    "satisfaction": ["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"],
    "frequency": ["never", "rarely", "sometimes", "often", "always"],
    "quality": ["very poor", "poor", "fair", "good", "excellent"],
    "acuerdo": ["totalmente en desacuerdo", "en desacuerdo", "neutral", "de acuerdo", "totalmente de acuerdo"],
    "satisfaccion": ["muy insatisfecho", "insatisfecho", "neutral", "satisfecho", "muy satisfecho"],
    "frecuencia": ["nunca", "casi nunca", "a veces", "a menudo", "siempre"],
}
ALIASES = {"neither agree nor disagree": "neutral", "neither satisfied nor dissatisfied": "neutral", "ni de acuerdo ni en desacuerdo": "neutral", "ni satisfecho ni insatisfecho": "neutral",
           "regular": "neutral", "average": "neutral", "mostly": "often", "frecuentemente": "a menudo", "casi siempre": "a menudo"}
MIN_RESPONSES = 20


def to_numeric_scale(series: pd.Series) -> tuple:
    """Convierte una pregunta a números. Devuelve (valores, escala usada o None). Los textos de una escala conocida se numeran de 1 a 5."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.astype(float), None
    text = series.dropna().astype(str).str.strip().str.lower().replace(ALIASES)
    for name, labels in SCALES.items():
        if len(text) and text.isin(labels).all():
            mapping = {label: i + 1 for i, label in enumerate(labels)}
            return series.astype(str).str.strip().str.lower().replace(ALIASES).map(mapping).where(series.notna()), name
    numeric = pd.to_numeric(series, errors="coerce")
    if series.notna().sum() and numeric.notna().sum() >= 0.95 * series.notna().sum():
        return numeric, None
    raise ValueError(f"'{series.name if series.name is not None else 'This question'}' is not a numeric scale or a known text scale (agreement, satisfaction, frequency, quality).")


def _items(df: pd.DataFrame, items: list, minimum: int = 1) -> tuple:
    if len(items) < minimum:
        raise ValueError(f"Choose at least {minimum} question{'s' if minimum > 1 else ''}.")
    for c in items:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    data, scales = pd.DataFrame(index=df.index), {}
    for c in items:
        data[c], scales[c] = to_numeric_scale(df[c])
    return data, scales


def _range(data: pd.DataFrame, scale: tuple | None) -> tuple:
    if scale:
        return float(scale[0]), float(scale[1])
    lo, hi = float(np.nanmin(data.to_numpy())), float(np.nanmax(data.to_numpy()))
    return (1.0, 5.0) if hi <= 5 and lo >= 1 else (lo, hi)


def _flip(data: pd.DataFrame, reverse, low: float, high: float) -> pd.DataFrame:
    out = data.copy()
    for c in reverse or []:
        if c not in out.columns:
            raise ValueError(f"'{c}' is not one of the questions chosen.")
        out[c] = low + high - out[c]
    return out


# ── Resumen ──────────────────────────────────────────────────────────────────
def likert_summary(df: pd.DataFrame, items: list, scale: tuple | None = None, reverse: list | None = None) -> dict:
    """Por pregunta: respuestas, media, desviación, % favorable (las dos casillas más altas), % desfavorable (las dos más bajas) y reparto."""
    data, scales = _items(df, items)
    low, high = _range(data, scale)
    if high - low < 2:
        raise ValueError("The scale needs at least three points.")
    data = _flip(data, reverse, low, high)
    points = np.arange(low, high + 1)
    rows = []
    for c in items:
        s = data[c].dropna()
        if len(s) < MIN_RESPONSES:
            raise ValueError(f"'{c}' has {len(s)} answers; at least {MIN_RESPONSES} are needed.")
        n = len(s)
        row = {"question": c, "answers": n, "mean": float(s.mean()), "sd": float(s.std(ddof=1)), "favourable": float((s >= high - 1).mean()), "neutral": float(((s > low + 1) & (s < high - 1)).mean()),
               "unfavourable": float((s <= low + 1).mean())}
        for p in points:
            row[f"% {p:g}"] = float((s == p).mean())
        row["mean low"], row["mean high"] = (float(s.mean() - 1.96 * s.std(ddof=1) / np.sqrt(n)), float(s.mean() + 1.96 * s.std(ddof=1) / np.sqrt(n)))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("favourable", ascending=False, ignore_index=True)
    fig = go.Figure()
    for p in points:
        fig.add_trace(go.Bar(y=table["question"], x=table[f"% {p:g}"], orientation="h", name=f"{p:g}"))
    fig.update_layout(barmode="stack", title="Distribution of answers by question", xaxis_tickformat=".0%", yaxis=dict(autorange="reversed"))
    notes = []
    off = [c for c in items if (data[c].dropna() % 1 != 0).any()]
    if off:
        notes.append(f"Question(s) {', '.join(off)} have decimals: they look like averages, not single answers.")
    outside = [c for c in items if ((data[c].dropna() < low) | (data[c].dropna() > high)).any()]
    if outside:
        notes.append(f"Question(s) {', '.join(outside)} have answers outside the scale {low:g} to {high:g}: check the codes for 'don't know' or 'not applicable'.")
    best, worst = table.iloc[0], table.iloc[-1]
    return {"table": table, "scale": (low, high), "reversed": list(reverse or []), "text_scales": {k: v for k, v in scales.items() if v}, "notes": notes, "figure": fig,
            "summary": f"Most favourable: '{best['question']}' ({best['favourable']:.0%} favourable); least: '{worst['question']}' ({worst['favourable']:.0%} favourable, {worst['unfavourable']:.0%} unfavourable)."}


# ── Fiabilidad ───────────────────────────────────────────────────────────────
def _alpha(data: np.ndarray) -> float:
    k = data.shape[1]
    total_var = data.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - data.var(axis=0, ddof=1).sum() / total_var)) if total_var > 0 else float("nan")


def reliability(df: pd.DataFrame, items: list, reverse: list | None = None, scale: tuple | None = None) -> dict:
    """Alfa de Cronbach (consistencia de preguntas que miden lo mismo), correlación de cada pregunta con el resto y alfa si se quitara. Solo usa a quien contestó todas."""
    data, _ = _items(df, items, minimum=3)
    low, high = _range(data, scale)
    data = _flip(data, reverse, low, high).dropna()
    if len(data) < MIN_RESPONSES:
        raise ValueError(f"Only {len(data)} people answered every question; at least {MIN_RESPONSES} are needed.")
    X = data.to_numpy(float)
    k, n = X.shape[1], X.shape[0]
    alpha = _alpha(X)
    corr = np.corrcoef(X, rowvar=False)
    mean_r = float((corr.sum() - k) / (k * (k - 1)))
    standardized = k * mean_r / (1 + (k - 1) * mean_r)
    # intervalo de Feldt (1965)
    df1, df2 = n - 1, (n - 1) * (k - 1)
    low_ci = float(1 - (1 - alpha) * stats.f.ppf(0.975, df1, df2))
    high_ci = float(1 - (1 - alpha) * stats.f.ppf(0.025, df1, df2))
    rows = []
    for i, c in enumerate(items):
        rest = np.delete(X, i, axis=1)
        rows.append({"question": c, "corrected item-total correlation": float(np.corrcoef(X[:, i], rest.sum(axis=1))[0, 1]), "alpha without it": _alpha(rest), "mean": float(X[:, i].mean())})
    table = pd.DataFrame(rows)
    notes = []
    negative = table[table["corrected item-total correlation"] < 0]["question"].tolist()
    if negative:
        notes.append(f"{', '.join(negative)} correlate negatively with the others: they may be worded the other way round (use reverse=) or measure something else.")
    weak = table[(table["corrected item-total correlation"] < 0.3) & (table["corrected item-total correlation"] >= 0)]["question"].tolist()
    if weak:
        notes.append(f"{', '.join(weak)} hardly move with the rest (correlation under 0.3).")
    better = table[table["alpha without it"] > alpha + 0.02]["question"].tolist()
    if better:
        notes.append(f"Removing {', '.join(better)} would raise alpha noticeably: they pull the group apart.")
    if alpha > 0.95:
        notes.append("Alpha above 0.95 often means questions that repeat each other: the survey could be shorter.")
    if k > 10:
        notes.append("Alpha grows with the number of questions, so a long group can look reliable only because of its length.")
    label = "excellent" if alpha >= 0.9 else "good" if alpha >= 0.8 else "acceptable" if alpha >= 0.7 else "questionable" if alpha >= 0.6 else "poor"
    return {"alpha": alpha, "interval": (low_ci, high_ci), "standardized_alpha": float(standardized), "mean_inter_item_correlation": mean_r, "label": label, "table": table, "n": int(n), "items": k,
            "notes": notes, "summary": f"Cronbach's alpha is {alpha:.2f} ({label}; 95% range {low_ci:.2f} to {high_ci:.2f}) for {k} questions answered by {n:,} people. The average correlation between questions is {mean_r:.2f}."}


# ── NPS ──────────────────────────────────────────────────────────────────────
def _mix_test(data: pd.DataFrame, groups: list) -> tuple:
    """Contraste global de la mezcla de detractores, pasivos y promotores entre grupos; devuelve (p, grupo más extremo)."""
    part = data[data["group"].isin(groups)]
    mix = pd.crosstab(part["group"], pd.cut(part["score"], [-1, 6, 8, 10], labels=["detractors", "passives", "promoters"])).reindex(columns=["detractors", "passives", "promoters"], fill_value=0)
    total = float(mix.to_numpy().sum())
    rows_t, cols_t = mix.sum(axis=1).to_numpy(float), mix.sum(axis=0).to_numpy(float)
    expected = np.outer(rows_t, cols_t) / total
    valid = expected > 0
    chi2 = float(((mix.to_numpy() - expected)[valid] ** 2 / expected[valid]).sum())
    p = float(stats.chi2.sf(chi2, (len(groups) - 1) * (int((cols_t > 0).sum()) - 1))) if len(groups) > 1 else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = (mix.to_numpy() - expected) / np.sqrt(expected * (1 - rows_t[:, None] / total) * (1 - cols_t[None, :] / total))
    extreme = np.nan_to_num(np.abs(resid[:, [0, 2]])).max(axis=1)
    return p, list(mix.index)[int(extreme.argmax())]


def _group_pattern(data: pd.DataFrame) -> str:
    """Qué grupos se salen: si la mezcla no es la misma, se aparta el grupo más extremo y se vuelve a comprobar con el resto, hasta que el resto sea homogéneo."""
    groups = [g for g in data["group"].unique() if (data["group"] == g).sum() >= 10]
    remaining, apart = list(groups), []
    omnibus, _ = _mix_test(data, remaining)
    if omnibus >= 0.05:
        return "The groups do not differ clearly in their mix of promoters, passives and detractors."
    while len(remaining) > 2:
        p, extreme = _mix_test(data, remaining)
        if p >= 0.05:
            break
        apart.append(extreme)
        remaining.remove(extreme)
    if len(remaining) == 2 and _mix_test(data, remaining)[0] < 0.05:
        return f"The groups are not all alike (chi-square p = {omnibus:.2g}): {' and '.join(remaining)} differ from each other" + (f", and {', '.join(apart)} also stand out." if apart else ".")
    return f"The groups are not all alike (chi-square p = {omnibus:.2g}); the one(s) that stand out from the rest: {', '.join(apart)}."


def nps(df: pd.DataFrame, score: str, group: str | None = None) -> dict:
    """Net Promoter Score: % de promotores (9-10) menos % de detractores (0-6). Con grupo, compara cada uno con el resto."""
    if score not in df.columns:
        raise ValueError(f"Column '{score}' not found.")
    if group and group not in df.columns:
        raise ValueError(f"Column '{group}' not found.")
    s = pd.to_numeric(df[score], errors="coerce")
    data = pd.DataFrame({"score": s}).assign(**({"group": df[group].astype(str)} if group else {})).dropna(subset=["score"])
    if len(data) < MIN_RESPONSES:
        raise ValueError(f"At least {MIN_RESPONSES} answers are needed (found {len(data)}).")
    if data["score"].min() < 0 or data["score"].max() > 10:
        raise ValueError("NPS scores go from 0 to 10.")

    def measure(x: pd.Series) -> dict:
        n = len(x)
        p, d = float((x >= 9).mean()), float((x <= 6).mean())
        se = float(np.sqrt(max(p + d - (p - d) ** 2, 0) / n))
        value = (p - d) * 100
        return {"n": n, "promoters": p, "passives": 1 - p - d, "detractors": d, "nps": value, "low": value - 196 * se, "high": value + 196 * se, "se": se * 100}

    overall = measure(data["score"])
    rows = [{"group": "All", **overall}]
    notes = []
    if group:
        for name, part in data.groupby("group"):
            if len(part) >= 10:
                m = measure(part["score"])
                rest = measure(data.loc[data["group"] != name, "score"]) if (data["group"] != name).sum() >= 10 else None
                row = {"group": name, **m}
                if rest:
                    z = (m["nps"] - rest["nps"]) / np.hypot(m["se"], rest["se"]) if m["se"] or rest["se"] else 0.0
                    row["vs the rest"], row["p"] = m["nps"] - rest["nps"], float(2 * stats.norm.sf(abs(z)))
                rows.append(row)
        if len(rows) > 2:
            ps = [r.get("p") for r in rows[1:] if r.get("p") is not None]
            adjusted = benjamini_hochberg(np.array(ps)) if ps else []
            for r, a in zip([r for r in rows[1:] if r.get("p") is not None], adjusted):
                r["p adjusted"] = float(a)
            notes.append(_group_pattern(data))
    if overall["n"] < 100:
        notes.append(f"Only {overall['n']} answers: an NPS moves by ±{196 * overall['se'] / 100:.0f} points just by chance, so do not read small changes.")
    return {"table": pd.DataFrame(rows), "nps": overall["nps"], "interval": (overall["low"], overall["high"]), "promoters": overall["promoters"], "passives": overall["passives"],
            "detractors": overall["detractors"], "n": overall["n"], "notes": notes,
            "figure": go.Figure(go.Bar(x=["Detractors (0-6)", "Passives (7-8)", "Promoters (9-10)"], y=[overall["detractors"], overall["passives"], overall["promoters"]],
                                       marker_color=["#c0392b", "#95a5a6", "#27ae60"])).update_layout(title="Share of answers by type", yaxis_tickformat=".0%"),
            "summary": f"NPS {overall['nps']:+.0f} (95% range {overall['low']:+.0f} to {overall['high']:+.0f}): {overall['promoters']:.0%} promoters, {overall['passives']:.0%} passives and {overall['detractors']:.0%} detractors."}


# ── Tablas cruzadas ──────────────────────────────────────────────────────────
def crosstab(df: pd.DataFrame, row: str, column: str) -> dict:
    """Recuento, porcentaje por fila, chi-cuadrado, V de Cramér y residuos estandarizados ajustados: |valor| > 1,96 marca celdas más o menos frecuentes de lo que daría el azar."""
    for c in (row, column):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    table = pd.crosstab(df[row], df[column])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Both questions need at least two different answers.")
    if table.shape[0] * table.shape[1] > 400:
        raise ValueError("The table has too many cells; group the answers first.")
    n = table.to_numpy().sum()
    if n < MIN_RESPONSES:
        raise ValueError(f"At least {MIN_RESPONSES} answers are needed.")
    observed = table.to_numpy(float)
    rows_total, cols_total = observed.sum(axis=1, keepdims=True), observed.sum(axis=0, keepdims=True)
    expected = rows_total @ cols_total / n
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p = float(stats.chi2.sf(chi2, dof))
    v = float(np.sqrt(chi2 / (n * (min(observed.shape) - 1))))
    adj = (observed - expected) / np.sqrt(expected * (1 - rows_total / n) * (1 - cols_total / n))
    residuals = pd.DataFrame(adj, index=table.index, columns=table.columns)
    notes = []
    if (expected < 5).mean() > 0.2:
        notes.append("More than 20% of the cells expect fewer than 5 answers, so the p-value is approximate; group some answers.")
    cells = [(r, c, residuals.loc[r, c]) for r in residuals.index for c in residuals.columns if abs(residuals.loc[r, c]) > 1.96]
    cells.sort(key=lambda x: -abs(x[2]))
    strength = "negligible" if v < 0.1 else "weak" if v < 0.2 else "moderate" if v < 0.4 else "strong"
    return {"counts": table, "row_percent": table.div(table.sum(axis=1), axis=0), "expected": pd.DataFrame(expected, index=table.index, columns=table.columns), "residuals": residuals,
            "chi2": chi2, "p_value": p, "dof": dof, "cramers_v": v, "strength": strength, "standouts": cells, "notes": notes, "table": table,
            "summary": (f"{row} and {column} are {'related' if p < 0.05 else 'not clearly related'} (chi-square p = {p:.3g}, Cramér's V {v:.2f}: {strength})."
                        + (f" Most unusual cell: {row} = {cells[0][0]} with {column} = {cells[0][1]} ({'more' if cells[0][2] > 0 else 'fewer'} than expected)." if cells else ""))}


# ── Factores clave ───────────────────────────────────────────────────────────
def key_drivers(df: pd.DataFrame, outcome: str, items: list, scale: tuple | None = None) -> dict:
    """Importancia de cada pregunta para la valoración global con pesos relativos (Johnson): reparten el R² entre preguntas aunque estén correlacionadas, y suman ese R².
    Se cruza con lo bien que puntúa cada una para decir qué mejorar primero."""
    data, _ = _items(df, items, minimum=2)
    y, _ = to_numeric_scale(df[outcome]) if outcome in df.columns else (None, None)
    if y is None:
        raise ValueError(f"Column '{outcome}' not found.")
    if outcome in items:
        raise ValueError("The overall rating cannot be one of its own drivers.")
    low, high = _range(data, scale)
    frame = data.assign(_y=y).dropna()
    if len(frame) < max(MIN_RESPONSES, 5 * len(items)):
        raise ValueError(f"At least {max(MIN_RESPONSES, 5 * len(items))} people who answered everything are needed (found {len(frame)}).")
    X = frame[items].to_numpy(float)
    yv = frame["_y"].to_numpy(float)
    if np.std(yv) == 0 or (np.std(X, axis=0) == 0).any():
        raise ValueError("A question or the overall rating has the same answer for everyone.")
    Xs, ys = (X - X.mean(0)) / X.std(0, ddof=1), (yv - yv.mean()) / yv.std(ddof=1)
    n = len(ys)
    Rxx = Xs.T @ Xs / (n - 1)
    rxy = Xs.T @ ys / (n - 1)
    evals, P = np.linalg.eigh(Rxx)
    if evals.min() < 1e-8:
        raise ValueError("Two questions are practically the same answer (perfectly correlated): drop one.")
    Lam = P @ np.diag(np.sqrt(evals)) @ P.T
    beta = np.linalg.solve(Lam, rxy)
    weights = (Lam ** 2) @ (beta ** 2)
    r2 = float(weights.sum())
    betas = np.linalg.solve(Rxx, rxy)
    mean_scores = frame[items].mean().to_numpy()
    table = pd.DataFrame({"question": items, "importance": weights / r2, "share of R2": weights, "correlation with overall": rxy, "standardised effect": betas, "mean score": mean_scores,
                          "favourable": (frame[items] >= high - 1).mean().to_numpy()})
    imp_cut, perf_cut = table["importance"].median(), table["mean score"].median()
    table["priority"] = np.where((table["importance"] >= imp_cut) & (table["mean score"] < perf_cut), "improve first",
                                 np.where((table["importance"] >= imp_cut) & (table["mean score"] >= perf_cut), "keep up", np.where(table["mean score"] < perf_cut, "low priority", "possible excess")))
    table = table.sort_values("importance", ascending=False, ignore_index=True)
    notes = []
    if r2 < 0.2:
        notes.append(f"These questions explain only {r2:.0%} of the overall rating: what really drives it may not be in this list.")
    if (table["standardised effect"] < -0.05).any():
        negatives = table.loc[table["standardised effect"] < -0.05, "question"].tolist()
        notes.append(f"{', '.join(negatives)} have a negative effect once the others are taken into account: they overlap with other questions; read their importance with care.")
    if len(frame) < 150:
        notes.append("Fewer than 150 complete answers: the order of importance can change with a few more people.")
    fig = go.Figure(go.Scatter(x=table["importance"], y=table["mean score"], mode="markers+text", text=table["question"], textposition="top center",
                               marker=dict(size=12, color=np.where(table["priority"] == "improve first", "crimson", "#2F6FB3"))))
    fig.add_vline(x=imp_cut, line_dash="dot")
    fig.add_hline(y=perf_cut, line_dash="dot")
    fig.update_layout(title="Importance against score (bottom right: important and low-scoring)", xaxis_title="Importance (share of what explains the overall rating)", yaxis_title="Average score",
                      xaxis_tickformat=".0%")
    first = table[table["priority"] == "improve first"]
    return {"table": table, "r2": r2, "n": int(n), "notes": notes, "figure": fig,
            "summary": (f"The questions explain {r2:.0%} of '{outcome}'. Most important: '{table.iloc[0]['question']}' ({table.iloc[0]['importance']:.0%} of the explained part)"
                        + (f"; improve first: {', '.join(first['question'].head(3))}." if len(first) else "."))}


# ── Grupos ───────────────────────────────────────────────────────────────────
def compare_groups(df: pd.DataFrame, items: list, group: str) -> dict:
    """Qué preguntas puntúan distinto entre grupos (Mann-Whitney con dos, Kruskal-Wallis con más), con el tamaño del efecto y la corrección de Benjamini-Hochberg."""
    if group not in df.columns:
        raise ValueError(f"Column '{group}' not found.")
    data, _ = _items(df, items, minimum=1)
    labels = df[group].astype(str).where(df[group].notna())
    groups = sorted(g for g in labels.dropna().unique() if (labels == g).sum() >= 10)
    if len(groups) < 2:
        raise ValueError("At least two groups with 10 or more people are needed.")
    rows = []
    for c in items:
        samples = [data.loc[labels == g, c].dropna().to_numpy() for g in groups]
        if min(len(x) for x in samples) < 5 or np.ptp(np.concatenate(samples)) == 0:
            continue
        if len(groups) == 2:
            stat, p = stats.mannwhitneyu(samples[0], samples[1], alternative="two-sided")
            effect = float(2 * stat / (len(samples[0]) * len(samples[1])) - 1)                    # correlación biserial por rangos
            measure = "rank-biserial correlation"
        else:
            stat, p = stats.kruskal(*samples)
            effect = float((stat - len(samples) + 1) / (sum(len(x) for x in samples) - len(samples)))   # epsilon cuadrado
            measure = "epsilon squared"
        row = {"question": c, **{f"mean: {g}": float(np.mean(x)) for g, x in zip(groups, samples)}, "effect": effect, "measure": measure, "p": float(p)}
        rows.append(row)
    if not rows:
        raise ValueError("No question has enough answers in every group.")
    table = pd.DataFrame(rows)
    table["p adjusted"] = benjamini_hochberg(table["p"].to_numpy())
    table["differs"] = table["p adjusted"] < 0.05
    table = table.sort_values("p adjusted", ignore_index=True)
    hits = table[table["differs"]]
    return {"table": table, "groups": groups, "notes": ([] if len(hits) else ["No question differs between the groups once the number of questions compared is accounted for."]),
            "summary": (f"{len(hits)} of {len(table)} question(s) score differently between the groups after correcting for the number compared"
                        + (f"; the clearest is '{hits.iloc[0]['question']}'." if len(hits) else "."))}


# ── Calidad de las respuestas ────────────────────────────────────────────────
def response_quality(df: pd.DataFrame, items: list, time_seconds: str | None = None, min_items: int = 5) -> dict:
    """Quién da la misma respuesta a todo (línea recta), quién deja muchas en blanco y, si hay tiempo, quién contesta demasiado rápido."""
    data, _ = _items(df, items, minimum=min_items)
    answered = data.notna().sum(axis=1)
    straight = (data.nunique(axis=1) == 1) & (answered >= min_items)
    blank = data.isna().mean(axis=1)
    table = pd.DataFrame({"straight-lining": straight, "share blank": blank})
    notes = []
    if time_seconds:
        if time_seconds not in df.columns or not pd.api.types.is_numeric_dtype(df[time_seconds]):
            raise ValueError(f"'{time_seconds}' must be a numeric column with the seconds spent.")
        cut = float(df[time_seconds].median() / 3)
        table["too fast"] = df[time_seconds] < cut
        notes.append(f"'Too fast' means under {cut:.0f} seconds, a third of the median.")
    table["suspect"] = table.drop(columns="share blank").any(axis=1) | (blank > 0.5)
    share_straight = float(straight.mean())
    if share_straight > 0.1:
        notes.append(f"{share_straight:.0%} of the people gave the same answer to every question: scales often invite this; consider mixing in reversed questions.")
    fig = go.Figure(go.Histogram(x=blank, nbinsx=20))
    fig.update_layout(title="Share of questions left blank by person", xaxis_tickformat=".0%")
    return {"table": table, "straight_lining": int(straight.sum()), "mostly_blank": int((blank > 0.5).sum()), "suspect": int(table["suspect"].sum()), "people": int(len(df)),
            "notes": notes, "figure": fig,
            "summary": f"{int(straight.sum()):,} of {len(df):,} people ({share_straight:.1%}) answered every question the same; {int((blank > 0.5).sum()):,} left more than half blank; "
                       f"{int(table['suspect'].sum()):,} are worth a second look."}
