"""Análisis de clientes y comportamiento: retención por cohortes, segmentación RFM, embudos de conversión y reglas de
asociación (qué se compra junto)."""
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PERIODS = {"Day": "D", "Week": "W", "Month": "M"}


def _dates(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")
    out = pd.to_datetime(df[col], errors="coerce")
    if out.notna().sum() == 0:
        raise ValueError(f"'{col}' has no valid dates.")
    return out


# ── Cohortes ─────────────────────────────────────────────────────────────────
def cohort_retention(df: pd.DataFrame, user: str, date: str, period: str = "Month", max_periods: int = 12) -> dict:
    """Para cada cohorte (periodo de la primera actividad), qué parte de sus usuarios sigue activa n periodos después."""
    if period not in PERIODS:
        raise ValueError(f"Period must be one of {list(PERIODS)}.")
    if user not in df.columns:
        raise ValueError(f"Column '{user}' not found.")
    freq = PERIODS[period]
    data = pd.DataFrame({"user": df[user], "d": _dates(df, date)}).dropna()
    if data["user"].nunique() < 5:
        raise ValueError("At least 5 distinct users are needed.")
    data["p"] = data["d"].dt.to_period(freq)
    data["cohort"] = data.groupby("user")["p"].transform("min")
    data["age"] = (data["p"] - data["cohort"]).apply(lambda x: x.n)
    active = data.groupby(["cohort", "age"])["user"].nunique().unstack(fill_value=0)
    active = active.loc[:, [c for c in active.columns if c <= max_periods]]
    sizes = active[0]
    matrix = active.div(sizes, axis=0)
    # las celdas posteriores a la última fecha observada no son "0 % de retención", son desconocidas
    last = data["p"].max()
    for cohort in matrix.index:
        possible = (last - cohort).n
        matrix.loc[cohort, [a for a in matrix.columns if a > possible]] = np.nan
    matrix.index = matrix.index.astype(str)
    average = matrix.mean(axis=0, skipna=True)
    return {"matrix": matrix, "sizes": sizes.rename(index=str), "average": average, "figure": _retention_figure(matrix),
            "summary": _retention_summary(matrix, average, period)}


def _retention_summary(matrix, average, period) -> str:
    text = f"{len(matrix)} cohorts."
    first = average.get(1, np.nan)
    if not np.isnan(first):
        text += f" On average {first:.0%} of users come back in the next {period.lower()}"
        third = average.get(3, np.nan)
        text += f" and {third:.0%} after three." if not np.isnan(third) else "."
    return text


def _retention_figure(matrix: pd.DataFrame) -> go.Figure:
    values = matrix.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(z=values, x=[str(c) for c in matrix.columns], y=list(matrix.index), colorscale="Oranges",
                               zmin=0, zmax=1, hoverongaps=False, text=np.round(values * 100), texttemplate="%{text:.0f}"))
    fig.update_layout(title="Share of each cohort still active (%)", xaxis_title="Periods since first activity",
                      yaxis_autorange="reversed")
    return fig


# ── RFM ──────────────────────────────────────────────────────────────────────
SEGMENTS = [("Champions", lambda r, f, m: r >= 4 and f >= 4), ("Loyal", lambda r, f, m: f >= 4),
            ("Big spenders", lambda r, f, m: m >= 4 and r >= 3), ("New or promising", lambda r, f, m: r >= 4 and f <= 2),
            ("At risk", lambda r, f, m: r <= 2 and f >= 3), ("Lost", lambda r, f, m: r <= 2)]


def _score(series: pd.Series, ascending_better: bool) -> pd.Series:
    ranked = series.rank(method="first", ascending=ascending_better)
    return pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def rfm(df: pd.DataFrame, customer: str, date: str, amount: str, today=None) -> dict:
    """Recencia, frecuencia y valor de cada cliente, con una nota de 1 a 5 en cada uno y un segmento."""
    for col in (customer, amount):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    money = pd.to_numeric(df[amount], errors="coerce")
    data = pd.DataFrame({"c": df[customer], "d": _dates(df, date), "m": money}).dropna()
    if data["c"].nunique() < 10:
        raise ValueError("At least 10 distinct customers are needed.")
    today = pd.Timestamp(today) if today is not None else data["d"].max() + pd.Timedelta(days=1)
    table = data.groupby("c").agg(recency=("d", lambda s: (today - s.max()).days), frequency=("d", "size"),
                                  monetary=("m", "sum"))
    table["R"] = _score(table["recency"], False)
    table["F"] = _score(table["frequency"], True)
    table["M"] = _score(table["monetary"], True)

    def label(row):
        for name, rule in SEGMENTS:
            if rule(row["R"], row["F"], row["M"]):
                return name
        return "Needs attention"

    table["segment"] = table.apply(label, axis=1)
    summary = table.groupby("segment").agg(customers=("segment", "size"), recency=("recency", "mean"),
                                           frequency=("frequency", "mean"), monetary=("monetary", "sum"))
    summary["share of value"] = summary["monetary"] / summary["monetary"].sum()
    summary = summary.sort_values("monetary", ascending=False)
    top = summary.index[0]
    return {"customers": table.reset_index().rename(columns={"c": customer}), "segments": summary.reset_index(),
            "summary": f"'{top}' is {summary.iloc[0]['share of value']:.0%} of the value, from "
                       f"{int(summary.iloc[0]['customers']):,} of {len(table):,} customers."}


# ── Embudo ───────────────────────────────────────────────────────────────────
def funnel(df: pd.DataFrame, user: str, stage: str, order: list) -> dict:
    """Cuántos usuarios llegan a cada etapa y qué parte se pierde entre una y la siguiente."""
    for col in (user, stage):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    if len(order) < 2:
        raise ValueError("A funnel needs at least two stages.")
    seen = df[[user, stage]].dropna()
    seen[stage] = seen[stage].astype(str)
    order = [str(s) for s in order]
    reached, previous = [], None
    for name in order:
        users = set(seen.loc[seen[stage] == name, user])
        previous = users if previous is None else previous & users   # llegó a esta etapa habiendo pasado por las anteriores
        reached.append(len(previous))
    if reached[0] == 0:
        raise ValueError(f"Nobody reached the first stage '{order[0]}'.")
    out = pd.DataFrame({"stage": order, "users": reached})
    out["of first stage"] = out["users"] / reached[0]
    out["of previous"] = (out["users"] / out["users"].shift(1).replace(0, np.nan)).fillna(1.0)
    worst = out.iloc[1:].sort_values("of previous").iloc[0]
    fig = go.Figure(go.Funnel(y=out["stage"], x=out["users"], textinfo="value+percent initial"))
    fig.update_layout(title="Users reaching each stage")
    return {"table": out, "figure": fig,
            "summary": f"{out['users'].iloc[-1]:,} of {reached[0]:,} users ({out['of first stage'].iloc[-1]:.0%}) complete the "
                       f"funnel. The biggest loss is before '{worst['stage']}', where {1 - worst['of previous']:.0%} drop out."}


# ── Reglas de asociación ─────────────────────────────────────────────────────
def association_rules(df: pd.DataFrame, transaction: str, item: str, min_support: float = 0.02, min_confidence: float = 0.3,
                      max_size: int = 3, limit: int = 50) -> dict:
    """Apriori: qué artículos aparecen juntos más de lo esperado. `lift` > 1 indica que se acompañan."""
    for col in (transaction, item):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    if not 0 < min_support < 1 or not 0 < min_confidence <= 1:
        raise ValueError("Support must be between 0 and 1, and confidence between 0 and 1.")
    baskets = df[[transaction, item]].dropna().astype(str).groupby(transaction)[item].agg(frozenset)
    n = len(baskets)
    if n < 20:
        raise ValueError("At least 20 transactions are needed.")
    min_count = max(2, int(np.ceil(min_support * n)))
    singles = pd.Series([i for b in baskets for i in b]).value_counts()
    counts = {frozenset([i]): int(c) for i, c in singles.items() if c >= min_count}
    level = dict(counts)
    for size in range(2, max_size + 1):
        items = sorted({i for s in level for i in s})
        candidates = {frozenset(c) for c in combinations(items, size)
                      if all(frozenset(sub) in counts for sub in combinations(c, size - 1))}
        level = {}
        for basket in baskets:
            if len(basket) < size:
                continue
            for cand in candidates:
                if cand <= basket:
                    level[cand] = level.get(cand, 0) + 1
        level = {k: v for k, v in level.items() if v >= min_count}
        if not level:
            break
        counts.update(level)
    rules = []
    for itemset, count in counts.items():
        if len(itemset) < 2:
            continue
        for k in range(1, len(itemset)):
            for left in map(frozenset, combinations(sorted(itemset), k)):
                right = itemset - left
                confidence = count / counts[left]
                if confidence >= min_confidence:
                    rules.append({"if bought": ", ".join(sorted(left)), "then also": ", ".join(sorted(right)),
                                  "support": count / n, "confidence": confidence,
                                  "lift": confidence / (counts[right] / n)})
    table = pd.DataFrame(rules, columns=["if bought", "then also", "support", "confidence", "lift"])
    table = table.sort_values(["lift", "confidence"], ascending=False).head(limit).reset_index(drop=True)
    best = (f" Strongest: {table.iloc[0]['if bought']} -> {table.iloc[0]['then also']} (lift {table.iloc[0]['lift']:.1f})."
            if len(table) else " Lower the support or confidence to find more.")
    return {"rules": table, "transactions": n, "summary": f"{len(rules)} rules over {n:,} baskets." + best}
