"""¿Por qué cambió esta cifra? Compara una métrica entre dos periodos y reparte el cambio entre los valores de cada columna
(región, canal, producto...). Para una suma, cada valor aporta la diferencia de su suma. Para una media, el cambio se separa en
efecto de mezcla (cambió el peso del valor) y efecto de tasa (cambió su media), y ambos suman el cambio total exactamente."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from kikolens.utils.dtypes import categorical_columns

AGGREGATIONS = ("sum", "mean", "count")
MAX_VALUES = 40


def default_periods(dates: pd.Series):
    """Último tercio del tiempo observado frente al tercio anterior. Devuelve ((inicio, fin), (inicio, fin)) con fin exclusivo."""
    d = pd.to_datetime(dates, errors="coerce").dropna()
    if d.nunique() < 6:
        raise ValueError("At least 6 different dates are needed to compare two periods.")
    start, end = d.min(), d.max() + pd.Timedelta(days=1)
    third = (end - start) / 3
    return (start + third, start + 2 * third), (start + 2 * third, end)


def _slice(df: pd.DataFrame, dates: pd.Series, period) -> pd.DataFrame:
    lo, hi = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    if hi <= lo:
        raise ValueError("A period must end after it starts.")
    return df[(dates >= lo) & (dates < hi)]


def _total(frame: pd.DataFrame, metric: str, agg: str) -> float:
    if agg == "count":
        return float(len(frame))
    values = frame[metric]
    return float(values.sum()) if agg == "sum" else (float(values.mean()) if len(values) else np.nan)


def explain_dimension(before: pd.DataFrame, after: pd.DataFrame, metric: str, dimension: str, agg: str) -> pd.DataFrame:
    """Aporte de cada valor de `dimension` al cambio total."""
    def labels(frame):
        column = frame[dimension]
        return column.astype(str).where(column.notna(), "(empty)")

    b, a = labels(before), labels(after)
    keys = sorted(set(b) | set(a))
    rows = []
    if agg in ("sum", "count"):
        for key in keys:
            vb = _total(before[b == key], metric, agg)
            va = _total(after[a == key], metric, agg)
            rows.append({"value": key, "before": vb, "after": va, "change": va - vb, "contribution": va - vb})
    else:
        n_b, n_a = max(len(before), 1), max(len(after), 1)
        mean_b = _total(before, metric, agg)
        for key in keys:
            wb, wa = (b == key).sum() / n_b, (a == key).sum() / n_a
            sb, sa = before.loc[b == key, metric], after.loc[a == key, metric]
            mb = float(sb.mean()) if len(sb) else None
            ma = float(sa.mean()) if len(sa) else None
            mb = ma if mb is None else mb          # un valor que no existía antes: todo su efecto es de mezcla
            ma = mb if ma is None else ma
            mix, rate = (wa - wb) * (mb - mean_b), wa * (ma - mb)
            rows.append({"value": key, "before": float(sb.mean()) if len(sb) else np.nan,
                         "after": float(sa.mean()) if len(sa) else np.nan, "weight before": wb, "weight after": wa,
                         "mix effect": mix, "rate effect": rate, "change": mix + rate, "contribution": mix + rate})
    table = pd.DataFrame(rows)
    total = table["contribution"].sum()
    table["share of change"] = table["contribution"] / total if total else np.nan
    return table.reindex(table["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def why_changed(df: pd.DataFrame, metric: str | None, date: str, before=None, after=None, dimensions=None, agg: str = "sum",
                top: int = 8) -> dict:
    """Compara dos periodos. Sin `before`/`after` se usan los dos últimos tercios del tiempo observado."""
    if date not in df.columns:
        raise ValueError(f"Column '{date}' not found.")
    if agg not in AGGREGATIONS:
        raise ValueError(f"agg must be one of {AGGREGATIONS}.")
    if agg != "count":
        if metric not in df.columns:
            raise ValueError(f"Column '{metric}' not found.")
        if not pd.api.types.is_numeric_dtype(df[metric]):
            raise ValueError(f"'{metric}' must be numeric.")
    dates = pd.to_datetime(df[date], errors="coerce")
    if dates.notna().sum() == 0:
        raise ValueError(f"'{date}' has no valid dates.")
    if (before is None) != (after is None):
        raise ValueError("Give both periods or neither.")
    if before is None:
        before, after = default_periods(dates)
    frame_b, frame_a = _slice(df, dates, before), _slice(df, dates, after)
    if len(frame_b) < 5 or len(frame_a) < 5:
        raise ValueError("Each period needs at least 5 rows.")
    if dimensions is None:
        dimensions = [c for c in categorical_columns(df) if 2 <= df[c].nunique() <= MAX_VALUES and c != date]
    dimensions = [d for d in dimensions if d in df.columns and d != metric]
    if not dimensions:
        raise ValueError("There is no column with a few distinct values (2 to 40) to break the change down.")

    total_b, total_a = _total(frame_b, metric, agg), _total(frame_a, metric, agg)
    change = total_a - total_b
    tables, ranking = {}, []
    for dim in dimensions:
        t = explain_dimension(frame_b, frame_a, metric, dim, agg)
        tables[dim] = t
        # cuánto se concentra el cambio en pocos valores: el aporte del mayor entre la suma de aportes absolutos
        absolute = t["contribution"].abs().sum()
        ranking.append({"dimension": dim, "concentration": float(t["contribution"].abs().iloc[0] / absolute) if absolute else 0.0,
                        "top value": t["value"].iloc[0], "top contribution": float(t["contribution"].iloc[0])})
    ranking = pd.DataFrame(ranking).sort_values("concentration", ascending=False).reset_index(drop=True)
    return {"before": before, "after": after, "total_before": total_b, "total_after": total_a, "change": change,
            "relative_change": change / total_b if total_b else np.nan, "tables": tables, "ranking": ranking,
            "pairs": _pairs(frame_b, frame_a, metric, ranking["dimension"].head(3).tolist(), agg, top) if agg != "mean" else None,
            "summary": _summary(metric, agg, total_b, total_a, change, tables, ranking)}


def _pairs(before, after, metric, dimensions, agg, top) -> pd.DataFrame | None:
    """Combinaciones de dos columnas que más contribuyen (solo para sumas y recuentos)."""
    if len(dimensions) < 2:
        return None
    out = []
    for i, d1 in enumerate(dimensions):
        for d2 in dimensions[i + 1:]:
            def group(frame):
                g = frame.groupby([frame[d1].astype(str), frame[d2].astype(str)])
                return g.size() if agg == "count" else g[metric].sum()
            diff = group(after).sub(group(before), fill_value=0)
            for (v1, v2), value in diff.items():
                out.append({"combination": f"{d1} = {v1}, {d2} = {v2}", "change": float(value)})
    table = pd.DataFrame(out)
    if table.empty:
        return None
    return table.reindex(table["change"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


def _summary(metric, agg, total_b, total_a, change, tables, ranking) -> str:
    name = "the number of rows" if agg == "count" else f"{agg} of {metric}"
    direction = "rose" if change > 0 else "fell" if change < 0 else "did not change"
    rel = f" ({change / total_b:+.1%})" if total_b else ""
    text = f"The {name} {direction} from {total_b:,.4g} to {total_a:,.4g}{rel}."
    if not change:
        return text
    best = ranking.iloc[0]
    table = tables[best["dimension"]]
    lead = table.iloc[0]
    share = lead["contribution"] / change
    text += (f" Most of it is explained by '{best['dimension']}': '{lead['value']}' alone accounts for {share:.0%} of the change "
             f"({lead['contribution']:+,.4g}).")
    if agg == "mean" and "mix effect" in table:
        mix, rate = table["mix effect"].sum(), table["rate effect"].sum()
        text += f" Across '{best['dimension']}', {mix:+,.4g} comes from a change in the mix and {rate:+,.4g} from changes inside each value."
    return text


def waterfall_figure(result: dict, dimension: str, top: int = 8) -> go.Figure:
    table = result["tables"][dimension]
    shown = table.head(top)
    rest = table["contribution"].iloc[top:].sum()
    labels = ["Before"] + shown["value"].tolist() + (["Other values"] if len(table) > top else []) + ["After"]
    values = [result["total_before"]] + shown["contribution"].tolist() + ([rest] if len(table) > top else []) + [result["total_after"]]
    measure = ["absolute"] + ["relative"] * (len(labels) - 2) + ["total"]
    fig = go.Figure(go.Waterfall(x=labels, y=values, measure=measure, connector={"line": {"width": 1}}))
    fig.update_layout(title=f"What moved it, by {dimension}", showlegend=False)
    return fig
