"""Galería de gráficos avanzados: flujos (Sankey), jerarquías (treemap), perfiles (radar), correlaciones agrupadas, red de
correlaciones, calendario, crestas (distribuciones por grupo), coordenadas paralelas y pequeños múltiplos. Cada función
devuelve una figura de Plotly y avisa con un ValueError claro si los datos no sirven para ese gráfico."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde

MAX_LEVELS = 30


def _need(df: pd.DataFrame, *columns) -> None:
    for c in columns:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")


def _numeric(df: pd.DataFrame, columns=None, minimum: int = 2) -> list:
    cols = [c for c in (columns or df.columns) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() > 1]
    if len(cols) < minimum:
        raise ValueError(f"At least {minimum} numeric columns that vary are needed.")
    return cols


def _top_levels(series: pd.Series, top: int) -> pd.Series:
    """Las `top` categorías más frecuentes; el resto se agrupa como «other»."""
    s = series.astype(str).where(series.notna(), "(empty)")
    keep = s.value_counts().head(top).index
    return s.where(s.isin(keep), "other")


# ── Flujos ───────────────────────────────────────────────────────────────────
def sankey(df: pd.DataFrame, source: str, target: str, value: str | None = None, top: int = 8) -> go.Figure:
    """Flujo entre los valores de dos columnas (por número de filas o por la suma de `value`)."""
    _need(df, source, target)
    if source == target:
        raise ValueError("Choose two different columns.")
    if value is not None:
        _need(df, value)
        if not pd.api.types.is_numeric_dtype(df[value]):
            raise ValueError(f"'{value}' must be numeric.")
    frame = pd.DataFrame({"s": _top_levels(df[source], top), "t": _top_levels(df[target], top)})
    frame["w"] = df[value].astype(float).clip(lower=0).fillna(0) if value else 1.0
    flows = frame.groupby(["s", "t"])["w"].sum().reset_index()
    flows = flows[flows["w"] > 0]
    if flows.empty:
        raise ValueError("There is no flow to draw.")
    left = [f"{source}: {v}" for v in sorted(flows["s"].unique())]
    right = [f"{target}: {v}" for v in sorted(flows["t"].unique())]
    labels = left + right
    index = {name: i for i, name in enumerate(labels)}
    fig = go.Figure(go.Sankey(
        node=dict(label=[l.split(": ", 1)[1] for l in labels], pad=14, thickness=16, line=dict(width=0)),
        link=dict(source=[index[f"{source}: {s}"] for s in flows["s"]], target=[index[f"{target}: {t}"] for t in flows["t"]],
                  value=flows["w"].tolist())))
    fig.update_layout(title=f"{source} to {target}" + (f" (sum of {value})" if value else " (rows)"))
    return fig


# ── Jerarquía ────────────────────────────────────────────────────────────────
def treemap(df: pd.DataFrame, path: list, value: str | None = None, top: int = 12) -> go.Figure:
    """Rectángulos anidados: cada nivel de `path` divide al anterior; el área es el número de filas o la suma de `value`."""
    if not path or len(path) > 3:
        raise ValueError("Choose between one and three columns for the levels.")
    _need(df, *path)
    if value is not None:
        _need(df, value)
        if not pd.api.types.is_numeric_dtype(df[value]):
            raise ValueError(f"'{value}' must be numeric.")
    frame = pd.DataFrame({c: _top_levels(df[c], top) for c in path})
    frame["w"] = df[value].astype(float).clip(lower=0).fillna(0) if value else 1.0
    grouped = frame.groupby(path)["w"].sum().reset_index()
    grouped = grouped[grouped["w"] > 0]
    if grouped.empty:
        raise ValueError("There is nothing to draw.")
    ids, labels, parents, values = [], [], [], []
    for depth in range(len(path)):
        part = grouped.groupby(path[: depth + 1])["w"].sum().reset_index()
        for _, row in part.iterrows():
            chain = [str(row[c]) for c in path[: depth + 1]]
            ids.append(" / ".join(chain))
            labels.append(chain[-1])
            parents.append(" / ".join(chain[:-1]))
            values.append(float(row["w"]))
    fig = go.Figure(go.Treemap(ids=ids, labels=labels, parents=parents, values=values, branchvalues="total", textinfo="label+percent root"))
    fig.update_layout(title=f"{' > '.join(path)}" + (f" (sum of {value})" if value else " (rows)"))
    return fig


# ── Perfiles ─────────────────────────────────────────────────────────────────
def radar(df: pd.DataFrame, group: str, metrics: list, top: int = 6) -> go.Figure:
    """Perfil medio de cada grupo en varias medidas, cada una escalada de 0 a 1 para poder compararlas."""
    _need(df, group)
    cols = _numeric(df, metrics, minimum=3)
    if len(cols) > 10:
        raise ValueError("Choose at most 10 measures.")
    labels = _top_levels(df[group], top)
    means = df[cols].groupby(labels).mean()
    lo, hi = df[cols].min(), df[cols].max()
    scaled = (means - lo) / (hi - lo).replace(0, 1)
    fig = go.Figure()
    for name, row in scaled.iterrows():
        fig.add_trace(go.Scatterpolar(r=list(row) + [row.iloc[0]], theta=cols + [cols[0]], name=str(name), fill="toself", opacity=0.45))
    fig.update_layout(title=f"Profile of each {group} (0 = lowest value in the data, 1 = highest)",
                      polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)))
    return fig


# ── Correlaciones ────────────────────────────────────────────────────────────
def clustered_heatmap(df: pd.DataFrame, columns=None, method: str = "spearman") -> go.Figure:
    """Matriz de correlaciones con las columnas reordenadas para que las parecidas queden juntas (agrupación jerárquica)."""
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError("method must be pearson, spearman or kendall.")
    cols = _numeric(df, columns, minimum=3)[:40]
    corr = df[cols].corr(method=method).fillna(0)
    distance = 1 - corr.abs().to_numpy()
    np.fill_diagonal(distance, 0)
    order = leaves_list(linkage(squareform((distance + distance.T) / 2, checks=False), method="average"))
    ordered = corr.iloc[order, order]
    fig = go.Figure(go.Heatmap(z=ordered.to_numpy(), x=list(ordered.columns), y=list(ordered.index), zmin=-1, zmax=1, colorscale="RdBu",
                               reversescale=True, text=np.round(ordered.to_numpy(), 2), texttemplate="%{text}" if len(cols) <= 14 else None))
    fig.update_layout(title=f"{method.title()} correlations, similar columns together", yaxis_autorange="reversed")
    return fig


def _layout(weights: np.ndarray, iterations: int = 200, seed: int = 0) -> np.ndarray:
    """Disposición por fuerzas (muelles entre columnas relacionadas, repulsión entre todas) sin bibliotecas de grafos."""
    n = len(weights)
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 2))
    for step in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2) + 1e-6
        repel = (delta / dist[..., None] ** 2).sum(axis=1) * 0.02
        attract = -(weights[..., None] * delta).sum(axis=1) * 0.05
        pos += (repel + attract) * (1 - step / iterations)
    pos -= pos.mean(axis=0)
    return pos / (np.abs(pos).max() or 1)


def correlation_network(df: pd.DataFrame, columns=None, threshold: float = 0.5) -> go.Figure:
    """Red: cada columna es un nodo y se unen las que se correlacionan (en valor absoluto) por encima del umbral."""
    if not 0.1 <= threshold < 1:
        raise ValueError("The threshold must be between 0.1 and 1.")
    cols = _numeric(df, columns, minimum=3)[:40]
    corr = df[cols].corr(method="spearman").fillna(0).to_numpy()
    strong = np.abs(corr) >= threshold
    np.fill_diagonal(strong, False)
    if not strong.any():
        raise ValueError(f"No pair of columns reaches a correlation of {threshold}. Lower the threshold.")
    pos = _layout(np.abs(corr) * strong)
    fig = go.Figure()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if strong[i, j]:
                fig.add_trace(go.Scatter(x=[pos[i, 0], pos[j, 0]], y=[pos[i, 1], pos[j, 1]], mode="lines", hoverinfo="text",
                                         text=f"{cols[i]} and {cols[j]}: {corr[i, j]:+.2f}", showlegend=False,
                                         line=dict(width=1 + 5 * abs(corr[i, j]), color="#2f5d62" if corr[i, j] > 0 else "#9a3412"), opacity=0.55))
    degree = strong.sum(axis=1)
    fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=cols, textposition="top center", showlegend=False,
                             marker=dict(size=14 + 6 * degree, color=degree, colorscale="Oranges", line=dict(width=1)),
                             hovertext=[f"{c}: linked to {d} column(s)" for c, d in zip(cols, degree)], hoverinfo="text"))
    fig.update_layout(title=f"Columns linked by a correlation of {threshold:.1f} or more (teal: positive, red: negative)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ── Tiempo ───────────────────────────────────────────────────────────────────
def calendar_heatmap(df: pd.DataFrame, date: str, value: str | None = None, agg: str = "sum", weeks: int = 52) -> go.Figure:
    """Un cuadrado por día (filas: día de la semana; columnas: semanas) con el color del total o la media de ese día."""
    _need(df, date)
    if agg not in ("sum", "mean", "count"):
        raise ValueError("agg must be sum, mean or count.")
    when = pd.to_datetime(df[date], errors="coerce")
    if when.notna().sum() < 7:
        raise ValueError("At least 7 valid dates are needed.")
    if value is not None:
        _need(df, value)
        if not pd.api.types.is_numeric_dtype(df[value]):
            raise ValueError(f"'{value}' must be numeric.")
    data = pd.DataFrame({"d": when.dt.normalize(), "v": df[value] if value else 1.0}).dropna()
    daily = data.groupby("d")["v"].agg("count" if agg == "count" or not value else agg)
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
    daily = daily.iloc[-weeks * 7:]
    start = daily.index[0] - pd.Timedelta(days=daily.index[0].weekday())
    week = ((daily.index - start).days // 7).to_numpy()
    grid = np.full((7, week.max() + 1), np.nan)
    grid[daily.index.weekday, week] = daily.to_numpy()
    labels = [(start + pd.Timedelta(weeks=int(w))).strftime("%b %d") for w in range(grid.shape[1])]
    fig = go.Figure(go.Heatmap(z=grid, x=labels, y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], colorscale="Oranges", xgap=2, ygap=2,
                               hoverongaps=False))
    fig.update_layout(title=f"{'Rows' if not value else value} per day ({agg})", yaxis_autorange="reversed")
    return fig


# ── Distribuciones ───────────────────────────────────────────────────────────
def ridgeline(df: pd.DataFrame, value: str, group: str, top: int = 8) -> go.Figure:
    """Una curva de densidad por grupo, apiladas con solapamiento: sirve para ver cómo cambia una distribución entre grupos."""
    _need(df, value, group)
    if not pd.api.types.is_numeric_dtype(df[value]):
        raise ValueError(f"'{value}' must be numeric.")
    labels = _top_levels(df[group], top)
    frame = pd.DataFrame({"v": df[value], "g": labels}).dropna()
    grid = np.linspace(frame["v"].min(), frame["v"].max(), 200)
    curves = {}
    for name, part in frame.groupby("g"):
        if len(part) >= 5 and part["v"].nunique() > 1:
            curves[name] = gaussian_kde(part["v"])(grid)
    if len(curves) < 2:
        raise ValueError("At least two groups with 5 or more different values are needed.")
    scale = 1.6 / max(c.max() for c in curves.values())
    order = sorted(curves, key=lambda g: float(np.average(grid, weights=curves[g])))
    fig = go.Figure()
    for i, name in enumerate(order):
        y = curves[name] * scale + i
        fig.add_trace(go.Scatter(x=grid, y=[i] * len(grid), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=grid, y=y, mode="lines", fill="tonexty", name=str(name), line=dict(width=1.2), opacity=0.8, showlegend=False,
                                 hovertemplate=f"{name}<extra></extra>"))
    fig.update_layout(title=f"Distribution of {value} by {group}", yaxis=dict(tickmode="array", tickvals=list(range(len(order))),
                                                                              ticktext=[str(o) for o in order]))
    return fig


def parallel(df: pd.DataFrame, columns: list, color: str | None = None, sample: int = 1500) -> go.Figure:
    """Coordenadas paralelas: una línea por fila que cruza un eje por columna; sirve para ver perfiles y grupos."""
    cols = _numeric(df, columns, minimum=3)[:10]
    frame = df[cols + ([color] if color and color not in cols else [])].dropna()
    if len(frame) < 5:
        raise ValueError("At least 5 complete rows are needed.")
    frame = frame.sample(min(len(frame), sample), random_state=0)
    if color:
        _need(df, color)
        shade = frame[color] if pd.api.types.is_numeric_dtype(frame[color]) else pd.factorize(frame[color])[0]
    else:
        shade = frame[cols[0]]
    fig = go.Figure(go.Parcoords(line=dict(color=shade, colorscale="Oranges"), dimensions=[dict(label=c, values=frame[c]) for c in cols]))
    fig.update_layout(title="One line per row across the columns")
    return fig


def small_multiples(df: pd.DataFrame, x: str, y: str, facet: str, kind: str = "line", top: int = 9) -> go.Figure:
    """El mismo gráfico repetido para cada valor de `facet`, con los mismos ejes para poder comparar."""
    _need(df, x, y, facet)
    if kind not in ("line", "bar", "scatter"):
        raise ValueError("kind must be line, bar or scatter.")
    if not pd.api.types.is_numeric_dtype(df[y]):
        raise ValueError(f"'{y}' must be numeric.")
    labels = _top_levels(df[facet], top)
    names = list(labels.value_counts().index)
    cols = 3 if len(names) > 4 else 2
    rows = int(np.ceil(len(names) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(n) for n in names], shared_xaxes=True, shared_yaxes=True)
    for i, name in enumerate(names):
        part = df[labels == name].sort_values(x) if kind == "line" else df[labels == name]
        trace = {"line": go.Scatter(x=part[x], y=part[y], mode="lines"), "scatter": go.Scatter(x=part[x], y=part[y], mode="markers"),
                 "bar": go.Bar(x=part[x], y=part[y])}[kind]
        fig.add_trace(trace, row=i // cols + 1, col=i % cols + 1)
    fig.update_layout(title=f"{y} against {x}, one panel per {facet}", showlegend=False)
    return fig


GALLERY = {
    "Flow between two columns (Sankey)": "sankey", "Nested groups (treemap)": "treemap", "Profile of each group (radar)": "radar",
    "Correlations grouped by similarity": "clustered_heatmap", "Network of correlated columns": "correlation_network",
    "Calendar of activity": "calendar_heatmap", "Distributions by group (ridgeline)": "ridgeline",
    "Rows across columns (parallel coordinates)": "parallel", "One panel per group": "small_multiples",
}
