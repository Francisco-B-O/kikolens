"""Reducción de dimensión y redundancia: PCA con sus cargas, proyección t-SNE y factor de inflación de la varianza
(VIF) para encontrar columnas que se pueden predecir a partir de otras."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

MAX_TSNE_ROWS = 3000
MAX_COLORS = 12


def numeric_columns(df: pd.DataFrame) -> list:
    return [c for c in df.select_dtypes("number").columns
            if not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique(dropna=True) > 1]


def _matrix(df: pd.DataFrame, columns) -> tuple:
    columns = list(columns) if columns is not None else numeric_columns(df)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {missing}")
    columns = [c for c in columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1]
    if len(columns) < 2:
        raise ValueError("At least two numeric columns with variation are needed.")
    if len(df) < 5:
        raise ValueError("At least 5 rows are needed.")
    X = SimpleImputer(strategy="median").fit_transform(df[columns].astype(float))
    return columns, StandardScaler().fit_transform(X)


def pca_analysis(df: pd.DataFrame, columns=None, n_components: int | None = None) -> dict:
    """PCA sobre columnas estandarizadas. Devuelve varianza explicada, cargas, puntuaciones y componentes para el 90%."""
    columns, X = _matrix(df, columns)
    limit = min(len(columns), len(X))
    n = limit if n_components is None else int(n_components)
    if not 2 <= n <= limit:
        raise ValueError(f"The number of components must be between 2 and {limit}.")
    pca = PCA(n_components=n, random_state=0).fit(X)
    names = [f"PC{i + 1}" for i in range(n)]
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    needed = int(np.searchsorted(cumulative, 0.9) + 1) if cumulative[-1] >= 0.9 else None
    loadings = pd.DataFrame(pca.components_.T, index=columns, columns=names)
    top = {pc: loadings[pc].abs().sort_values(ascending=False).head(3).index.tolist() for pc in names[:5]}
    scores = pd.DataFrame(pca.transform(X), columns=names, index=df.index)
    return {"columns": columns, "explained": pca.explained_variance_ratio_, "cumulative": cumulative,
            "components_for_90": needed, "loadings": loadings, "scores": scores, "top_features": top, "names": names,
            "summary": (f"{needed} component(s) capture 90% of the variation in {len(columns)} columns." if needed else
                        f"{n} component(s) capture {cumulative[-1]:.0%} of the variation in {len(columns)} columns.")}


def scree_figure(result: dict) -> go.Figure:
    names = result["names"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=result["explained"], name="Each component"))
    fig.add_trace(go.Scatter(x=names, y=result["cumulative"], mode="lines+markers", name="Cumulative"))
    fig.add_hline(y=0.9, line_dash="dot", annotation_text="90%")
    fig.update_layout(title="Variance explained", yaxis_title="Share of variation", yaxis_tickformat=".0%")
    return fig


def loadings_figure(result: dict, components=("PC1", "PC2")) -> go.Figure:
    fig = go.Figure()
    for pc in components:
        ordered = result["loadings"][pc].sort_values()
        fig.add_trace(go.Bar(x=ordered.values, y=ordered.index, orientation="h", name=pc))
    fig.update_layout(title="What each component is made of", barmode="group", xaxis_title="Loading")
    return fig


def projection_figure(coords: pd.DataFrame, x: str, y: str, color: pd.Series | None = None, title: str = "") -> go.Figure:
    fig = go.Figure()
    if color is not None:
        series = color.reindex(coords.index)
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > MAX_COLORS:
            fig.add_trace(go.Scattergl(x=coords[x], y=coords[y], mode="markers", text=series.round(3).astype(str),
                                       marker=dict(size=6, opacity=0.7, color=series, colorscale="Viridis",
                                                   colorbar=dict(title=str(series.name)))))
        else:
            labels = series.astype(str).fillna("(empty)")
            keep = labels.value_counts().head(MAX_COLORS).index
            for value in keep:
                mask = labels == value
                fig.add_trace(go.Scattergl(x=coords.loc[mask, x], y=coords.loc[mask, y], mode="markers", name=str(value),
                                           marker=dict(size=6, opacity=0.7)))
    else:
        fig.add_trace(go.Scattergl(x=coords[x], y=coords[y], mode="markers", marker=dict(size=6, opacity=0.6),
                                   showlegend=False))
    fig.update_layout(title=title or f"{x} against {y}", xaxis_title=x, yaxis_title=y)
    return fig


def tsne_embedding(df: pd.DataFrame, columns=None, perplexity: float = 30.0, seed: int = 0) -> tuple:
    """Proyección t-SNE a 2D. Con más de 3.000 filas usa una muestra. Devuelve (coordenadas, aviso)."""
    columns, X = _matrix(df, columns)
    note = None
    index = df.index
    if len(X) > MAX_TSNE_ROWS:
        pick = np.random.default_rng(seed).choice(len(X), MAX_TSNE_ROWS, replace=False)
        X, index = X[np.sort(pick)], index[np.sort(pick)]
        note = f"t-SNE is slow, so a random sample of {MAX_TSNE_ROWS:,} rows was used."
    if len(X) < 10:
        raise ValueError("t-SNE needs at least 10 rows.")
    perplexity = float(min(max(perplexity, 2.0), (len(X) - 1) / 3))
    coords = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(X)
    return pd.DataFrame(coords, columns=["t-SNE 1", "t-SNE 2"], index=index), note


def vif_table(df: pd.DataFrame, columns=None, threshold: float = 5.0) -> pd.DataFrame:
    """VIF de cada columna: cuánto se explica con las demás. Más de 5 indica redundancia; más de 10, grave."""
    columns = list(columns) if columns is not None else numeric_columns(df)
    columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1]
    if len(columns) < 2:
        raise ValueError("At least two numeric columns with variation are needed.")
    data = df[columns].dropna().astype(float)
    if len(data) < 2 * len(columns):
        raise ValueError("Too few complete rows for this many columns. Fill missing values or choose fewer columns.")
    corr = np.corrcoef(data.to_numpy(), rowvar=False)
    inverse = np.linalg.pinv(corr)
    vif = np.diag(inverse)
    out = pd.DataFrame({"column": columns, "vif": np.clip(vif, 1.0, None)})
    out["verdict"] = np.where(out["vif"] >= 10, "severe", np.where(out["vif"] >= threshold, "high", "fine"))
    out["explained by the others"] = 1 - 1 / out["vif"]
    return out.sort_values("vif", ascending=False, ignore_index=True)
