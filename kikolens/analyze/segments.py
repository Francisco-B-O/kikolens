"""Segmentación avanzada. Tres familias de métodos sobre las columnas numéricas: mezclas gaussianas (eligen solas el número de grupos con el
criterio BIC y dan la confianza de cada fila), agrupación jerárquica (con dendrograma) y DBSCAN (encuentra grupos de forma libre y marca
el ruido, con el radio elegido a partir de las distancias a los vecinos). Cada grupo sale descrito con palabras («units alto, price bajo»)."""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

MAX_HIERARCHICAL = 2500


def prepare(df: pd.DataFrame, columns=None) -> tuple:
    """Columnas numéricas que varían, con vacíos rellenos con la mediana y escaladas. Devuelve (matriz escalada, columnas, tabla imputada)."""
    cols = [c for c in (columns or df.columns) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() > 1]
    if len(cols) < 2:
        raise ValueError("At least two numeric columns that vary are needed.")
    if len(df) < 30:
        raise ValueError("At least 30 rows are needed.")
    filled = df[cols].astype(float).fillna(df[cols].median())
    return StandardScaler().fit_transform(filled), cols, filled


def describe_clusters(filled: pd.DataFrame, labels: np.ndarray, top: int = 2) -> pd.DataFrame:
    """Tamaño de cada grupo, su media en cada columna y una descripción con las dos columnas que más lo distinguen."""
    z = (filled - filled.mean()) / filled.std(ddof=0).replace(0, 1)
    rows = []
    for label in sorted(set(labels)):
        mask = labels == label
        profile = z[mask].mean()
        strong = profile.reindex(profile.abs().sort_values(ascending=False).index)[:top]
        words = [f"{c} {'high' if v > 0 else 'low'}" for c, v in strong.items() if abs(v) >= 0.4]
        rows.append({"group": "noise" if label == -1 else int(label), "rows": int(mask.sum()), "share": float(mask.mean()),
                     "described as": ", ".join(words) or "close to the average", **{c: float(filled.loc[mask, c].mean()) for c in filled.columns}})
    return pd.DataFrame(rows)


def map_figure(X: np.ndarray, labels: np.ndarray, title: str = "Groups on the two main components") -> go.Figure:
    pts = PCA(n_components=2, random_state=0).fit_transform(X)
    if len(pts) > 3000:
        keep = np.random.default_rng(0).choice(len(pts), 3000, replace=False)
        pts, labels = pts[keep], labels[keep]
    fig = go.Figure()
    for label in sorted(set(labels)):
        part = pts[labels == label]
        fig.add_trace(go.Scatter(x=part[:, 0], y=part[:, 1], mode="markers", name="noise" if label == -1 else f"Group {label}",
                                 marker=dict(size=6, opacity=0.65, symbol="x" if label == -1 else "circle")))
    fig.update_layout(title=title, xaxis_title="Component 1", yaxis_title="Component 2")
    return fig


def _quality(X: np.ndarray, labels: np.ndarray):
    mask = labels != -1
    if len(set(labels[mask])) < 2 or mask.sum() < 3:
        return None
    sample = np.random.default_rng(0).choice(np.where(mask)[0], min(2000, mask.sum()), replace=False)
    return float(silhouette_score(X[sample], labels[sample]))


# ── Mezclas gaussianas ───────────────────────────────────────────────────────
def gmm_auto(df: pd.DataFrame, columns=None, k_max: int = 8, seed: int = 0) -> dict:
    """Prueba de 1 a `k_max` grupos y se queda con el de menor BIC. Cada fila recibe la probabilidad de pertenecer a su grupo."""
    if not 2 <= k_max <= 15:
        raise ValueError("k_max must be between 2 and 15.")
    X, cols, filled = prepare(df, columns)
    covariance = "full" if len(X) >= 20 * len(cols) else "diag"
    rows, models = [], {}
    for k in range(1, min(k_max, len(X) // 10) + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianMixture(k, covariance_type=covariance, n_init=3, random_state=seed).fit(X)
        rows.append({"groups": k, "BIC": float(model.bic(X)), "AIC": float(model.aic(X))})
        models[k] = model
    table = pd.DataFrame(rows)
    best = int(table.loc[table["BIC"].idxmin(), "groups"])
    model = models[best]
    labels = model.predict(X)
    confidence = model.predict_proba(X).max(axis=1)
    fig = go.Figure(go.Scatter(x=table["groups"], y=table["BIC"], mode="lines+markers"))
    fig.add_vline(x=best, line_dash="dash", annotation_text=f"Lowest BIC: {best}")
    fig.update_layout(title="BIC by number of groups (lower is better)", xaxis_title="Groups", yaxis_title="BIC")
    profiles = describe_clusters(filled, labels)
    return {"method": "Gaussian mixture", "labels": pd.Series(labels, index=df.index, name="group"), "confidence": pd.Series(confidence, index=df.index),
            "n_groups": best, "table": table, "profiles": profiles, "silhouette": _quality(X, labels), "figure": fig, "map": map_figure(X, labels),
            "uncertain": float((confidence < 0.7).mean()),
            "summary": (f"{best} group(s) fit best (BIC). {(confidence < 0.7).mean():.0%} of the rows sit between groups (confidence below 70%). "
                        + _silhouette_note(_quality(X, labels)))}


def _silhouette_note(value) -> str:
    if value is None:
        return ""
    quality = "well separated" if value > 0.5 else "moderately separated" if value > 0.25 else "overlapping"
    return f"The groups are {quality} (silhouette {value:.2f})."


# ── Jerárquico ───────────────────────────────────────────────────────────────
def hierarchical(df: pd.DataFrame, columns=None, n_groups: int = 4, method: str = "ward") -> dict:
    """Agrupación jerárquica aglomerativa. Con más de 2.500 filas se agrupa una muestra y el resto se asigna al grupo más cercano."""
    if method not in ("ward", "average", "complete", "single"):
        raise ValueError("method must be ward, average, complete or single.")
    if not 2 <= n_groups <= 20:
        raise ValueError("n_groups must be between 2 and 20.")
    X, cols, filled = prepare(df, columns)
    sample = np.arange(len(X))
    if len(X) > MAX_HIERARCHICAL:
        sample = np.sort(np.random.default_rng(0).choice(len(X), MAX_HIERARCHICAL, replace=False))
    tree = linkage(X[sample], method=method)
    part = fcluster(tree, n_groups, criterion="maxclust") - 1
    labels = np.empty(len(X), int)
    labels[sample] = part
    if len(sample) < len(X):
        centroids = np.array([X[sample][part == g].mean(axis=0) for g in range(part.max() + 1)])
        rest = np.setdiff1d(np.arange(len(X)), sample)
        labels[rest] = np.argmin(((X[rest, None, :] - centroids[None]) ** 2).sum(axis=2), axis=1)
    profiles = describe_clusters(filled, labels)
    return {"method": f"Hierarchical ({method})", "labels": pd.Series(labels, index=df.index, name="group"), "n_groups": int(labels.max() + 1),
            "profiles": profiles, "silhouette": _quality(X, labels), "figure": dendrogram_figure(tree, n_groups), "map": map_figure(X, labels),
            "summary": f"Cut into {int(labels.max() + 1)} groups. " + _silhouette_note(_quality(X, labels))
                       + (f" Grouped a sample of {len(sample):,} rows and assigned the rest to the nearest group." if len(sample) < len(X) else "")}


def dendrogram_figure(tree: np.ndarray, n_groups: int) -> go.Figure:
    data = dendrogram(tree, no_plot=True, truncate_mode="lastp", p=min(40, len(tree) + 1))
    fig = go.Figure()
    for xs, ys in zip(data["icoord"], data["dcoord"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#6b675f", width=1.2), hoverinfo="skip", showlegend=False))
    cut = float(tree[-(n_groups - 1), 2]) if n_groups > 1 and len(tree) >= n_groups - 1 else None
    if cut is not None:
        fig.add_hline(y=cut * 0.999, line_dash="dash", annotation_text=f"{n_groups} groups")
    fig.update_layout(title="Dendrogram (the last 40 merges)", xaxis=dict(showticklabels=False), yaxis_title="Distance at which groups merge")
    return fig


# ── DBSCAN ───────────────────────────────────────────────────────────────────
def suggest_eps(X: np.ndarray, min_samples: int = 5) -> float:
    """Radio donde la curva ordenada de distancias al vecino k-ésimo gira con más fuerza (el «codo»)."""
    k = min(min_samples, len(X) - 1)
    dist = np.sort(NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, -1])
    smooth = np.convolve(dist, np.ones(5) / 5, mode="same")
    knee = int(np.argmax(np.gradient(np.gradient(smooth)))) if len(smooth) > 10 else len(smooth) // 2
    knee = int(np.clip(knee, 5, len(dist) - 6)) if len(dist) > 12 else len(dist) // 2
    return float(max(dist[knee], 1e-3))


def dbscan_auto(df: pd.DataFrame, columns=None, min_samples: int = 5, eps: float | None = None) -> dict:
    """DBSCAN con el radio elegido por el codo de las distancias, o el que se indique."""
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2.")
    X, cols, filled = prepare(df, columns)
    radius = eps if eps is not None else suggest_eps(X, min_samples)
    if radius <= 0:
        raise ValueError("eps must be positive.")
    labels = DBSCAN(eps=radius, min_samples=min_samples).fit_predict(X)
    n_groups = len(set(labels) - {-1})
    noise = float((labels == -1).mean())
    if n_groups == 0:
        raise ValueError(f"With a radius of {radius:.3g} every row is noise. Increase eps or lower min_samples.")
    profiles = describe_clusters(filled, labels)
    return {"method": "DBSCAN", "labels": pd.Series(labels, index=df.index, name="group"), "n_groups": n_groups, "eps": radius, "noise_share": noise,
            "profiles": profiles, "silhouette": _quality(X, labels), "figure": map_figure(X, labels, f"DBSCAN groups (radius {radius:.3g})"),
            "summary": f"Found {n_groups} group(s) and {noise:.0%} noise at a radius of {radius:.3g}. " + _silhouette_note(_quality(X, labels))}
