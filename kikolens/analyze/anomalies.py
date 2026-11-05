"""Detección de anomalías: filas que no se parecen al resto. Cuatro métodos, y para cada fila marcada se explica
qué columnas la hacen rara (en desviaciones robustas respecto a lo típico)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

METHODS = ["Isolation Forest", "Local Outlier Factor", "Robust z-score", "Mahalanobis distance"]
MIN_ROWS = 20
MAX_COLUMNS = 50
MAD_SCALE = 1.4826  # convierte la desviación absoluta mediana en un equivalente de desviación típica


def usable_columns(df: pd.DataFrame) -> list:
    cols = [c for c in df.select_dtypes("number").columns
            if not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique(dropna=True) > 1]
    return cols[:MAX_COLUMNS]


def _robust_z(X: pd.DataFrame) -> pd.DataFrame:
    median = X.median()
    mad = (X - median).abs().median() * MAD_SCALE
    spread = mad.where(mad > 0, X.std().replace(0, np.nan)).replace(0, np.nan)
    return ((X - median) / spread).fillna(0.0)


def _scores(method: str, X: pd.DataFrame, contamination: float, seed: int) -> np.ndarray:
    scaled = RobustScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    if method == "Isolation Forest":
        forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed).fit(scaled)
        return -forest.score_samples(scaled)
    if method == "Local Outlier Factor":
        lof = LocalOutlierFactor(n_neighbors=min(20, len(X) - 1), contamination=contamination)
        lof.fit_predict(scaled)
        return -lof.negative_outlier_factor_
    if method == "Robust z-score":
        return _robust_z(X.fillna(X.median())).abs().max(axis=1).to_numpy()
    if method == "Mahalanobis distance":
        centered = scaled - np.median(scaled, axis=0)
        inverse = np.linalg.pinv(np.cov(scaled, rowvar=False).reshape(scaled.shape[1], scaled.shape[1]))
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centered, inverse, centered), 0))
    raise ValueError(f"Unknown method '{method}'. Use one of {METHODS}.")


def detect_anomalies(df: pd.DataFrame, columns=None, method: str = "Isolation Forest", contamination: float = 0.02,
                     seed: int = 42, top_reasons: int = 3) -> dict:
    """Puntúa cada fila y marca la fracción `contamination` más rara. Devuelve puntuaciones, marcas y razones."""
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Use one of {METHODS}.")
    if not 0.001 <= contamination <= 0.3:
        raise ValueError("The expected share of anomalies must be between 0.1% and 30%.")
    columns = list(columns) if columns is not None else usable_columns(df)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {missing}")
    columns = [c for c in columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1]
    if not columns:
        raise ValueError("There are no numeric columns with variation to analyze.")
    if method in ("Isolation Forest", "Local Outlier Factor", "Mahalanobis distance") and len(columns) < 2:
        raise ValueError(f"{method} needs at least two numeric columns; use Robust z-score for a single one.")
    if len(df) < MIN_ROWS:
        raise ValueError(f"At least {MIN_ROWS} rows are needed to find anomalies.")

    X = df[columns].astype(float)
    scores = pd.Series(_scores(method, X, contamination, seed), index=df.index, name="anomaly_score")
    n_flag = max(1, int(round(len(df) * contamination)))
    threshold = float(scores.nlargest(n_flag).min())
    flags = scores >= threshold
    z = _robust_z(X.fillna(X.median()))

    reasons = {}
    for idx in scores[flags].sort_values(ascending=False).index:
        row = z.loc[idx]
        top = row.abs().sort_values(ascending=False).head(top_reasons)
        reasons[idx] = "; ".join(
            f"{c} is {abs(row[c]):.1f} robust SDs {'above' if row[c] > 0 else 'below'} typical ({df.at[idx, c]:.4g})"
            for c in top.index if abs(row[c]) >= 2) or "unusual combination of values"
    table = df.loc[list(reasons)].copy()
    table.insert(0, "anomaly_score", scores.loc[table.index].round(4))
    table["why"] = pd.Series(reasons)
    return {"scores": scores, "flags": flags, "table": table, "columns": columns, "method": method,
            "threshold": threshold, "n_flagged": int(flags.sum()), "contamination": contamination,
            "summary": f"{int(flags.sum()):,} of {len(df):,} rows ({flags.mean():.1%}) were flagged using {method} "
                       f"on {len(columns)} numeric column(s)."}


def anomaly_map(df: pd.DataFrame, result: dict) -> go.Figure:
    """Las filas en el plano de los dos componentes principales, con las anomalías resaltadas."""
    X = SimpleImputer(strategy="median").fit_transform(df[result["columns"]].astype(float))
    scaled = RobustScaler().fit_transform(X)
    flags = result["flags"].to_numpy()
    if scaled.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=0).fit_transform(scaled)
        labels = ("Component 1", "Component 2")
    else:
        coords = np.column_stack([np.arange(len(scaled)), scaled[:, 0]])
        labels = ("Row", result["columns"][0])
    fig = go.Figure()
    for mask, name, size in ((~flags, "Normal", 6), (flags, "Anomaly", 10)):
        fig.add_trace(go.Scattergl(x=coords[mask, 0], y=coords[mask, 1], mode="markers", name=name,
                                   marker=dict(size=size, opacity=0.55 if name == "Normal" else 0.95),
                                   text=[f"row {i} · score {s:.3f}" for i, s in zip(df.index[mask], result["scores"][mask])]))
    fig.update_layout(title="Rows in two dimensions", xaxis_title=labels[0], yaxis_title=labels[1])
    return fig


def score_histogram(result: dict) -> go.Figure:
    fig = go.Figure(go.Histogram(x=result["scores"], nbinsx=60, name="Rows"))
    fig.add_vline(x=result["threshold"], line_dash="dash", annotation_text="flagged above")
    fig.update_layout(title="Distribution of anomaly scores", xaxis_title="Score (higher is stranger)", showlegend=False)
    return fig
