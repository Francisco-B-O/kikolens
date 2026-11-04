"""Datos sintéticos: una tabla nueva que se parece a la real (cada columna con su distribución y las columnas con las mismas
correlaciones) pero cuyas filas no son personas reales. Se genera con una cópula gaussiana: cada columna se convierte a puntuaciones
normales, se muestrea de una normal multivariante con la correlación observada y se devuelve a la escala original. El informe mide tres
cosas: fidelidad (¿se parece?), privacidad (¿hay copias o casi copias de filas reales?) y utilidad (¿un modelo entrenado con los datos
sintéticos funciona casi igual sobre los reales?)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

MAX_CATEGORIES = 50


def _ns(series: pd.Series) -> pd.Series:
    """Fechas como enteros en nanosegundos, sea cual sea la resolución de la columna (pandas 3 usa microsegundos por defecto)."""
    return series.astype("datetime64[ns]").astype("int64").where(series.notna())


def _kinds(df: pd.DataFrame) -> dict:
    kinds = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            kinds[col] = "date"
        elif pd.api.types.is_bool_dtype(s):
            kinds[col] = "category"
        elif pd.api.types.is_numeric_dtype(s):
            kinds[col] = "number"
        elif s.nunique(dropna=True) <= MAX_CATEGORIES:
            kinds[col] = "category"
        elif s.nunique(dropna=True) > 0.9 * s.notna().sum():
            kinds[col] = "identifier"
        else:
            kinds[col] = "skip"
    return kinds


def _normal_scores(s: pd.Series, kind: str, rng) -> np.ndarray:
    """La columna como puntuaciones normales. En las categorías, cada valor ocupa un tramo de probabilidad y se elige un punto al azar de él."""
    n = s.notna().sum()
    out = np.full(len(s), np.nan)
    if kind == "category":
        counts = s.astype(str).where(s.notna()).value_counts()
        edges = np.concatenate([[0], np.cumsum(counts.to_numpy() / counts.sum())])
        bounds = {k: (edges[i], edges[i + 1]) for i, k in enumerate(counts.index)}
        values = s.astype(str).where(s.notna())
        low = values.map(lambda v: bounds.get(v, (np.nan, np.nan))[0]).to_numpy(float)
        high = values.map(lambda v: bounds.get(v, (np.nan, np.nan))[1]).to_numpy(float)
        u = low + rng.random(len(s)) * (high - low)
    else:
        numeric = _ns(s).astype(float) if kind == "date" else s.astype(float)
        ranks = numeric.rank(method="average").to_numpy()
        u = (ranks - 0.5) / max(n, 1)
    mask = ~np.isnan(u)
    out[mask] = stats.norm.ppf(np.clip(u[mask], 1e-6, 1 - 1e-6))
    return out


def _nearest_psd(corr: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((corr + corr.T) / 2)
    values = np.clip(values, 1e-6, None)
    fixed = vectors @ np.diag(values) @ vectors.T
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


def synthesize(df: pd.DataFrame, n: int | None = None, seed: int = 0, keep_missing: bool = True) -> dict:
    """Devuelve `data` (la tabla sintética), `skipped` (columnas que no se pudieron imitar) y las columnas de identificadores regeneradas."""
    if df is None or len(df) < 30:
        raise ValueError("At least 30 rows are needed to learn what the data look like.")
    n = len(df) if n is None else int(n)
    if not 1 <= n <= 1_000_000:
        raise ValueError("n must be between 1 and 1,000,000.")
    rng = np.random.default_rng(seed)
    kinds = _kinds(df)
    modelled = [c for c, k in kinds.items() if k in ("number", "category", "date")]
    if len(modelled) < 1:
        raise ValueError("No column can be imitated (they are identifiers or long free text).")
    scores = pd.DataFrame({c: _normal_scores(df[c], kinds[c], rng) for c in modelled})
    corr = np.array(scores.corr().fillna(0.0)) if len(modelled) > 1 else np.ones((1, 1))      # copia: con pandas 3 to_numpy() es de solo lectura
    np.fill_diagonal(corr, 1.0)
    corr = _nearest_psd(corr)
    z = rng.multivariate_normal(np.zeros(len(modelled)), corr, size=n)
    u = stats.norm.cdf(z)
    out = pd.DataFrame(index=range(n))
    for j, col in enumerate(modelled):
        s = df[col].dropna()
        if kinds[col] == "category":
            counts = df[col].astype(str).where(df[col].notna()).value_counts()
            edges = np.cumsum(counts.to_numpy() / counts.sum())
            labels = counts.index.to_numpy()
            picked = labels[np.minimum(np.searchsorted(edges, u[:, j]), len(labels) - 1)]
            out[col] = pd.Series(picked).astype(object)
            if pd.api.types.is_bool_dtype(df[col]):
                out[col] = out[col].map({"True": True, "False": False})
            elif pd.api.types.is_numeric_dtype(df[col]):
                out[col] = pd.to_numeric(out[col])
        else:
            base = _ns(s).astype(float) if kinds[col] == "date" else s.astype(float)
            values = np.quantile(base.to_numpy(), u[:, j])
            if kinds[col] == "date":
                out[col] = pd.to_datetime(values.astype("int64"), unit="ns")
            elif pd.api.types.is_integer_dtype(df[col]):
                out[col] = np.round(values).astype(df[col].dtype if df[col].isna().sum() == 0 else "float64")
            else:
                out[col] = values
    for col, kind in kinds.items():
        if kind == "identifier":
            out[col] = [f"{col}_{i + 1:06d}" for i in range(n)]
    skipped = [c for c, k in kinds.items() if k == "skip"]
    if keep_missing:
        for col in out.columns:
            if col in df.columns and df[col].isna().any():
                out.loc[rng.random(n) < df[col].isna().mean(), col] = np.nan
    ordered = [c for c in df.columns if c in out.columns]
    return {"data": out[ordered], "skipped": skipped, "identifiers": [c for c, k in kinds.items() if k == "identifier"], "kinds": kinds,
            "summary": f"{n:,} synthetic rows with {len(ordered)} column(s)." + (f" Not imitated: {', '.join(skipped)}." if skipped else "")}


# ── Informe ──────────────────────────────────────────────────────────────────
def _matrix(real: pd.DataFrame, synth: pd.DataFrame, kinds: dict) -> tuple:
    """Filas reales y sintéticas como una matriz numérica común (categorías en indicadores, todo estandarizado) para medir distancias."""
    cols = [c for c, k in kinds.items() if k in ("number", "category", "date") and c in synth.columns]
    both = pd.concat([real[cols], synth[cols]], ignore_index=True)
    flag = np.r_[np.zeros(len(real), bool), np.ones(len(synth), bool)]
    parts = []
    for c in cols:
        if kinds[c] == "category":
            parts.append(pd.get_dummies(both[c].astype(str), prefix=c, dtype=float))
        elif kinds[c] == "date":
            parts.append(_ns(both[c]).astype(float).rename(c).to_frame())
        else:
            parts.append(both[c].astype(float).to_frame())
    matrix = pd.concat(parts, axis=1)
    matrix = ((matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, 1)).fillna(0.0)
    return matrix[~flag].to_numpy(), matrix[flag].to_numpy()


def report(real: pd.DataFrame, synth: pd.DataFrame, max_rows: int = 3000, seed: int = 0) -> dict:
    """Fidelidad por columna y de las correlaciones, y privacidad (filas idénticas y casi idénticas a una real)."""
    kinds = _kinds(real)
    rows = []
    for col in synth.columns:
        k = kinds.get(col)
        if k in ("number", "date"):
            a = (_ns(real[col]) if k == "date" else real[col]).dropna().astype(float)
            b = (_ns(synth[col]) if k == "date" else synth[col]).dropna().astype(float)
            stat = float(stats.ks_2samp(a, b).statistic)
            rows.append({"column": col, "kind": k, "difference": stat, "measure": "Kolmogorov-Smirnov distance"})
        elif k == "category":
            pa = real[col].astype(str).value_counts(normalize=True)
            pb = synth[col].astype(str).value_counts(normalize=True)
            idx = pa.index.union(pb.index)
            tvd = float(0.5 * np.abs(pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).sum())
            rows.append({"column": col, "kind": k, "difference": tvd, "measure": "Total variation distance"})
    table = pd.DataFrame(rows)
    num = [c for c in synth.columns if kinds.get(c) == "number"]
    corr_gap = None
    if len(num) >= 2:
        corr_gap = float(np.abs(real[num].corr(method="spearman") - synth[num].corr(method="spearman")).to_numpy()[np.triu_indices(len(num), 1)].mean())
    fidelity = float(100 * (1 - min(1.0, (table["difference"].mean() if len(table) else 0.0) * 2 + (corr_gap or 0.0))))

    cols = [c for c in synth.columns if kinds.get(c) in ("number", "category", "date")]
    exact = float(synth[cols].astype(str).apply(tuple, axis=1).isin(set(real[cols].astype(str).apply(tuple, axis=1))).mean()) if cols else 0.0
    close = None
    if cols:
        R, S = _matrix(real.sample(min(len(real), max_rows), random_state=seed), synth.sample(min(len(synth), max_rows), random_state=seed), kinds)
        nn_real = NearestNeighbors(n_neighbors=2).fit(R)
        baseline = np.median(nn_real.kneighbors(R)[0][:, 1])           # a qué distancia está normalmente una fila real de otra real
        to_real = np.min(NearestNeighbors(n_neighbors=1).fit(R).kneighbors(S)[0], axis=1)
        close = float((to_real < 0.2 * baseline).mean()) if baseline > 0 else 0.0
    return {"table": table, "correlation_gap": corr_gap, "fidelity": fidelity, "exact_copies": exact, "near_copies": close,
            "summary": (f"Fidelity {fidelity:.0f}/100 (columns differ by {table['difference'].mean():.3f} on average"
                        + (f", correlations by {corr_gap:.3f}" if corr_gap is not None else "") + f"). {exact:.1%} of the synthetic rows are exact copies of a real row"
                        + (f" and {close:.1%} are very close to one." if close is not None else "."))}


def utility(real: pd.DataFrame, synth: pd.DataFrame, target: str, seed: int = 0) -> dict:
    """Entrena un modelo con los datos sintéticos y otro con los reales (la mitad) y prueba ambos con la otra mitad real."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import f1_score, r2_score
    from sklearn.model_selection import train_test_split

    from kikolens.analyze.modellab import make_pipeline, prepare_xy

    X, y, info = prepare_xy(real, target)
    task = info["task"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=seed)
    Xs, ys, _ = prepare_xy(synth, target)
    Xs = Xs[[c for c in X.columns if c in Xs.columns]]
    cols = list(Xs.columns)
    make = (lambda: HistGradientBoostingRegressor(max_iter=100, random_state=seed)) if task == "regression" else (lambda: HistGradientBoostingClassifier(max_iter=100, random_state=seed))
    score = (lambda t, p: r2_score(t, p)) if task == "regression" else (lambda t, p: f1_score(t, p, average="weighted"))
    real_model = make_pipeline(X_tr[cols], make()).fit(X_tr[cols], y_tr)
    synth_model = make_pipeline(Xs, make()).fit(Xs, ys)
    on_real = float(score(y_te, real_model.predict(X_te[cols])))
    from_synth = float(score(y_te, synth_model.predict(X_te[cols])))
    metric = "R2" if task == "regression" else "F1"
    return {"metric": metric, "trained_on_real": on_real, "trained_on_synthetic": from_synth, "ratio": from_synth / on_real if on_real else np.nan,
            "summary": f"Predicting {target}: {metric} {from_synth:.3f} when trained on synthetic data against {on_real:.3f} on real data "
                       f"({(from_synth / on_real if on_real else 0):.0%} of the real performance), both tested on real rows."}
