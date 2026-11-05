"""Limpieza avanzada: patrones de datos faltantes, imputación (KNN, iterativa, por grupo), categorías escritas de
varias formas y filas casi duplicadas. Las operaciones devuelven (df nuevo, mensaje) y se pueden repetir como pasos
de una receta; los imputadores se ajustan con los datos a los que se aplican."""
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from kikolens.utils.dtypes import is_text

IMPUTE_METHODS = ["knn", "iterative", "group"]
MAX_IMPUTE_ROWS = 50000
MAX_FEATURES = 30
MAX_UNIQUE_SIMILAR = 2000


def _numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


# ── Patrones de datos faltantes ──────────────────────────────────────────────
def missing_patterns(df: pd.DataFrame, top: int = 8) -> dict:
    """Qué columnas faltan juntas y si la ausencia depende de otras columnas (pista de que no es aleatoria)."""
    with_gaps = [c for c in df.columns if df[c].isna().any()]
    by_column = pd.DataFrame({"column": with_gaps, "missing": [int(df[c].isna().sum()) for c in with_gaps],
                              "missing %": [float(df[c].isna().mean() * 100) for c in with_gaps]})
    by_column = by_column.sort_values("missing", ascending=False, ignore_index=True)
    if not with_gaps:
        return {"by_column": by_column, "co_missing": pd.DataFrame(), "patterns": pd.DataFrame(), "related": pd.DataFrame(),
                "summary": "There are no missing values."}
    flags = df[with_gaps].isna()
    co = flags.astype(float).corr().fillna(0.0) if len(with_gaps) > 1 else pd.DataFrame()
    patterns = flags.apply(lambda r: ", ".join(c for c in with_gaps if r[c]) or "(complete)", axis=1)
    patterns = patterns.value_counts().rename_axis("missing together").reset_index(name="rows").head(top)
    patterns["share"] = patterns["rows"] / len(df)

    related = []
    for col in with_gaps:
        missing = flags[col]
        if missing.sum() < 5 or (~missing).sum() < 5:
            continue
        best = None
        for other in df.columns:
            if other == col:
                continue
            s = df[other]
            try:
                if _numeric(s):
                    a, b = s[missing].dropna(), s[~missing].dropna()
                    if len(a) < 5 or len(b) < 5:
                        continue
                    p = float(stats.mannwhitneyu(a, b).pvalue)
                    effect = abs(1 - 2 * stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)))
                elif is_text(s) and s.nunique() <= 20:
                    table = pd.crosstab(s.astype(str), missing)
                    if table.shape[0] < 2 or table.shape[1] < 2:
                        continue
                    chi2, p, _, _ = stats.chi2_contingency(table)[:4]
                    effect = float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))
                    p = float(p)
                else:
                    continue
            except ValueError:
                continue
            if best is None or p < best[1]:
                best = (other, p, effect)
        if best and best[1] < 0.001 and best[2] >= 0.2:
            related.append({"column": col, "depends on": best[0], "p": best[1], "effect": best[2]})
    related = pd.DataFrame(related, columns=["column", "depends on", "p", "effect"])
    summary = (f"{len(with_gaps)} column(s) have gaps. " +
               (f"For {len(related)} of them the gaps are not random: they depend on other columns, so filling them with "
                "a single typical value could bias the results." if len(related) else
                "No column predicts where the gaps are, so they look random."))
    return {"by_column": by_column, "co_missing": co, "patterns": patterns, "related": related, "summary": summary}


# ── Imputación ───────────────────────────────────────────────────────────────
def impute_advanced(df: pd.DataFrame, method: str = "knn", columns=None, n_neighbors: int = 5, group_by=None,
                    add_indicator: bool = False):
    """Rellena huecos: `knn` e `iterative` usan las demás columnas numéricas; `group` usa la mediana (o la moda) de cada
    grupo de `group_by`. `add_indicator` añade una columna 0/1 que marca las celdas rellenadas."""
    if method not in IMPUTE_METHODS:
        raise ValueError(f"Method must be one of {IMPUTE_METHODS}.")
    columns = list(columns) if columns else [c for c in df.columns if df[c].isna().any()]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {missing}")
    columns = [c for c in columns if df[c].isna().any()]
    if not columns:
        raise ValueError("None of the chosen columns has missing values.")
    out = df.copy()
    filled = {c: int(df[c].isna().sum()) for c in columns}
    flags = {c: df[c].isna() for c in columns}

    if method == "group":
        if not group_by or group_by not in df.columns:
            raise ValueError("Choose a column to form the groups.")
        for c in columns:
            if _numeric(df[c]):
                out[c] = df[c].fillna(df.groupby(group_by)[c].transform("median")).fillna(df[c].median())
            else:
                mode = df[c].mode()
                by_group = df.groupby(group_by)[c].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
                out[c] = df[c].fillna(by_group).fillna(mode.iloc[0] if not mode.empty else np.nan)
    else:
        bad = [c for c in columns if not _numeric(df[c])]
        if bad:
            raise ValueError(f"{method} works on numbers only; use the group method for: {', '.join(map(str, bad))}.")
        if len(df) > MAX_IMPUTE_ROWS:
            raise ValueError(f"{method} imputation is limited to {MAX_IMPUTE_ROWS:,} rows; use the group method instead.")
        numeric = [c for c in df.columns if _numeric(df[c]) and df[c].notna().any()]
        features = list(dict.fromkeys(columns + [c for c in numeric if c not in columns]))[:MAX_FEATURES]
        if len(features) < 2:
            raise ValueError("Imputing from other columns needs at least two numeric columns.")
        data = df[features].astype(float)
        scaler = StandardScaler().fit(data)
        scaled = pd.DataFrame(scaler.transform(data), columns=features, index=df.index)
        if method == "knn":
            imputer = KNNImputer(n_neighbors=int(max(1, min(n_neighbors, 30))), weights="distance")
        else:
            imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=15, random_state=0, sample_posterior=False)
        result = pd.DataFrame(scaler.inverse_transform(imputer.fit_transform(scaled)), columns=features, index=df.index)
        for c in columns:
            col = result[c]
            observed = df[c].dropna()
            if pd.api.types.is_integer_dtype(df[c]) or (len(observed) and (observed % 1 == 0).all()):
                col = col.round()  # un recuento o un entero sigue siendo entero
            out[c] = df[c].fillna(col)
    if add_indicator:
        for c in columns:
            out[f"{c}_was_missing"] = flags[c].astype(int)
    total = sum(filled.values())
    where = f" within groups of '{group_by}'" if method == "group" else ""
    return out, f"Filled {total:,} missing value(s) in {len(columns)} column(s) using {method}{where}."


# ── Categorías parecidas ─────────────────────────────────────────────────────
def _key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def similar_values(series: pd.Series, threshold: float = 0.85) -> pd.DataFrame:
    """Grupos de valores que parecen el mismo escrito de distinta forma ('New York', 'new york ', 'NewYork', 'New Yrok')."""
    if not 0.6 <= threshold <= 1.0:
        raise ValueError("The similarity threshold must be between 0.6 and 1.")
    counts = series.dropna().astype(str).value_counts()
    if len(counts) > MAX_UNIQUE_SIMILAR:
        raise ValueError(f"This column has {len(counts):,} distinct values; the limit is {MAX_UNIQUE_SIMILAR:,}.")
    keyed = {}
    for value, n in counts.items():
        keyed.setdefault(_key(value), []).append((value, int(n)))
    keys = sorted(keyed, key=lambda k: -sum(n for _, n in keyed[k]))
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if not a or not b or abs(len(a) - len(b)) > max(2, len(a) * (1 - threshold) + 1):
                continue
            if SequenceMatcher(None, a, b).ratio() >= threshold:
                parent[find(b)] = find(a)
    groups = {}
    for k in keys:
        groups.setdefault(find(k), []).extend(keyed[k])
    rows = []
    for members in groups.values():
        if len(members) > 1:
            members.sort(key=lambda m: -m[1])
            rows.append({"keep": members[0][0], "variants": [m[0] for m in members[1:]],
                         "rows affected": sum(m[1] for m in members[1:])})
    return pd.DataFrame(rows, columns=["keep", "variants", "rows affected"]).sort_values(
        "rows affected", ascending=False, ignore_index=True)


def standardize_categories(df: pd.DataFrame, column: str, threshold: float = 0.85):
    """Unifica las variantes de escritura de un mismo valor, dejando la forma más frecuente."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    if not is_text(df[column]):
        raise ValueError(f"'{column}' is not a text column.")
    groups = similar_values(df[column], threshold)
    if groups.empty:
        raise ValueError(f"No similar spellings found in '{column}' at this threshold.")
    mapping = {variant: row["keep"] for _, row in groups.iterrows() for variant in row["variants"]}
    out = df.copy()
    out[column] = df[column].map(lambda v: mapping.get(v, v) if isinstance(v, str) else v)
    return out, f"Merged {len(mapping)} spelling variant(s) in '{column}' into {len(groups)} value(s)."


# ── Filas casi duplicadas ────────────────────────────────────────────────────
def near_duplicate_rows(df: pd.DataFrame, columns=None, threshold: float = 0.9, max_rows: int = 5000, limit: int = 100) -> pd.DataFrame:
    """Pares de filas casi iguales (no idénticas), comparando su texto con n-gramas de caracteres."""
    if not 0.5 <= threshold < 1.0:
        raise ValueError("The similarity threshold must be between 0.5 and 1 (exclusive).")
    columns = list(columns) if columns else list(df.columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {missing}")
    if len(df) < 2:
        raise ValueError("At least two rows are needed.")
    if len(df) > max_rows:
        raise ValueError(f"This check is limited to {max_rows:,} rows. Filter or sample the data first.")
    text = df[columns].astype(str).agg(" | ".join, axis=1).str.lower()
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1).fit_transform(text)
    k = min(6, len(df))
    distance, neighbour = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X).kneighbors(X)
    seen, rows = set(), []
    for i in range(len(df)):
        for d, j in zip(distance[i][1:], neighbour[i][1:]):
            sim = 1 - d
            pair = (min(i, j), max(i, j))
            if sim >= threshold and pair not in seen and text.iloc[i] != text.iloc[j]:
                seen.add(pair)
                rows.append({"row A": df.index[pair[0]], "row B": df.index[pair[1]], "similarity": float(sim),
                             "A": text.iloc[pair[0]][:90], "B": text.iloc[pair[1]][:90]})
    out = pd.DataFrame(rows, columns=["row A", "row B", "similarity", "A", "B"])
    return out.sort_values("similarity", ascending=False, ignore_index=True).head(limit)


def drop_rows(df: pd.DataFrame, rows):
    """Elimina filas por su índice (por ejemplo, las que se confirmaron como duplicadas)."""
    rows = [r for r in rows if r in df.index]
    if not rows:
        raise ValueError("None of those rows exist.")
    return df.drop(index=rows).reset_index(drop=True), f"Removed {len(rows):,} row(s)."


OPERATIONS = {"impute_advanced": impute_advanced, "standardize_categories": standardize_categories}
