"""Perfilado de calidad de datos y operaciones de limpieza reversibles (cada una devuelve un DataFrame nuevo)."""
import re
import warnings

import numpy as np
import pandas as pd

from kikolens.utils.dtypes import is_text, text_columns

_NUMERIC_TEXT = re.compile(r"^\s*[-+]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?\s*%?\s*$")
_BOOL_WORDS = {"true", "false", "yes", "no", "y", "n", "si", "sí", "0", "1"}


def _non_null_text(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).str.strip()


def looks_numeric_text(series: pd.Series, threshold: float = 0.95) -> bool:
    """Columna de texto cuyos valores son casi todos números ('1,200', '45%', ' 3.5 ')."""
    if not is_text(series):
        return False
    text = _non_null_text(series)
    text = text[text != ""]
    return len(text) > 0 and text.map(lambda v: bool(_NUMERIC_TEXT.match(v))).mean() >= threshold


def looks_datetime_text(series: pd.Series, threshold: float = 0.9) -> bool:
    """Columna de texto que se interpreta como fecha en casi todas las filas."""
    if not is_text(series):
        return False
    text = _non_null_text(series).head(500)
    if text.empty or text.map(lambda v: bool(_NUMERIC_TEXT.match(v))).mean() > 0.5:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(text, errors="coerce")
    return parsed.notna().mean() >= threshold


def looks_boolean_text(series: pd.Series) -> bool:
    if not is_text(series):
        return False
    values = set(_non_null_text(series).str.lower().unique())
    return 0 < len(values) <= 2 and values <= _BOOL_WORDS and not values <= {"0", "1"}


def count_outliers_iqr(series: pd.Series) -> int:
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return 0
    s = series.dropna()
    if s.empty:
        return 0
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def suspected_type(series: pd.Series) -> str | None:
    if looks_numeric_text(series):
        return "numeric"
    if looks_datetime_text(series):
        return "datetime"
    if looks_boolean_text(series):
        return "boolean"
    return None


_PROFILE_COLUMNS = ["column", "dtype", "missing", "missing_pct", "unique", "constant", "outliers", "suspected_type"]


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        unique = int(s.nunique(dropna=True))
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(s.isna().sum() / n * 100, 2),
            "unique": unique,
            "constant": unique <= 1,
            "outliers": count_outliers_iqr(s),
            "suspected_type": suspected_type(s),
        })
    return pd.DataFrame(rows, columns=_PROFILE_COLUMNS)


def profile_quality(df: pd.DataFrame) -> dict:
    """Resumen de calidad: puntuación 0-100, problemas detectados y perfil por columna."""
    n_rows, n_cols = df.shape
    cells = max(n_rows * n_cols, 1)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum()) if n_rows else 0
    columns = profile_columns(df)

    constant = columns.loc[columns["constant"], "column"].tolist() if n_cols else []
    mistyped = columns.loc[columns["suspected_type"].notna(), "column"].tolist() if n_cols else []
    heavy_missing = columns.loc[columns["missing_pct"] > 40, "column"].tolist() if n_cols else []

    missing_pct = missing / cells * 100
    dup_pct = duplicates / max(n_rows, 1) * 100
    penalty = (min(40, missing_pct * 2) + min(20, dup_pct * 2)
               + min(20, 5 * len(constant)) + min(20, 5 * len(mistyped)))
    score = round(max(0.0, 100.0 - penalty), 1)

    issues = []
    if missing:
        issues.append({"severity": "high" if missing_pct > 10 else "medium", "kind": "missing",
                       "message": f"{missing:,} missing cells ({missing_pct:.1f}%)."})
    for col in heavy_missing:
        issues.append({"severity": "high", "kind": "missing", "column": col,
                       "message": f"'{col}' is more than 40% empty; consider dropping it."})
    if duplicates:
        issues.append({"severity": "medium", "kind": "duplicates",
                       "message": f"{duplicates:,} duplicated rows ({dup_pct:.1f}%)."})
    for col in constant:
        issues.append({"severity": "low", "kind": "constant", "column": col,
                       "message": f"'{col}' has a single value and carries no information."})
    for _, r in columns[columns["suspected_type"].notna()].iterrows():
        issues.append({"severity": "medium", "kind": "type", "column": r["column"],
                       "message": f"'{r['column']}' is stored as text but looks {r['suspected_type']}."})
    for _, r in columns[columns["outliers"] > 0].iterrows():
        issues.append({"severity": "low", "kind": "outliers", "column": r["column"],
                       "message": f"'{r['column']}' has {r['outliers']:,} IQR outliers."})

    return {
        "rows": n_rows, "columns_count": n_cols, "missing_cells": missing, "missing_pct": round(missing_pct, 2),
        "duplicate_rows": duplicates, "constant_columns": constant, "mistyped_columns": mistyped,
        "score": score, "issues": issues, "columns": columns,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }


# ── Operaciones de limpieza ───────────────────────────────────────────────────
FILL_NUMERIC = ("mean", "median", "mode", "zero", "constant")
FILL_OTHER = ("mode", "constant")


def drop_duplicates(df):
    out = df.drop_duplicates().reset_index(drop=True)
    return out, f"Removed {len(df) - len(out):,} duplicated rows."


def drop_columns(df, columns):
    columns = [c for c in columns if c in df.columns]
    return df.drop(columns=columns), f"Dropped {len(columns)} column(s): {', '.join(map(str, columns)) or 'none'}."


def drop_constant_columns(df):
    const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    return df.drop(columns=const), f"Dropped {len(const)} constant column(s)."


def drop_missing_rows(df, subset=None):
    out = df.dropna(subset=subset).reset_index(drop=True)
    return out, f"Removed {len(df) - len(out):,} rows with missing values."


def fill_missing(df, column, strategy, value=None):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    s = df[column]
    numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    allowed = FILL_NUMERIC if numeric else FILL_OTHER
    if strategy not in allowed:
        raise ValueError(f"Strategy '{strategy}' is not valid for column '{column}'. Use one of {allowed}.")
    if strategy == "mean":
        fill = s.mean()
    elif strategy == "median":
        fill = s.median()
    elif strategy == "mode":
        mode = s.mode()
        if mode.empty:
            raise ValueError(f"Column '{column}' has no values to compute a mode from.")
        fill = mode.iloc[0]
    elif strategy == "zero":
        fill = 0
    else:
        if value is None:
            raise ValueError("A constant fill value is required.")
        fill = value
    n = int(s.isna().sum())
    out = df.copy()
    out[column] = s.fillna(fill)
    return out, f"Filled {n:,} missing value(s) in '{column}' using {strategy}."


def convert_type(df, column, target):
    """Convierte una columna a 'numeric', 'datetime', 'boolean' o 'text'."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    out = df.copy()
    s = out[column]
    if target == "numeric":
        cleaned = s.astype(str).str.replace(r"[,%\s]", "", regex=True).where(s.notna())
        out[column] = pd.to_numeric(cleaned, errors="coerce")
    elif target == "datetime":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[column] = pd.to_datetime(s, errors="coerce")
    elif target == "boolean":
        truthy = {"true", "yes", "y", "si", "sí", "1"}
        mapped = s.astype(str).str.strip().str.lower().map(lambda v: v in truthy)
        out[column] = mapped.where(s.notna())
    elif target == "text":
        out[column] = s.astype("string").astype(object).where(s.notna())
    else:
        raise ValueError(f"Unknown target type '{target}'.")
    lost = int(out[column].isna().sum() - s.isna().sum())
    note = f" ({lost:,} value(s) could not be converted)" if lost > 0 else ""
    return out, f"Converted '{column}' to {target}{note}."


def clip_outliers(df, column, factor=1.5):
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric.")
    s = df[column]
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    n = int(((s < lo) | (s > hi)).sum())
    out = df.copy()
    out[column] = s.clip(lo, hi)
    return out, f"Clipped {n:,} outlier(s) in '{column}' to [{lo:.4g}, {hi:.4g}]."


def strip_whitespace(df):
    out = df.copy()
    cols = text_columns(out)
    for c in cols:
        out[c] = out[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    return out, f"Trimmed whitespace in {len(cols)} text column(s)."


def standardize_column_names(df):
    def clean(name):
        name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()
        return name or "column"

    new, seen = [], {}
    for c in df.columns:
        base = clean(c)
        seen[base] = seen.get(base, 0) + 1
        new.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    out = df.copy()
    out.columns = new
    return out, "Standardized column names to snake_case."


OPERATIONS = {
    "drop_duplicates": drop_duplicates,
    "drop_columns": drop_columns,
    "drop_constant_columns": drop_constant_columns,
    "drop_missing_rows": drop_missing_rows,
    "fill_missing": fill_missing,
    "convert_type": convert_type,
    "clip_outliers": clip_outliers,
    "strip_whitespace": strip_whitespace,
    "standardize_column_names": standardize_column_names,
}


def apply_operation(df: pd.DataFrame, name: str, **params):
    """Ejecuta una operación por nombre. Devuelve (nuevo_df, mensaje); el original no se modifica."""
    if name not in OPERATIONS:
        raise ValueError(f"Unknown cleaning operation '{name}'.")
    return OPERATIONS[name](df, **params)


def auto_clean(df: pd.DataFrame):
    """Aplica las correcciones seguras: espacios, duplicados, tipos mal inferidos y columnas constantes."""
    log = []
    for name in ("strip_whitespace", "drop_duplicates"):
        df, msg = apply_operation(df, name)
        log.append(msg)
    for col in list(df.columns):
        kind = suspected_type(df[col])
        if kind:
            df, msg = convert_type(df, col, kind)
            log.append(msg)
    df, msg = drop_constant_columns(df)
    log.append(msg)
    return df, log
