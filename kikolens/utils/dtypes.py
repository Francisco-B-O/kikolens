"""Detección de columnas de texto compatible con pandas 2 (object) y pandas 3 (dtype str)."""
import pandas as pd


def is_text(series: pd.Series) -> bool:
    return series.dtype == object or isinstance(series.dtype, pd.StringDtype)


def text_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if is_text(df[c])]


def categorical_columns(df: pd.DataFrame) -> list:
    """Texto y categorías: las columnas que se tratan como categóricas."""
    return [c for c in df.columns if is_text(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)]
