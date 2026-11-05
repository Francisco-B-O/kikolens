"""Detección de columnas de texto compatible con pandas 2 (object) y pandas 3 (dtype str)."""
import pandas as pd


def is_text(series: pd.Series) -> bool:
    return series.dtype == object or isinstance(series.dtype, pd.StringDtype)


def text_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if is_text(df[c])]


def categorical_columns(df: pd.DataFrame) -> list:
    """Texto y categorías: las columnas que se tratan como categóricas."""
    return [c for c in df.columns if is_text(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)]


def expand_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Los modelos no aceptan fechas: cada una pasa a días desde 1970, mes y día de la semana (devuelve una copia)."""
    out = df.copy()
    for col in out.select_dtypes(include=["datetime", "datetimetz"]).columns:
        stamps = out[col]
        if getattr(stamps.dt, "tz", None) is not None:
            stamps = stamps.dt.tz_localize(None)
        out[f"{col}_days"] = (stamps - pd.Timestamp("1970-01-01")).dt.days
        out[f"{col}_month"] = stamps.dt.month
        out[f"{col}_weekday"] = stamps.dt.dayofweek
        out = out.drop(columns=[col])
    return out
