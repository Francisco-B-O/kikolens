"""Diccionario de datos: una ficha por columna (qué tipo de dato es, cuántos vacíos, valores típicos, rangos y avisos) que se puede
descargar como Markdown o Excel y entregar junto a los datos. Las descripciones las escribe la persona; el resto se calcula."""
import re

import numpy as np
import pandas as pd

from kikolens.analyze import privacy
from kikolens.analyze.autopilot import column_roles
from kikolens.analyze.quality import infer_types, suspected_type

ROLE_NAMES = {"number": "number", "category": "category", "date": "date", "identifier": "identifier", "text": "free text",
              "constant": "constant"}


def _example(series: pd.Series, n: int = 3) -> str:
    values = series.dropna()
    if values.empty:
        return ""
    top = values.astype(str).value_counts().head(n).index
    return ", ".join(v if len(v) <= 24 else v[:23] + "…" for v in top)


def build_dictionary(df: pd.DataFrame, descriptions: dict | None = None) -> pd.DataFrame:
    """Una fila por columna."""
    if df.shape[1] == 0:
        raise ValueError("The data has no columns.")
    descriptions = descriptions or {}
    typed, _ = infer_types(df)
    roles = column_roles(typed)
    personal = set(privacy.scan(df)["column"])
    rows = []
    for col in df.columns:
        s, t = df[col], typed[col]
        role = roles[col]
        row = {"column": col, "description": descriptions.get(col, ""), "meaning": ROLE_NAMES.get(role, role), "stored as": str(s.dtype),
               "empty %": round(float(s.isna().mean() * 100), 1), "distinct": int(s.nunique(dropna=True)), "typical values": _example(s)}
        if pd.api.types.is_numeric_dtype(t) and not pd.api.types.is_bool_dtype(t):
            row.update({"min": float(t.min()), "max": float(t.max()), "mean": float(t.mean())})
        elif pd.api.types.is_datetime64_any_dtype(t):
            row.update({"min": str(t.min().date()), "max": str(t.max().date())})
        notes = []
        if s.isna().mean() > 0.4:
            notes.append("mostly empty")
        if role == "constant":
            notes.append("single value")
        if role == "identifier":
            notes.append("different in every row")
        if suspected_type(s):
            notes.append(f"stored as text but looks {suspected_type(s)}")
        if col in personal:
            notes.append("may contain personal data")
        row["notes"] = "; ".join(notes)
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame, title: str = "Data dictionary", rows: int | None = None) -> str:
    """El diccionario como tabla Markdown (con los `|` escapados)."""
    def cell(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        text = f"{value:,.4g}" if isinstance(value, float) else str(value)
        return re.sub(r"\s+", " ", text).replace("|", "\\|")

    columns = list(table.columns)
    lines = [f"# {title}", ""]
    if rows is not None:
        lines += [f"{rows:,} rows, {len(table)} columns.", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, r in table.iterrows():
        lines.append("| " + " | ".join(cell(r[c]) for c in columns) + " |")
    return "\n".join(lines) + "\n"
