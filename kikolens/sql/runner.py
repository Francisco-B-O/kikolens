import re
from pathlib import Path

import duckdb
import pandas as pd

_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRINGS = re.compile(r"'[^']*(?:''[^']*)*'|\"[^\"]*(?:\"\"[^\"]*)*\"")
_ALLOWED_START = ("SELECT", "WITH", "VALUES", "DESCRIBE", "SUMMARIZE", "SHOW")
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|ATTACH|DETACH|COPY|EXPORT|INSTALL|PRAGMA|TRUNCATE)\b"
    r"|\b(read_\w+|glob|parquet_scan|sniff_csv)\s*\(",
    re.IGNORECASE,
)


def validate_read_only(query: str) -> str:
    """Valida que la consulta sea una única sentencia de solo lectura y devuelve la consulta limpia."""
    cleaned = _COMMENTS.sub(" ", query).strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("KikoLens Security: Empty query.")

    skeleton = _STRINGS.sub("''", cleaned)
    if ";" in skeleton:
        raise ValueError("KikoLens Security: Only a single statement is allowed.")
    if not skeleton.upper().startswith(_ALLOWED_START):
        raise ValueError("KikoLens Security: Only read-only queries (SELECT) are allowed.")
    if _FORBIDDEN.search(skeleton):
        raise ValueError("KikoLens Security: Query contains a forbidden keyword.")
    return cleaned


def _register_source(con, source) -> None:
    if isinstance(source, pd.DataFrame):
        con.register("df", source)
        return

    path = Path(str(source))
    suffix = path.suffix.lower()
    escaped = path.as_posix().replace("'", "''")
    if suffix == ".csv":
        con.execute(f"CREATE VIEW df AS SELECT * FROM read_csv_auto('{escaped}')")
    elif suffix == ".parquet":
        con.execute(f"CREATE VIEW df AS SELECT * FROM read_parquet('{escaped}')")
    elif suffix in (".xlsx", ".xls"):
        con.register("df", pd.read_excel(path))
    elif suffix == ".json":
        con.register("df", pd.read_json(path))
    else:
        raise ValueError(f"Unsupported file type: {suffix or path.name}")


def run_sql_query(filepath_or_df, query: str) -> pd.DataFrame:
    """Ejecuta una consulta de solo lectura sobre un DataFrame o fichero, expuesto como la tabla `df`."""
    try:
        safe_query = validate_read_only(query)
    except ValueError as e:
        raise ValueError(str(e))

    con = duckdb.connect(database=":memory:")
    try:
        _register_source(con, filepath_or_df)
        return con.execute(safe_query).df()
    except Exception as e:
        raise ValueError(f"SQL Execution Failed: {e}")
    finally:
        con.close()
