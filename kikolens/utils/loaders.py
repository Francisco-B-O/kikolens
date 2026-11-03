import io
import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

SQL_PREFIXES = ("postgresql://", "postgres://", "mysql://", "sqlite:", "oracle://", "mssql://", "mariadb://")
SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".json", ".ndjson", ".jsonl", ".xlsx", ".xls", ".parquet")


def _read_json(source) -> pd.DataFrame:
    """JSON estándar primero; si falla, NDJSON (un objeto por línea)."""
    try:
        return pd.read_json(source)
    except ValueError:
        if hasattr(source, "seek"):
            source.seek(0)
            lines = source.read()
            lines = lines.decode("utf-8") if isinstance(lines, bytes) else lines
            lines = lines.splitlines()
        else:
            with open(source, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        return pd.DataFrame([json.loads(line) for line in lines if line.strip()])


def _read_by_extension(name: str, source) -> pd.DataFrame:
    ext = Path(name).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(source)
    if ext == ".tsv":
        return pd.read_csv(source, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(source)
    if ext in (".json", ".ndjson", ".jsonl"):
        return _read_json(source)
    if ext == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"Unsupported source format: {name}")


def _load_from_database(uri: str) -> pd.DataFrame:
    if uri.lower().startswith("sqlite:") and not uri.lower().startswith("sqlite://"):
        uri = "sqlite:///" + uri.split(":", 1)[1].lstrip("/")

    try:
        from sqlalchemy import create_engine, inspect

        engine = create_engine(uri)
        tables = inspect(engine).get_table_names()
        if not tables:
            raise ValueError("No tables found in the database.")
        quoted = engine.dialect.identifier_preparer.quote(tables[0])
        return pd.read_sql(f"SELECT * FROM {quoted}", engine)
    except ImportError as ie:
        if "mysql" in uri:
            raise ImportError("MySQL Driver missing. Please run: pip install pymysql cryptography")
        if "postgres" in uri:
            raise ImportError("PostgreSQL Driver missing. Please run: pip install psycopg2-binary")
        raise ie
    except Exception as e:
        raise ValueError(f"Database connection failed: {e}")


def _load_from_url(url: str) -> pd.DataFrame:
    import requests

    name = urlparse(url).path
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported URL format: {url}")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Could not download {url}: {e}")
    return _read_by_extension(name, io.BytesIO(response.content))


def load_data(source) -> pd.DataFrame:
    """Carga un DataFrame desde un fichero subido, ruta local, URL http(s) o URI de base de datos."""
    if hasattr(source, "name") and hasattr(source, "read"):
        return _read_by_extension(source.name, source)

    text = str(source).strip()
    lowered = text.lower()

    if lowered.startswith(("http://", "https://")):
        return _load_from_url(text)
    if lowered.startswith(SQL_PREFIXES) or "://" in text:
        return _load_from_database(text)

    return _read_by_extension(text, text)
