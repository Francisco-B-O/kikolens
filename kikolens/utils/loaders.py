import csv
import io
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

SQL_PREFIXES = ("postgresql://", "postgres://", "mysql://", "sqlite:", "oracle://", "mssql://", "mariadb://")
REMOTE_PREFIXES = ("s3://", "gs://", "gcs://", "az://", "abfs://")
_SHEETS = re.compile(r"^https?://docs\.google\.com/spreadsheets/d/([A-Za-z0-9_-]+)")
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


def google_sheets_csv_url(url: str) -> str | None:
    """Convierte un enlace de Google Sheets (compartido para quien tenga el enlace) en su descarga CSV."""
    match = _SHEETS.match(url)
    if not match:
        return None
    gid = re.search(r"[#&?]gid=(\d+)", url)
    return f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=csv" + (f"&gid={gid.group(1)}" if gid else "")


def list_sheets(source) -> list:
    """Nombres de las hojas de un libro de Excel (fichero subido o ruta)."""
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        names = pd.ExcelFile(source).sheet_names
        if hasattr(source, "seek"):
            source.seek(0)
        return list(names)
    except Exception as e:
        raise ValueError(f"Could not read the Excel file: {e}") from None


def load_text(text: str, sep: str | None = None) -> pd.DataFrame:
    """Tabla pegada como texto (de Excel, de una web...). Detecta el separador: tabulador, coma, punto y coma o barra."""
    text = str(text).strip("\n")
    if not text.strip():
        raise ValueError("There is nothing to read: paste some rows first.")
    if len(text) > 50_000_000:
        raise ValueError("The pasted text is too large; upload a file instead.")
    if sep is None:
        try:
            sep = csv.Sniffer().sniff(text.splitlines()[0] if "\n" in text else text, delimiters="\t,;|").delimiter
        except csv.Error:
            sep = "\t" if "\t" in text else ","
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep)
    except Exception as e:
        raise ValueError(f"Could not read the pasted table: {e}") from None
    if df.empty or len(df.columns) < 1:
        raise ValueError("No rows were found in the pasted text.")
    return df


def list_tables(uri: str) -> list:
    """Tablas de una base de datos."""
    return _database(uri, lambda engine, inspect: inspect(engine).get_table_names())


def _database(uri: str, action):
    if uri.lower().startswith("sqlite:") and not uri.lower().startswith("sqlite://"):
        uri = "sqlite:///" + uri.split(":", 1)[1].lstrip("/")
    try:
        from sqlalchemy import create_engine, inspect

        engine = create_engine(uri)
        try:
            return action(engine, inspect)
        finally:
            engine.dispose()
    except ImportError as ie:
        if "mysql" in uri:
            raise ImportError("MySQL Driver missing. Please run: pip install pymysql cryptography")
        if "postgres" in uri:
            raise ImportError("PostgreSQL Driver missing. Please run: pip install psycopg2-binary")
        raise ie
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Database connection failed: {e}")


def _load_remote(url: str) -> pd.DataFrame:
    """Almacenamiento en la nube (s3://, gs://...): necesita fsspec y su complemento (s3fs, gcsfs...)."""
    name = urlparse(url).path
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {url}")
    try:
        import fsspec

        with fsspec.open(url, "rb") as handle:
            return _read_by_extension(name, io.BytesIO(handle.read()))
    except ImportError:
        scheme = url.split("://")[0]
        extra = {"s3": "s3fs", "gs": "gcsfs", "gcs": "gcsfs", "az": "adlfs", "abfs": "adlfs"}.get(scheme, "fsspec")
        raise ValueError(f"Reading {scheme}:// needs an extra package. Run: pip install fsspec {extra}") from None
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not read {url}: {e}") from None


def _read_by_extension(name: str, source, sheet=None) -> pd.DataFrame:
    ext = Path(name).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(source)
    if ext == ".tsv":
        return pd.read_csv(source, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(source, sheet_name=sheet if sheet is not None else 0)
    if ext in (".json", ".ndjson", ".jsonl"):
        return _read_json(source)
    if ext == ".parquet":
        return pd.read_parquet(source)
    frame = _plugin_loader(name, source)
    if frame is not None:
        return frame
    raise ValueError(f"Unsupported source format: {name}")


def _plugin_loader(name: str, source):
    """Los formatos que añade un plugin (ver `kikolens.plugins`). Un archivo subido se guarda un momento en disco porque los lectores trabajan con rutas."""
    from kikolens import plugins

    registered = next((ext for ext in plugins.loaders() if ext == Path(name).suffix.lower()), None)       # la extensión sale de la lista registrada, no del texto recibido
    if registered is None:
        return None
    if isinstance(source, str):
        return plugins.read_with_loader(source)
    if not isinstance(source, io.IOBase):                                  # solo un archivo subido de verdad (nunca texto, que podría ser una cadena de conexión)
        return None
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / ("upload" + registered)
        target.write_bytes(source.read())
        return plugins.read_with_loader(str(target))


def _load_from_database(uri: str, table: str | None = None, query: str | None = None) -> pd.DataFrame:
    """Una tabla (la primera si no se indica) o una consulta de solo lectura."""
    def read(engine, inspect):
        if query:
            from kikolens.sql.runner import validate_read_only

            return pd.read_sql(validate_read_only(query), engine)
        tables = inspect(engine).get_table_names()
        if not tables:
            raise ValueError("No tables found in the database.")
        chosen = table or tables[0]
        if chosen not in tables:
            raise ValueError(f"Table '{chosen}' not found. Available: {', '.join(tables)}")
        return pd.read_sql(f"SELECT * FROM {engine.dialect.identifier_preparer.quote(chosen)}", engine)

    return _database(uri, read)


def _load_from_url(url: str) -> pd.DataFrame:
    import requests

    sheets = google_sheets_csv_url(url)
    name = "sheet.csv" if sheets else urlparse(url).path
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported URL format: {url}")
    try:
        response = requests.get(sheets or url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Could not download {url}: {e}")
    if sheets and "text/html" in response.headers.get("content-type", ""):
        raise ValueError("Google Sheets did not give the data. Share the sheet so that anyone with the link can view it.")
    return _read_by_extension(name, io.BytesIO(response.content))


def load_data(source, sheet=None, table: str | None = None, query: str | None = None) -> pd.DataFrame:
    """Carga un DataFrame desde un fichero subido, ruta local, URL http(s), Google Sheets, s3:// / gs://, o una base
    de datos (`table` o `query` de solo lectura). `sheet` elige la hoja de un libro de Excel."""
    if hasattr(source, "name") and hasattr(source, "read"):
        return _read_by_extension(source.name, source, sheet)

    text = str(source).strip()
    lowered = text.lower()

    if lowered.startswith(("http://", "https://")):
        return _load_from_url(text)
    if lowered.startswith(REMOTE_PREFIXES):
        return _load_remote(text)
    if lowered.startswith(SQL_PREFIXES) or "://" in text:
        return _load_from_database(text, table, query)

    return _read_by_extension(text, text, sheet)
