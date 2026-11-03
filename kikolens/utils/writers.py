"""Escritura de resultados: archivos (CSV, Excel, JSON, Parquet), libros de Excel con formato y bases de datos."""
import io
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

FORMATS = {".csv": "CSV", ".tsv": "TSV", ".xlsx": "Excel", ".json": "JSON", ".ndjson": "JSON lines", ".parquet": "Parquet",
           ".feather": "Feather"}
EXCEL_MAX_ROWS = 1_048_575
TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
IF_EXISTS = ("fail", "append", "replace")
HEADER_FILL = "F3F0E8"


def write_frame(df: pd.DataFrame, path) -> Path:
    """Guarda según la extensión. Excel sale con formato; el resto, tal cual."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in FORMATS:
        raise ValueError(f"Output must end in one of: {', '.join(FORMATS)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".tsv":
        df.to_csv(path, index=False, sep="\t")
    elif suffix == ".xlsx":
        path.write_bytes(build_workbook({"Data": df}))
    elif suffix == ".json":
        df.to_json(path, orient="records", date_format="iso", force_ascii=False, indent=2)
    elif suffix == ".ndjson":
        df.to_json(path, orient="records", lines=True, date_format="iso", force_ascii=False)
    elif suffix == ".feather":
        df.reset_index(drop=True).to_feather(path)
    else:
        df.to_parquet(path, index=False)
    return path


def frame_to_bytes(df: pd.DataFrame, suffix: str) -> bytes:
    """El contenido que tendría el archivo, sin dejar nada en disco (para botones de descarga)."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        return write_frame(df, Path(folder) / f"data{suffix}").read_bytes()


def _excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Excel no admite fechas con zona horaria, y los objetos mixtos se pasan a texto."""
    out = frame.copy()
    for col in out.columns:
        s = out[col]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            out[col] = s.dt.tz_localize(None)
        elif s.dtype == object or isinstance(s.dtype, pd.StringDtype):
            out[col] = s.map(lambda v: v if v is None or isinstance(v, (str, int, float, bool, datetime)) or pd.isna(v) else str(v))
    out.columns = [str(c) for c in out.columns]
    return out


def build_workbook(sheets: dict, notes: dict | None = None, title: str = "KikoLens export") -> bytes:
    """Libro de Excel con una hoja por DataFrame: cabecera destacada, filas fijas, filtros, anchos ajustados y formatos.
    Los textos que empiezan por = + - @ se guardan como texto, nunca como fórmula."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    if not sheets:
        raise ValueError("There is nothing to export.")
    wb = Workbook()
    wb.remove(wb.active)
    if notes:
        info = wb.create_sheet("About")
        info.append([title])
        info["A1"].font = Font(bold=True, size=14)
        info.append([f"Created {datetime.now():%Y-%m-%d %H:%M} with KikoLens"])
        info.append([])
        for key, value in notes.items():
            info.append([str(key), str(value)])
        info.column_dimensions["A"].width = 26
        info.column_dimensions["B"].width = 80
    used = set()
    for name, frame in sheets.items():
        if len(frame) > EXCEL_MAX_ROWS:
            raise ValueError(f"'{name}' has {len(frame):,} rows; Excel holds at most {EXCEL_MAX_ROWS:,}. Export to Parquet or CSV.")
        title_ = re.sub(r"[\[\]:*?/\\]", " ", str(name))[:31] or "Sheet"
        while title_ in used:
            title_ = title_[:28] + "_" + str(len(used))
        used.add(title_)
        ws = wb.create_sheet(title_)
        data = _excel_safe(frame)
        ws.append(list(data.columns))
        for row in data.itertuples(index=False, name=None):
            ws.append([None if (not isinstance(v, str) and pd.isna(v)) else v for v in row])
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor=HEADER_FILL)
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.freeze_panes = "A2"
        if len(data.columns) and len(data):
            ws.auto_filter.ref = f"A1:{get_column_letter(len(data.columns))}{len(data) + 1}"
        for i, col in enumerate(data.columns, start=1):
            letter = get_column_letter(i)
            sample = data[col].head(200).astype(str)
            ws.column_dimensions[letter].width = min(60, max(10, len(str(col)) + 2, int(sample.str.len().quantile(0.9)) + 2 if len(sample) else 10))
            kind = data[col].dtype
            for cell in ws[letter][1:]:
                if isinstance(cell.value, str) and cell.value[:1] in ("=", "+", "-", "@"):
                    cell.data_type = "s"          # texto, no fórmula
                elif pd.api.types.is_float_dtype(kind):
                    cell.number_format = "#,##0.00##"
                elif pd.api.types.is_datetime64_any_dtype(kind):
                    cell.number_format = "yyyy-mm-dd hh:mm" if (data[col].dt.normalize() != data[col]).any() else "yyyy-mm-dd"
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def write_database(df: pd.DataFrame, uri: str, table: str, if_exists: str = "fail", chunksize: int = 1000) -> int:
    """Escribe el DataFrame en una tabla. Por defecto no pisa una tabla existente; `replace` y `append` son explícitos."""
    if if_exists not in IF_EXISTS:
        raise ValueError(f"if_exists must be one of {IF_EXISTS}.")
    if not TABLE_NAME.match(str(table)):
        raise ValueError("The table name must start with a letter or underscore and use only letters, digits and underscores.")
    if df.empty or not len(df.columns):
        raise ValueError("There is no data to write.")
    uri = str(uri).strip()
    if uri.lower().startswith("sqlite:") and not uri.lower().startswith("sqlite://"):
        uri = "sqlite:///" + uri.split(":", 1)[1].lstrip("/")
    from sqlalchemy import create_engine

    frame = df.copy()
    frame.columns = [str(c) for c in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("Column names must be unique to write a table.")
    try:
        engine = create_engine(uri)
        frame.to_sql(table, engine, if_exists=if_exists, index=False, chunksize=chunksize, method="multi" if "sqlite" not in uri else None)
    except ValueError as e:
        raise ValueError(str(e)) from None
    except ImportError as e:
        raise ValueError(f"The database driver is missing: {e}") from None
    except Exception as e:
        raise ValueError(f"Could not write to the database: {e}") from None
    finally:
        try:
            engine.dispose()
        except Exception:
            pass
    return int(len(frame))
