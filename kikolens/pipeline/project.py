"""Proyectos de KikoLens: un único archivo `.kikolens` (zip) con los datos originales, la receta de pasos, los
datasets auxiliares, notas y, opcionalmente, los modelos entrenados. Los datos se guardan como Parquet, nunca como
pickle, así que abrir un proyecto ajeno no ejecuta código. Solo los modelos usan joblib y son opcionales."""
import io
import json
import zipfile
from datetime import datetime, timezone

import pandas as pd

from kikolens import __version__
from kikolens.pipeline.recipe import Recipe

EXTENSION = ".kikolens"
FORMAT_VERSION = 1
MAX_MEMBER_BYTES = 2 * 1024**3  # protege contra zips que se expanden de forma desmedida


def _parquet(df: pd.DataFrame) -> bytes:
    frame = df.copy()
    for col in frame.columns:  # Parquet no admite columnas de objetos mixtos
        if frame[col].dtype == object:
            frame[col] = frame[col].map(lambda v: v if v is None or isinstance(v, (str, bytes)) or pd.isna(v) else str(v))
    frame.columns = [str(c) for c in frame.columns]
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def save_project(original: pd.DataFrame, recipe: Recipe, name: str = "project", datasets: dict | None = None,
                 notes: str = "", source: str = "", models: dict | None = None) -> bytes:
    """Empaqueta el trabajo en bytes. `models` es {nombre: bytes de joblib} y es opcional."""
    manifest = {
        "kikolens_project": FORMAT_VERSION, "kikolens_version": __version__, "name": name, "source": source,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "notes": notes,
        "rows": int(len(original)), "columns": [str(c) for c in original.columns],
        "datasets": sorted((datasets or {}).keys()), "models": sorted((models or {}).keys()),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        zf.writestr("recipe.json", recipe.to_json())
        zf.writestr("data/original.parquet", _parquet(original))
        for key, frame in (datasets or {}).items():
            zf.writestr(f"datasets/{key}.parquet", _parquet(frame))
        for key, blob in (models or {}).items():
            zf.writestr(f"models/{key}.joblib", blob)
    return buffer.getvalue()


def _read(zf: zipfile.ZipFile, member: str) -> bytes:
    info = zf.getinfo(member)
    if info.file_size > MAX_MEMBER_BYTES:
        raise ValueError(f"'{member}' is too large to open safely.")
    return zf.read(member)


def load_project(blob: bytes, load_models: bool = False) -> dict:
    """Abre un proyecto. Devuelve manifest, original, recipe, datasets, notes y (si se pide) models en bytes."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        raise ValueError("This is not a KikoLens project file.") from None
    with zf:
        names = set(zf.namelist())
        if not {"manifest.json", "recipe.json", "data/original.parquet"} <= names:
            raise ValueError("This is not a KikoLens project file.")
        manifest = json.loads(_read(zf, "manifest.json"))
        if manifest.get("kikolens_project", 0) > FORMAT_VERSION:
            raise ValueError("This project was saved by a newer version of KikoLens.")
        original = pd.read_parquet(io.BytesIO(_read(zf, "data/original.parquet")))
        recipe = Recipe.from_json(_read(zf, "recipe.json").decode("utf-8"))
        datasets = {n[len("datasets/"):-len(".parquet")]: pd.read_parquet(io.BytesIO(_read(zf, n)))
                    for n in sorted(names) if n.startswith("datasets/") and n.endswith(".parquet")}
        models = {n[len("models/"):-len(".joblib")]: _read(zf, n)
                  for n in sorted(names) if load_models and n.startswith("models/") and n.endswith(".joblib")}
    return {"manifest": manifest, "original": original, "recipe": recipe, "datasets": datasets,
            "notes": manifest.get("notes", ""), "models": models}
