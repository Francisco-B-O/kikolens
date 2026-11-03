"""Modelos guardados y predicción por lotes. Un "paquete" (bundle) junta el modelo entrenado con lo necesario para
usarlo después con datos nuevos: el esquema de columnas de entrada, las clases, las métricas y el margen de error.
Los paquetes se guardan con joblib (pickle), así que solo deben abrirse si se confía en su origen."""
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from kikolens import __version__
from kikolens.analyze.quality import infer_types
from kikolens.config import home_dir
from kikolens.pipeline.recipe import Recipe
from kikolens.utils.dtypes import expand_datetime_columns

BUNDLE_KEY = "kikolens_model"
BUNDLE_VERSION = 1
EXTENSION = ".klmodel"
MAX_SCHEMA_VALUES = 300
MAX_LISTED = 8


# ── Esquema de entrada ───────────────────────────────────────────────────────
def input_schema(df: pd.DataFrame, columns) -> dict:
    """Describe cada columna de entrada tal como se vio al entrenar: tipo, rango o categorías y % de vacíos."""
    schema = {}
    for col in columns:
        s = df[col]
        info = {"missing_pct": round(float(s.isna().mean() * 100), 2)}
        if pd.api.types.is_datetime64_any_dtype(s):
            info.update(kind="datetime", min=str(s.min()), max=str(s.max()))
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            info.update(kind="numeric", min=float(s.min()), max=float(s.max()), median=float(s.median()),
                        mean=float(s.mean()), std=float(s.std() if len(s) > 1 else 0.0))
        else:
            values = s.dropna().astype(str)
            info.update(kind="categorical", values=sorted(values.unique())[:MAX_SCHEMA_VALUES],
                        n_unique=int(values.nunique()))
        schema[col] = info
    return schema


def _used_columns(df: pd.DataFrame, target: str, model_columns, mappings) -> list:
    """Columnas originales de las que depende el modelo (antes de fechas expandidas y variables ficticias)."""
    model_columns = set(model_columns)
    used = []
    for col in df.columns:
        if col == target:
            continue
        derived = {f"{col}_days", f"{col}_month", f"{col}_weekday"}
        dummies = any(m.startswith(f"{col}_") for m in model_columns) if col in mappings else False
        if col in model_columns or derived & model_columns or dummies:
            used.append(col)
    return used


def build_bundle(res: dict, df: pd.DataFrame, name: str | None = None, recipe: Recipe | None = None) -> dict:
    """Empaqueta el resultado de AutoML, del laboratorio o de la red neuronal para usarlo más tarde.
    `df` es el dataset con el que se entrenó (tal cual entró al modelo, con sus columnas originales)."""
    target, task = res["target"], res["task_type"]
    df, _ = infer_types(df)  # los modelos se entrenan con esos tipos, y el esquema debe describirlos igual
    if "pipeline" in res:
        kind, model_columns, mappings = "pipeline", list(res["features"]), {}
        estimator = res["pipeline"]
        label = res.get("best_model", "Model lab")
    else:
        kind, model_columns, mappings = "classic", list(res["columns"]), res.get("categorical_mappings", {})
        estimator = {"model": res["model"], "imputer": res["imputer"], "scaler": res["scaler"]}
        label = "Neural network" if "architecture" in res else "AutoML"

    final = estimator if kind == "pipeline" else estimator["model"]
    classes = res.get("classes") or (list(getattr(final, "classes_", [])) if task == "classification" else None)
    if task == "classification" and classes is not None:
        classes = [c.item() if hasattr(c, "item") else c for c in classes]
    label_encoded = kind == "classic" and bool(res.get("classes"))  # el objetivo de texto se entrenó como 0..n-1

    used = _used_columns(df, target, model_columns, mappings)
    if not used:
        raise ValueError("The model does not use any column of this dataset.")

    spread = None
    if kind == "pipeline" and task == "regression" and "X_test" in res:
        residuals = np.asarray(res["y_test"]) - estimator.predict(res["X_test"])
        spread = {"low": float(np.quantile(residuals, 0.05)), "high": float(np.quantile(residuals, 0.95))}

    return {
        BUNDLE_KEY: BUNDLE_VERSION, "kikolens_version": __version__, "name": name or f"{label} for {target}",
        "label": label, "kind": kind, "task": task, "target": target, "classes": classes,
        "label_encoded": label_encoded,
        "model_columns": model_columns, "mappings": mappings, "estimator": estimator,
        "schema": input_schema(df, used), "metrics": {k: float(v) for k, v in res["metrics"].items()
                                                      if isinstance(v, (int, float, np.floating))},
        "importance": {k: float(v) for k, v in list(res.get("feature_importance", {}).items())[:15]},
        "rows_trained": int(len(df)), "residual_spread": spread,
        "recipe": recipe.to_json() if recipe is not None and len(recipe) else None,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def metadata(bundle: dict) -> dict:
    """Todo lo del paquete salvo el modelo: es lo que se muestra en listados sin abrir el pickle."""
    keys = ("name", "label", "kind", "task", "target", "classes", "metrics", "importance", "rows_trained", "created",
            "kikolens_version")
    meta = {k: bundle.get(k) for k in keys}
    meta["inputs"] = {c: i["kind"] for c, i in bundle["schema"].items()}
    meta["has_recipe"] = bool(bundle.get("recipe"))
    return meta


# ── Preparación de datos nuevos ──────────────────────────────────────────────
def _coerce(series: pd.Series, kind: str) -> pd.Series:
    if kind == "numeric":
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            return series
        cleaned = series.astype(str).str.replace(r"[,%\s]", "", regex=True).where(series.notna())
        return pd.to_numeric(cleaned, errors="coerce")
    if kind == "datetime":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(series, errors="coerce")
    return series.astype(str).where(series.notna())


def prepare_inputs(bundle: dict, new_df: pd.DataFrame):
    """Alinea el dataset nuevo con el esquema de entrenamiento. Devuelve (entradas, informe)."""
    schema = bundle["schema"]
    present = [c for c in schema if c in new_df.columns]
    if not present:
        raise ValueError("None of the columns the model needs are in this data. Expected: "
                         + ", ".join(list(schema)[:MAX_LISTED]) + ("..." if len(schema) > MAX_LISTED else ""))
    frame = pd.DataFrame(index=new_df.index)
    report = {"missing_columns": [c for c in schema if c not in new_df.columns], "coerced": {},
              "unknown_categories": {}, "out_of_range": {}, "missing_values": {}}
    for col, info in schema.items():
        if col not in new_df.columns:
            frame[col] = pd.Series(np.nan, index=new_df.index, dtype="object" if info["kind"] == "categorical" else "float64")
            continue
        raw = new_df[col]
        value = _coerce(raw, info["kind"])
        lost = int(value.isna().sum() - raw.isna().sum())
        if lost > 0:
            report["coerced"][col] = lost
        frame[col] = value
        n_missing = int(value.isna().sum())
        if n_missing:
            report["missing_values"][col] = n_missing
        if info["kind"] == "categorical":
            known = set(info["values"])
            if info["n_unique"] <= MAX_SCHEMA_VALUES:
                unseen = sorted(set(value.dropna().unique()) - known)
                if unseen:
                    report["unknown_categories"][col] = {"count": int(value.isin(unseen).sum()), "values": unseen[:MAX_LISTED]}
        elif info["kind"] == "numeric":
            span = (info["max"] - info["min"]) or abs(info["max"]) or 1.0
            outside = int(((value < info["min"] - 0.05 * span) | (value > info["max"] + 0.05 * span)).sum())
            if outside:
                report["out_of_range"][col] = outside
        else:  # fechas: lo habitual es predecir un periodo posterior al entrenamiento
            first, last = pd.Timestamp(info["min"]), pd.Timestamp(info["max"])
            if first.tzinfo is not None:
                first, last = first.tz_localize(None), last.tz_localize(None)
            stamps = value.dt.tz_localize(None) if getattr(value.dt, "tz", None) is not None else value
            margin = (last - first) * 0.05
            outside = int(((stamps < first - margin) | (stamps > last + margin)).sum())
            if outside:
                report["out_of_range"][col] = outside
    return frame, report


def _model_matrix(bundle: dict, inputs: pd.DataFrame) -> pd.DataFrame:
    X = expand_datetime_columns(inputs)
    for col, values in bundle["mappings"].items():  # modelos clásicos: variables ficticias con las mismas columnas
        if col not in X.columns:
            continue
        for column in [c for c in bundle["model_columns"] if c.startswith(f"{col}_")]:
            X[column] = (X[col].astype(str) == column[len(col) + 1:]).astype(float)
        X = X.drop(columns=[col])
    for column in bundle["model_columns"]:
        if column not in X.columns:
            X[column] = np.nan
    return X[bundle["model_columns"]]


def _decode(bundle: dict, raw) -> np.ndarray:
    classes = bundle.get("classes")
    if bundle["task"] == "classification" and bundle.get("label_encoded") and classes:
        return np.asarray(classes, dtype=object)[np.asarray(raw, dtype=int)]
    return np.asarray(raw)


def predict_frame(bundle: dict, new_df: pd.DataFrame, apply_recipe: bool = False, interval: bool = True):
    """Predice cada fila. Devuelve (tabla con las predicciones añadidas, informe de calidad de la entrada)."""
    if apply_recipe:
        if not bundle.get("recipe"):
            raise ValueError("This model has no recipe stored with it.")
        new_df, _ = Recipe.from_json(bundle["recipe"]).apply(new_df)
    inputs, report = prepare_inputs(bundle, new_df)
    X = _model_matrix(bundle, inputs)
    out = new_df.copy()
    target = bundle["target"]

    if bundle["kind"] == "pipeline":
        pipe = bundle["estimator"]
        pred = pipe.predict(X)
        proba = pipe.predict_proba(X) if bundle["task"] == "classification" and hasattr(pipe, "predict_proba") else None
        classes = list(pipe.classes_) if proba is not None else None
    else:
        parts = bundle["estimator"]
        Z = parts["scaler"].transform(parts["imputer"].transform(X))
        model = parts["model"]
        pred = model.predict(Z)
        proba = model.predict_proba(Z) if bundle["task"] == "classification" and hasattr(model, "predict_proba") else None
        classes = list(_decode(bundle, model.classes_)) if proba is not None else None
        pred = _decode(bundle, pred) if bundle["task"] == "classification" else pred

    out[f"predicted_{target}"] = pred
    if proba is not None:
        out["confidence"] = proba.max(axis=1)
        if len(classes) <= 10:
            for i, label in enumerate(classes):
                out[f"probability_{label}"] = proba[:, i]
    spread = bundle.get("residual_spread")
    if interval and spread and bundle["task"] == "regression":
        out[f"{target}_low"] = pred + spread["low"]
        out[f"{target}_high"] = pred + spread["high"]
    report["rows"] = int(len(out))
    return out, report


def input_warnings(report: dict) -> list:
    """Frases listas para mostrar al usuario a partir del informe de `predict_frame`."""
    notes = []
    if report["missing_columns"]:
        notes.append(("high", f"Missing column(s) filled as empty: {', '.join(report['missing_columns'][:MAX_LISTED])}. "
                              "Predictions will be less reliable."))
    for col, n in report["coerced"].items():
        notes.append(("medium", f"'{col}': {n:,} value(s) could not be read and were treated as empty."))
    for col, info in report["unknown_categories"].items():
        notes.append(("medium", f"'{col}': {info['count']:,} row(s) have values never seen in training "
                                f"({', '.join(map(str, info['values']))})."))
    for col, n in report["out_of_range"].items():
        notes.append(("medium", f"'{col}': {n:,} row(s) are well outside the range seen in training."))
    gaps = {c: n for c, n in report["missing_values"].items() if c not in report["missing_columns"]}
    if gaps:
        notes.append(("low", "Empty values are filled the same way as during training: "
                             + ", ".join(f"{c} ({n:,})" for c, n in list(gaps.items())[:MAX_LISTED]) + "."))
    return notes


def model_card(bundle: dict) -> str:
    """Ficha del modelo en Markdown: qué predice, con qué datos, qué tan bien y qué entradas espera."""
    meta = metadata(bundle)
    lines = [f"# {meta['name']}", "",
             f"- **Predicts:** `{meta['target']}` ({meta['task']})",
             f"- **Algorithm:** {meta['label']}",
             f"- **Trained on:** {meta['rows_trained']:,} rows · {meta['created']} · KikoLens {meta['kikolens_version']}"]
    if meta["classes"]:
        lines.append(f"- **Classes:** {', '.join(map(str, meta['classes'][:20]))}")
    lines += ["", "## Held-out test results", ""]
    lines += [f"- {k}: {v:.4f}" for k, v in meta["metrics"].items()] or ["- not available"]
    if meta["importance"]:
        lines += ["", "## Most influential features", ""]
        lines += [f"- {k}: {v:.4f}" for k, v in list(meta["importance"].items())[:10]]
    lines += ["", "## Expected inputs", "", "| Column | Type | Training range / values |", "| :--- | :--- | :--- |"]
    for col, info in bundle["schema"].items():
        if info["kind"] == "numeric":
            span = f"{info['min']:.4g} to {info['max']:.4g}"
        elif info["kind"] == "datetime":
            span = f"{info['min'][:10]} to {info['max'][:10]}"
        else:
            span = ", ".join(info["values"][:6]) + (" ..." if info["n_unique"] > 6 else "")
        lines.append(f"| {col} | {info['kind']} | {span} |")
    lines += ["", "## Limits", "",
              "Predictions are only reliable for rows that look like the training data. Check the input warnings "
              "before trusting a batch, and retrain when the data changes."]
    return "\n".join(lines)


# ── Registro en disco ────────────────────────────────────────────────────────
def models_dir(directory=None) -> Path:
    return Path(directory) if directory else home_dir() / "models"


def slugify(name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", str(name)).strip("-").lower()
    return slug[:60] or "model"


def save_bundle(bundle: dict, path=None, directory=None) -> Path:
    """Guarda el paquete. Sin `path` entra en el registro (nunca pisa otra versión: añade -v2, -v3...)."""
    if path is not None:
        target = Path(path)
    else:
        folder = models_dir(directory)
        slug, n = slugify(bundle["name"]), 1
        while (folder / f"{slug}{'' if n == 1 else f'-v{n}'}{EXTENSION}").exists():
            n += 1
        target = folder / f"{slug}{'' if n == 1 else f'-v{n}'}{EXTENSION}"
        bundle = {**bundle, "name": bundle["name"] if n == 1 else f"{bundle['name']} (v{n})"}
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target)
    target.with_suffix(".json").write_text(json.dumps(metadata(bundle), indent=2, default=str), encoding="utf-8")
    return target


def load_bundle(path) -> dict:
    """Abre un paquete. Lee un pickle: usar solo con archivos propios o de confianza."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Model file not found: {path}")
    try:
        bundle = joblib.load(path)
    except Exception as e:
        raise ValueError(f"Could not read the model file: {e}") from None
    if not isinstance(bundle, dict) or BUNDLE_KEY not in bundle:
        raise ValueError("This file is a bare model, not a KikoLens model package. Save it again from KikoLens.")
    if bundle[BUNDLE_KEY] > BUNDLE_VERSION:
        raise ValueError("This model was saved by a newer version of KikoLens.")
    return bundle


def list_models(directory=None) -> pd.DataFrame:
    """Modelos del registro, leídos de los .json laterales (no se abre ningún pickle)."""
    rows = []
    for meta_path in sorted(models_dir(directory).glob("*.json")):
        if not meta_path.with_suffix(EXTENSION).exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = meta.get("metrics") or {}
        first = next(iter(metrics.items()), (None, None))
        rows.append({"file": meta_path.stem, "name": meta.get("name"), "target": meta.get("target"),
                     "task": meta.get("task"), "algorithm": meta.get("label"),
                     "score": f"{first[0]} {first[1]:.3f}" if first[0] else "", "created": meta.get("created")})
    columns = ["file", "name", "target", "task", "algorithm", "score", "created"]
    return pd.DataFrame(rows, columns=columns).sort_values("created", ascending=False, ignore_index=True)


def resolve_model(reference: str, directory=None) -> Path:
    """Acepta una ruta a un archivo o el nombre de un modelo del registro."""
    candidate = Path(reference)
    if candidate.is_file():
        return candidate
    stem = Path(reference).stem if str(reference).endswith(EXTENSION) else slugify(reference)
    registered = models_dir(directory) / f"{stem}{EXTENSION}"
    if registered.is_file():
        return registered
    raise ValueError(f"No model called '{reference}'. See the saved ones with: kikolens models")


def delete_model(reference: str, directory=None) -> Path:
    path = resolve_model(reference, directory)
    if path.parent != models_dir(directory):
        raise ValueError("Only models inside the KikoLens registry can be deleted this way.")
    path.unlink()
    path.with_suffix(".json").unlink(missing_ok=True)
    return path
