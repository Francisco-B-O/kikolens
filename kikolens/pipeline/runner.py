"""Ejecutor de pipelines: un archivo JSON describe qué hacer con unos datos (receta, reglas de validación, comparación con una
referencia, puntuación con un modelo) y dónde dejar los resultados. Pensado para programarlo a diario o en CI:
`kikolens run pipeline.json` devuelve 0 si todo va bien, 2 si salta una condición de `fail_on` y 1 si algo no se pudo leer.

Ejemplo:
{"source": "data.csv", "recipe": "recipe.json", "rules": "rules.json", "baseline": "training.csv", "model": "sales-model",
 "outputs": {"data": "clean.csv", "predictions": "scored.csv", "report": "report.html"}, "fail_on": ["rules", "drift"]}"""
import json
from pathlib import Path

import pandas as pd

KNOWN = {"source", "recipe", "rules", "baseline", "model", "outputs", "fail_on", "min_quality", "name"}
OUTPUTS = {"data", "predictions", "report", "log"}
CONDITIONS = {"rules", "drift", "quality", "input"}


def _path(value: str, base: Path) -> str:
    """Un archivo relativo se busca junto al pipeline; una URL o una ruta que no existe ahí se deja como está."""
    if "://" in str(value):
        return str(value)
    candidate = base / value
    return str(candidate) if candidate.exists() or not Path(value).exists() else str(value)


def load_spec(path) -> tuple:
    path = Path(path)
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Could not read the pipeline file: {e}") from None
    if not isinstance(spec, dict) or "source" not in spec:
        raise ValueError("The pipeline file must be a JSON object with at least a \"source\".")
    unknown = set(spec) - KNOWN
    if unknown:
        raise ValueError(f"Unknown pipeline key(s): {', '.join(sorted(unknown))}. Known: {', '.join(sorted(KNOWN))}.")
    bad = set(spec.get("outputs", {})) - OUTPUTS
    if bad:
        raise ValueError(f"Unknown output(s): {', '.join(sorted(bad))}. Known: {', '.join(sorted(OUTPUTS))}.")
    bad = set(spec.get("fail_on", [])) - CONDITIONS
    if bad:
        raise ValueError(f"Unknown fail_on condition(s): {', '.join(sorted(bad))}. Known: {', '.join(sorted(CONDITIONS))}.")
    return spec, path.parent


def run_pipeline(path) -> dict:
    """Ejecuta el pipeline. Devuelve `steps` (lo hecho), `alerts` (condiciones de fail_on que saltaron), `status` y `written`."""
    from kikolens.analyze import drift
    from kikolens.analyze.predict import input_warnings, load_bundle, predict_frame, resolve_model
    from kikolens.analyze.quality import infer_types, profile_quality
    from kikolens.analyze.rules import RuleSet
    from kikolens.pipeline.recipe import Recipe
    from kikolens.report.builder import save_report
    from kikolens.utils.loaders import load_data
    from kikolens.utils.writers import write_frame

    spec, base = load_spec(path)
    fail_on = set(spec.get("fail_on", []))
    steps, alerts, written = [], [], []

    def log(step, status, message):
        steps.append({"step": step, "status": status, "message": message})

    def alert(condition, message):
        log(condition, "alert" if condition in fail_on else "warning", message)
        if condition in fail_on:
            alerts.append(message)

    def out(key):
        value = spec.get("outputs", {}).get(key)
        return _resolve_out(value, base) if value else None

    df = load_data(_path(spec["source"], base))
    log("load", "ok", f"Read {len(df):,} rows and {df.shape[1]} columns from {spec['source']}.")
    recipe = None
    if spec.get("recipe"):
        recipe = Recipe.from_json(Path(_path(spec["recipe"], base)).read_text(encoding="utf-8"))
        df, messages = recipe.apply(df)
        log("recipe", "ok", f"Applied {len(messages)} step(s): the data now has {len(df):,} rows and {df.shape[1]} columns.")
    typed, _ = infer_types(df)

    quality = profile_quality(typed)
    log("quality", "ok", f"Quality score {quality['score']:.0f}/100.")
    if spec.get("min_quality") is not None and quality["score"] < float(spec["min_quality"]):
        alert("quality", f"Quality score {quality['score']:.0f} is below the minimum of {spec['min_quality']}.")

    if spec.get("rules"):
        result = RuleSet.from_json(Path(_path(spec["rules"], base)).read_text(encoding="utf-8")).validate(typed)
        log("rules", "ok" if result["passed"] else "warning", result["summary"])
        if not result["passed"]:
            alert("rules", f"{result['rows_with_problems']:,} row(s) break a validation rule.")

    if spec.get("baseline"):
        reference = load_data(_path(spec["baseline"], base))
        if recipe is not None:
            reference, _ = recipe.apply(reference)
        change = drift.compare_datasets(infer_types(reference)[0], typed)
        log("drift", "ok" if change["level"] == "low" else "warning", f"Drift against the baseline: {change['level']} "
            f"({change['drifted']} column(s) drifted, {change['moderate']} moderate).")
        if change["level"] == "high":
            alert("drift", f"The data drifted a lot from the baseline ({change['drifted']} column(s)).")

    predictions = None
    if spec.get("model"):
        bundle = load_bundle(resolve_model(_path(spec["model"], base)))
        predictions, report = predict_frame(bundle, df)
        log("score", "ok", f"Scored {len(predictions):,} rows with '{bundle['name']}'.")
        for severity, message in input_warnings(report):
            if severity in ("high", "medium"):
                alert("input", message)

    if out("data"):
        write_frame(df, out("data"))
        written.append(str(out("data")))
    if predictions is not None and out("predictions"):
        write_frame(predictions, out("predictions"))
        written.append(str(out("predictions")))
    if out("report"):
        save_report(df, out("report"), name=spec.get("name") or Path(str(spec["source"])).stem or "dataset")
        written.append(str(out("report")))
    status = "alert" if alerts else "ok"
    result = {"status": status, "steps": steps, "alerts": alerts, "written": written, "rows": len(df)}
    if out("log"):
        Path(out("log")).write_text(json.dumps({**result, "finished": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")},
                                               indent=2), encoding="utf-8")
        written.append(str(out("log")))
    return result


def _resolve_out(value: str, base: Path) -> Path:
    target = Path(value)
    target = target if target.is_absolute() else base / target
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
