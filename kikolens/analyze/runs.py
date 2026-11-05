"""Historial de experimentos: cada vez que se entrena en el laboratorio queda un registro (datos, columna a predecir, parámetros,
métricas) en `~/.kikolens/runs`, para comparar intentos, ver si se mejora con el tiempo y no perder el que funcionó mejor."""
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from kikolens import __version__

HIGHER_IS_BETTER = {"R2", "Accuracy", "Balanced accuracy", "F1 (weighted)", "ROC AUC"}


def runs_dir(directory=None) -> Path:
    base = Path(directory) if directory else Path(os.environ.get("KIKOLENS_HOME", Path.home() / ".kikolens")) / "runs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _clean(value):
    """Solo JSON: números, textos, listas y diccionarios; el resto, como texto."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return None if value != value else value
    if hasattr(value, "item"):
        try:
            return _clean(value.item())
        except Exception:
            pass
    return str(value)


def log_run(kind: str, target: str, metrics: dict, params: dict | None = None, name: str = "", dataset: str = "", notes: str = "",
            directory=None, **extra) -> dict:
    """Guarda un registro y lo devuelve. Los nombres de los archivos llevan la fecha, así que salen ordenados."""
    if not kind or not target:
        raise ValueError("A run needs a kind and a target.")
    record = {"id": uuid.uuid4().hex[:8], "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind, "target": target,
              "name": name, "dataset": dataset, "notes": notes, "kikolens_version": __version__,
              "metrics": {k: v for k, v in _clean(metrics).items() if isinstance(v, (int, float))}, "params": _clean(params or {}), **_clean(extra)}
    stamp = re.sub(r"[^0-9]", "", record["created"])
    (runs_dir(directory) / f"{stamp}-{record['id']}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def log_lab(res: dict, dataset: str = "", params: dict | None = None, directory=None) -> dict:
    """Registra el resultado de `run_model_lab`."""
    board = res["leaderboard"]
    tuning = res.get("tuning") or {}
    return log_run("lab", res["target"], {**res["metrics"], f"CV {res['primary_metric']}": float(board.loc[board["model"] == res["best_model"], res["primary_metric"]].iloc[0])},
                   {**(params or {}), **({"tuning": tuning.get("params"), "tuner": tuning.get("method")} if tuning else {})},
                   name=res["best_model"], dataset=dataset, task=res["task_type"], rows=res["n_train"] + res["n_test"],
                   features=len(res["features"]), primary_metric=res["primary_metric"], directory=directory)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_runs(target: str | None = None, kind: str | None = None, directory=None) -> pd.DataFrame:
    """Un registro por fila, el más reciente primero, con las métricas como columnas."""
    rows = []
    for path in sorted(runs_dir(directory).glob("*.json"), reverse=True):
        record = _read(path)
        if not record or "id" not in record:
            continue
        if (target and record.get("target") != target) or (kind and record.get("kind") != kind):
            continue
        rows.append({"id": record["id"], "created": record["created"], "kind": record["kind"], "target": record["target"],
                     "model": record.get("name", ""), "dataset": record.get("dataset", ""), "task": record.get("task", ""),
                     "rows": record.get("rows"), **record.get("metrics", {})})
    return pd.DataFrame(rows)


def get_run(run_id: str, directory=None) -> dict:
    for path in runs_dir(directory).glob(f"*-{run_id}.json"):
        record = _read(path)
        if record:
            return record
    raise ValueError(f"No run with id '{run_id}'.")


def delete_run(run_id: str, directory=None) -> None:
    found = list(runs_dir(directory).glob(f"*-{run_id}.json"))
    if not found:
        raise ValueError(f"No run with id '{run_id}'.")
    for path in found:
        path.unlink()


def compare(run_ids: list, directory=None) -> pd.DataFrame:
    """Las métricas y los parámetros de varios registros, uno por columna; solo las filas en que difieren llevan la marca."""
    if len(run_ids) < 2:
        raise ValueError("Choose at least two runs to compare.")
    records = [get_run(i, directory) for i in run_ids]
    rows = {}
    for r in records:
        flat = {**{f"metric: {k}": v for k, v in r["metrics"].items()}, **{f"param: {k}": json.dumps(v) if isinstance(v, (dict, list)) else v
                                                                          for k, v in r["params"].items()},
                "model": r.get("name"), "dataset": r.get("dataset"), "created": r["created"]}
        rows[r["id"]] = flat
    table = pd.DataFrame(rows)
    table["differs"] = table.apply(lambda row: len({str(v) for v in row if not (isinstance(v, float) and v != v)}) > 1, axis=1)
    return table


def best_run(target: str, metric: str | None = None, directory=None) -> dict:
    """El mejor registro para una columna según una métrica (por defecto la primaria de cada uno)."""
    frame = list_runs(target, directory=directory)
    if frame.empty:
        raise ValueError(f"No runs for '{target}' yet.")
    metric = metric or next((m for m in ("R2", "F1 (weighted)", "Accuracy") if m in frame.columns and frame[m].notna().any()), None)
    if metric is None or metric not in frame.columns:
        raise ValueError("The runs have no common metric to rank by.")
    ordered = frame.dropna(subset=[metric]).sort_values(metric, ascending=metric not in HIGHER_IS_BETTER)
    return get_run(ordered.iloc[0]["id"], directory)
