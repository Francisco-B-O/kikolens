"""Monitorización: vigila un archivo de datos que se actualiza, lo compara con una línea base, puntúa con un modelo
si se indica, y registra cada comprobación con sus alertas. No instala tareas programadas: genera la orden para que
la programe quien lo desee, y puede avisar a un webhook solo si se configura de forma explícita."""
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from kikolens.analyze.drift import compare_datasets
from kikolens.analyze.predict import input_warnings, load_bundle, predict_frame, resolve_model
from kikolens.analyze.quality import infer_types, profile_quality
from kikolens.config import home_dir
from kikolens.pipeline.recipe import Recipe
from kikolens.utils.loaders import load_data

DEFAULT_THRESHOLDS = {"drift_level": "high", "quality_drop": 10.0, "rows_change": 0.5, "prediction_shift": 0.2,
                      "confidence_drop": 0.1}
LEVELS = ["low", "medium", "high"]
SEVERITIES = {"high": 0, "medium": 1, "low": 2}
EVERY = {"hourly": ("0 * * * *", "HOURLY"), "daily": ("0 6 * * *", "DAILY"), "weekly": ("0 6 * * 1", "WEEKLY")}


def slugify(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "-", str(name)).strip("-").lower()[:60] or "monitor"


def monitors_dir(directory=None) -> Path:
    return Path(directory) if directory else home_dir() / "monitors"


@dataclass
class Monitor:
    name: str
    source: str
    recipe: str | None = None          # JSON de la receta (se copia dentro del monitor)
    model: str | None = None           # nombre o ruta de un modelo guardado
    thresholds: dict = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    webhook: str | None = None
    created: str = ""
    baseline_quality: float | None = None

    @property
    def slug(self) -> str:
        return slugify(self.name)

    def folder(self, directory=None) -> Path:
        return monitors_dir(directory) / self.slug


# ── Definición ───────────────────────────────────────────────────────────────
def _prepare(df: pd.DataFrame, recipe_json: str | None) -> pd.DataFrame:
    if recipe_json:
        df, _ = Recipe.from_json(recipe_json).apply(df)
    return df


def create_monitor(name: str, source: str, baseline: pd.DataFrame | None = None, recipe: Recipe | None = None,
                   model: str | None = None, thresholds: dict | None = None, webhook: str | None = None,
                   directory=None) -> Monitor:
    """Crea un monitor. La línea base (por defecto, el propio `source`) se guarda como Parquet dentro del monitor."""
    if webhook and not re.match(r"^https?://", webhook):
        raise ValueError("The webhook must be an http(s) address.")
    merged = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    unknown = set(merged) - set(DEFAULT_THRESHOLDS)
    if unknown or merged["drift_level"] not in LEVELS:
        raise ValueError(f"Thresholds: unknown {sorted(unknown)} or drift_level not in {LEVELS}.")
    if model:
        resolve_model(model)  # falla pronto si no existe
    if baseline is None:
        baseline = load_data(source)
    mon = Monitor(name=name, source=str(source), recipe=recipe.to_json() if recipe is not None and len(recipe) else None,
                  model=model, thresholds=merged, webhook=webhook,
                  created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    prepared, _ = infer_types(_prepare(baseline, mon.recipe))
    mon.baseline_quality = float(profile_quality(prepared)["score"])
    folder = mon.folder(directory)
    if folder.exists():
        raise ValueError(f"A monitor called '{name}' already exists.")
    (folder / "runs").mkdir(parents=True)
    prepared.to_parquet(folder / "baseline.parquet", index=False)
    (folder / "monitor.json").write_text(json.dumps(asdict(mon), indent=2, default=str), encoding="utf-8")
    return mon


def load_monitor(name: str, directory=None) -> Monitor:
    path = monitors_dir(directory) / slugify(name) / "monitor.json"
    if not path.is_file():
        raise ValueError(f"No monitor called '{name}'. See them with: kikolens monitor list")
    return Monitor(**json.loads(path.read_text(encoding="utf-8")))


def list_monitors(directory=None) -> pd.DataFrame:
    rows = []
    for path in sorted(monitors_dir(directory).glob("*/monitor.json")):
        try:
            mon = Monitor(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, TypeError, json.JSONDecodeError):
            continue
        runs = sorted((path.parent / "runs").glob("*.json"))
        last = json.loads(runs[-1].read_text(encoding="utf-8")) if runs else {}
        rows.append({"name": mon.name, "source": mon.source, "model": mon.model or "", "runs": len(runs),
                     "last run": last.get("time", ""), "last status": last.get("status", "never run")})
    return pd.DataFrame(rows, columns=["name", "source", "model", "runs", "last run", "last status"])


def delete_monitor(name: str, directory=None) -> None:
    mon = load_monitor(name, directory)
    shutil.rmtree(mon.folder(directory))


# ── Comprobación ─────────────────────────────────────────────────────────────
def _alert(alerts: list, severity: str, message: str) -> None:
    alerts.append({"severity": severity, "message": message})


def _prediction_summary(bundle: dict, out: pd.DataFrame) -> dict:
    col = f"predicted_{bundle['target']}"
    if bundle["task"] == "regression":
        return {"kind": "regression", "mean": float(out[col].mean()), "std": float(out[col].std() or 0.0)}
    shares = out[col].value_counts(normalize=True)
    return {"kind": "classification", "top_class": str(shares.index[0]), "top_share": float(shares.iloc[0]),
            "confidence": float(out["confidence"].mean()) if "confidence" in out else None}


def history(name: str, directory=None) -> pd.DataFrame:
    folder = load_monitor(name, directory).folder(directory)
    rows = []
    for path in sorted((folder / "runs").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        pred = rec.get("predictions") or {}
        rows.append({"time": rec["time"], "status": rec["status"], "rows": rec.get("rows"), "quality": rec.get("quality"),
                     "drift share": rec.get("drift_share"), "drift level": rec.get("drift_level"),
                     "alerts": len(rec.get("alerts", [])),
                     "prediction": pred.get("mean", pred.get("top_share"))})
    return pd.DataFrame(rows, columns=["time", "status", "rows", "quality", "drift share", "drift level", "alerts",
                                       "prediction"])


def _notify(url: str, record: dict) -> str | None:
    import requests

    summary = {"monitor": record["monitor"], "status": record["status"], "time": record["time"],
               "alerts": record["alerts"][:10]}
    try:
        requests.post(url, json=summary, timeout=10).raise_for_status()
        return None
    except requests.RequestException as e:
        return f"Could not reach the webhook: {e}"


def run_check(mon: Monitor, data: pd.DataFrame | None = None, directory=None, notify: bool = True) -> dict:
    """Una comprobación: calidad, comparación con la línea base, predicciones y alertas. Se guarda en el historial."""
    folder = mon.folder(directory)
    now = datetime.now(timezone.utc)
    record = {"monitor": mon.name, "time": now.isoformat(timespec="seconds"), "alerts": [], "status": "ok"}
    alerts, th = record["alerts"], mon.thresholds
    prev_runs = sorted((folder / "runs").glob("*.json"))
    prev = json.loads(prev_runs[-1].read_text(encoding="utf-8")) if prev_runs else {}

    try:
        raw = data if data is not None else load_data(mon.source)
        df, _ = infer_types(_prepare(raw, mon.recipe))
    except Exception as e:  # el archivo desapareció, cambió de formato o la receta ya no encaja
        _alert(alerts, "high", f"Could not read or prepare the data: {e}")
        record["status"] = "failed"
        return _finish(record, folder, mon, notify)

    record["rows"] = int(len(df))
    quality = float(profile_quality(df)["score"])
    record["quality"] = quality
    if mon.baseline_quality is not None and mon.baseline_quality - quality >= th["quality_drop"]:
        _alert(alerts, "high" if mon.baseline_quality - quality >= 2 * th["quality_drop"] else "medium",
               f"Data quality fell from {mon.baseline_quality:.0f} to {quality:.0f}.")
    if prev.get("rows") and abs(len(df) - prev["rows"]) / prev["rows"] >= th["rows_change"]:
        _alert(alerts, "medium", f"The number of rows changed from {prev['rows']:,} to {len(df):,}.")
    if len(df) == 0:
        _alert(alerts, "high", "The data has no rows.")

    baseline = pd.read_parquet(folder / "baseline.parquet")
    try:
        drift = compare_datasets(baseline, df)
    except ValueError as e:
        _alert(alerts, "high", f"Cannot compare with the baseline: {e}")
        drift = None
    if drift:
        record.update(drift_level=drift["level"], drift_share=round((drift["drifted"] + 0.5 * drift["moderate"]) /
                      max(drift["drifted"] + drift["moderate"] + drift["stable"], 1), 3),
                      drifted=drift["table"].loc[drift["table"]["verdict"] == "drift", "column"].tolist())
        if LEVELS.index(drift["level"]) >= LEVELS.index(th["drift_level"]):
            _alert(alerts, "high" if drift["level"] == "high" else "medium",
                   f"Drift level {drift['level']}: {drift['summary']}")
        if drift["only_in_reference"]:
            _alert(alerts, "high", f"Columns missing from the new data: {', '.join(map(str, drift['only_in_reference']))}.")
        if drift["only_in_current"]:
            _alert(alerts, "low", f"New columns: {', '.join(map(str, drift['only_in_current']))}.")
        if drift["type_changes"]:
            _alert(alerts, "high", f"Columns changed between numbers and text: {', '.join(map(str, drift['type_changes']))}.")

    if mon.model:
        try:
            bundle = load_bundle(resolve_model(mon.model))
            out, report = predict_frame(bundle, df)
            for severity, message in input_warnings(report):
                if severity != "low":
                    _alert(alerts, severity, f"Model input: {message}")
            summary = record["predictions"] = _prediction_summary(bundle, out)
            before = prev.get("predictions")
            if before and before.get("kind") == summary["kind"]:
                if summary["kind"] == "regression" and before["mean"]:
                    shift = abs(summary["mean"] - before["mean"]) / abs(before["mean"])
                    if shift >= th["prediction_shift"]:
                        _alert(alerts, "medium", f"The average prediction moved {shift:.0%} "
                                                 f"({before['mean']:.4g} to {summary['mean']:.4g}).")
                elif summary["kind"] == "classification" and None not in (summary["confidence"], before.get("confidence")) \
                        and before["confidence"] - summary["confidence"] >= th["confidence_drop"]:
                    _alert(alerts, "medium", f"The model's average confidence fell from {before['confidence']:.0%} "
                                             f"to {summary['confidence']:.0%}.")
        except ValueError as e:
            _alert(alerts, "high", f"Could not score with the model: {e}")

    if any(a["severity"] == "high" for a in alerts):
        record["status"] = "alert"
    elif alerts:
        record["status"] = "warning"
    return _finish(record, folder, mon, notify)


def _finish(record: dict, folder: Path, mon: Monitor, notify: bool) -> dict:
    record["alerts"].sort(key=lambda a: SEVERITIES[a["severity"]])
    if notify and mon.webhook and record["status"] in ("alert", "failed"):
        problem = _notify(mon.webhook, record)
        if problem:
            record["alerts"].append({"severity": "low", "message": problem})
    stamp = record["time"].replace(":", "").replace("-", "")
    (folder / "runs").mkdir(parents=True, exist_ok=True)
    (folder / "runs" / f"{stamp}.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


# ── Vigilancia y programación ────────────────────────────────────────────────
def watch(mon: Monitor, interval: float = 300.0, max_runs: int | None = None, directory=None, sleep=time.sleep,
          on_run=None) -> int:
    """Comprueba cuando el archivo cambia (y siempre en la primera vuelta). Devuelve cuántas comprobaciones hizo."""
    source = Path(mon.source)
    if not source.is_file():
        raise ValueError("watch needs a local file as the source.")
    if interval < 1:
        raise ValueError("The interval must be at least 1 second.")
    last, runs = None, 0
    while max_runs is None or runs < max_runs:
        try:
            stamp = source.stat().st_mtime_ns
        except OSError:
            stamp = None
        if stamp != last:
            last = stamp
            record = run_check(mon, directory=directory)
            runs += 1
            if on_run:
                on_run(record)
            if max_runs is not None and runs >= max_runs:
                break
        sleep(interval)
    return runs


def schedule_hint(mon: Monitor, every: str = "daily") -> str:
    """Las órdenes para programar la comprobación con cron o con el Programador de tareas de Windows."""
    if every not in EVERY:
        raise ValueError(f"Choose one of {list(EVERY)}.")
    cron, win = EVERY[every]
    command = f'"{sys.executable}" -m kikolens.cli.main monitor run "{mon.name}"'
    home = os.getenv("KIKOLENS_HOME")
    prefix = f"KIKOLENS_HOME={home} " if home else ""
    return (f"# Linux / macOS (crontab -e)\n{cron} {prefix}{command}\n\n"
            f"# Windows (PowerShell)\nschtasks /Create /SC {win} /TN \"KikoLens {mon.name}\" /TR '{command}'")
