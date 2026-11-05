"""Gobierno de modelos: quién es responsable de cada modelo guardado, en qué etapa está, qué se comprobó antes de ponerlo en producción y qué pasó con él.

  * Etapas: `draft` -> `staging` -> `production` -> `archived`. Solo hay un modelo en producción por objetivo; al promover otro, el anterior se archiva
    y se puede volver a él con `rollback`.
  * Puertas de promoción: documentación completa, archivo sin modificar desde que se registró, una validación reciente con datos nuevos que no empeore
    más de lo permitido, que no sea claramente peor que el modelo en producción y, si el riesgo es alto, la aprobación de otra persona.
  * Registro de auditoría: cada cambio queda anotado en una cadena de hashes, de modo que borrar o editar una línea se detecta (`verify_audit`).
  * Ficha del modelo (Markdown) con lo anterior.
Las personas (`actor`) son los nombres que se indican: es un registro para el equipo, no un control de acceso. Quien pueda escribir en la carpeta de datos
puede cambiar los archivos; la cadena de hashes sirve para notar el cambio, no para impedirlo."""
import getpass
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from kikolens.analyze import perfwatch
from kikolens.analyze.predict import load_bundle, predict_frame, resolve_model, slugify
from kikolens.config import home_dir

STAGES = ("draft", "staging", "production", "archived")
RISKS = ("low", "medium", "high")
TRANSITIONS = {"draft": {"staging", "archived"}, "staging": {"production", "draft", "archived"}, "production": {"staging", "archived"}, "archived": {"draft"}}
VALIDATION_DAYS = 90
MIN_VALIDATION_ROWS = 30
ERROR_METRICS = ("mae", "rmse")
SCORE_METRICS = ("r2", "accuracy", "f1")
_LOCK = threading.Lock()


def governance_dir(directory=None) -> Path:
    return Path(directory) if directory else home_dir() / "governance"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _actor(actor: str | None) -> str:
    name = (actor or "").strip()
    if name:
        return name[:80]
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── Auditoría ────────────────────────────────────────────────────────────────
def _audit_path(directory=None) -> Path:
    return governance_dir(directory) / "audit.jsonl"


def _entry_hash(entry: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in entry.items() if k != "hash"}, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _log(action: str, model: str, actor: str, details: dict | None = None, directory=None) -> dict:
    with _LOCK:
        path = _audit_path(directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = ""
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            if lines:
                try:
                    previous = json.loads(lines[-1]).get("hash", "")
                except json.JSONDecodeError:
                    previous = ""
        entry = {"ts": _now(), "actor": actor, "action": action, "model": model, "details": details or {}, "prev": previous}
        entry["hash"] = _entry_hash(entry)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry


def audit_log(model: str | None = None, directory=None) -> pd.DataFrame:
    path = _audit_path(directory)
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    frame = pd.DataFrame(rows, columns=["ts", "actor", "action", "model", "details", "prev", "hash"])
    if model is not None:
        frame = frame[frame["model"] == _stem(model, strict=False)]
    return frame.reset_index(drop=True)


def verify_audit(directory=None) -> dict:
    """Recorre la cadena y dice si alguna línea se ha editado, quitado o insertado."""
    path = _audit_path(directory)
    if not path.exists():
        return {"ok": True, "entries": 0, "broken_at": None, "message": "The audit log is empty."}
    previous = ""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return {"ok": False, "entries": len(lines), "broken_at": i + 1, "message": f"Line {i + 1} is not valid: the log was edited."}
        if entry.get("prev") != previous or entry.get("hash") != _entry_hash(entry):
            return {"ok": False, "entries": len(lines), "broken_at": i + 1,
                    "message": f"Line {i + 1} does not match the chain: an entry before it was changed, removed or inserted."}
        previous = entry["hash"]
    return {"ok": True, "entries": len(lines), "broken_at": None, "message": f"{len(lines)} entries, the chain is intact."}


# ── Registros ────────────────────────────────────────────────────────────────
def _stem(reference: str, models=None, strict: bool = True) -> str:
    try:
        return resolve_model(reference, models).stem
    except ValueError:
        if strict:
            raise
        return slugify(reference)


def _record_path(stem: str, directory=None) -> Path:
    return governance_dir(directory) / f"{stem}.json"


def _save(record: dict, directory=None) -> None:
    path = _record_path(record["model"], directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def get(reference: str, directory=None, models=None) -> dict:
    stem = _stem(reference, models, strict=False)
    path = _record_path(stem, directory)
    if not path.exists():
        raise ValueError(f"'{reference}' is not registered for governance yet. Register it with register().")
    return json.loads(path.read_text(encoding="utf-8"))


def register(reference: str, owner: str, description: str = "", intended_use: str = "", limitations: str = "", risk: str = "medium", actor: str | None = None,
             directory=None, models=None) -> dict:
    """Da de alta un modelo del registro con un responsable y su ficha. Empieza en `draft`."""
    if not owner or not owner.strip():
        raise ValueError("A model needs an owner: someone who answers for it.")
    if risk not in RISKS:
        raise ValueError(f"risk must be one of {', '.join(RISKS)}.")
    path = resolve_model(reference, models)
    stem = path.stem
    if _record_path(stem, directory).exists():
        raise ValueError(f"'{stem}' is already registered; use update_card() to change its details.")
    bundle_meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) if path.with_suffix(".json").exists() else {}
    record = {"model": stem, "file": str(path), "name": bundle_meta.get("name") or stem, "target": bundle_meta.get("target"), "task": bundle_meta.get("task"),
              "label": bundle_meta.get("label"), "stage": "draft", "owner": owner.strip()[:80], "description": description.strip(), "intended_use": intended_use.strip(),
              "limitations": limitations.strip(), "risk": risk, "registered": _now(), "validations": [], "approvals": [],
              "stage_history": [{"stage": "draft", "actor": _actor(actor), "ts": _now(), "reason": "registered"}],
              "lineage": {"sha256": _sha256(path), "created": bundle_meta.get("created"), "kikolens_version": bundle_meta.get("kikolens_version"),
                          "rows_trained": bundle_meta.get("rows_trained"), "has_recipe": bundle_meta.get("has_recipe"), "metrics": bundle_meta.get("metrics") or {},
                          "inputs": bundle_meta.get("inputs") or {}}}
    _save(record, directory)
    _log("register", stem, _actor(actor), {"owner": record["owner"], "risk": risk}, directory)
    return record


def update_card(reference: str, actor: str | None = None, directory=None, models=None, **fields) -> dict:
    allowed = {"owner", "description", "intended_use", "limitations", "risk"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"These fields cannot be changed: {', '.join(sorted(unknown))}. Allowed: {', '.join(sorted(allowed))}.")
    record = get(reference, directory, models)
    if "risk" in fields and fields["risk"] not in RISKS:
        raise ValueError(f"risk must be one of {', '.join(RISKS)}.")
    if "owner" in fields and not str(fields["owner"]).strip():
        raise ValueError("A model needs an owner.")
    changed = {k: [record.get(k), str(v).strip()] for k, v in fields.items() if str(v).strip() != str(record.get(k, "")).strip()}
    for k, (_, new) in changed.items():
        record[k] = new
    if changed:
        _save(record, directory)
        _log("update_card", record["model"], _actor(actor), {k: v[1] for k, v in changed.items()}, directory)
    return record


def integrity(reference: str, directory=None, models=None) -> dict:
    """¿Es el archivo el mismo que se registró? Si alguien lo cambió después, no."""
    record = get(reference, directory, models)
    path = Path(record["file"])
    if not path.exists():
        return {"ok": False, "message": "The model file is missing."}
    current = _sha256(path)
    same = current == record["lineage"]["sha256"]
    return {"ok": same, "message": "The model file is unchanged since it was registered." if same else
            "The model file changed after it was registered: it is not the model that was documented and validated. Register it again as a new version."}


# ── Validación con datos nuevos ──────────────────────────────────────────────
def _reference_metric(bundle: dict, new: dict) -> tuple:
    trained = {str(k).lower(): v for k, v in (bundle.get("metrics") or {}).items()}
    for key in ERROR_METRICS + SCORE_METRICS:
        for name, value in new.items():
            if name.lower() == key and key in trained and isinstance(trained[key], (int, float)):
                return name, float(trained[key]), float(value), key in ERROR_METRICS
    return None, None, None, None


def validate(reference: str, df: pd.DataFrame, actor: str | None = None, apply_recipe: bool = False, max_drop: float = 0.20, directory=None, models=None) -> dict:
    """Prueba el modelo con filas de las que ya se conoce el resultado y anota si aguanta: no puede empeorar más de `max_drop` (relativo) frente al entrenamiento."""
    if not 0 <= max_drop <= 1:
        raise ValueError("max_drop must be between 0 and 1.")
    record = get(reference, directory, models)
    bundle = load_bundle(record["file"])
    target = bundle["target"]
    if target not in df.columns:
        raise ValueError(f"The data needs the real values in a column named '{target}'.")
    data = df.dropna(subset=[target]).reset_index(drop=True)
    if len(data) < MIN_VALIDATION_ROWS:
        raise ValueError(f"At least {MIN_VALIDATION_ROWS} rows with a known result are needed (found {len(data)}).")
    scored, _ = predict_frame(bundle, data, apply_recipe=apply_recipe, interval=False)
    actual, predicted = data[target].reset_index(drop=True), scored[f"predicted_{target}"].reset_index(drop=True)
    metrics = perfwatch._metrics(bundle["task"], actual, predicted)
    name, trained, now, lower_is_better = _reference_metric(bundle, metrics)
    interval = None
    if name is None:
        degradation, passed, note = None, None, "The training metrics cannot be compared with these; the validation is recorded but does not count as passed."
    else:
        def loss(value):
            return (value - trained) / max(abs(trained), 1e-9) if lower_is_better else (trained - value) / max(abs(trained), 1e-9)

        # un intervalo por remuestreo: con pocas filas el acierto varía por azar, y solo se suspende si ni el extremo optimista cumple lo permitido
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(150):
            pick = rng.integers(0, len(actual), len(actual))
            draws.append(perfwatch._metrics(bundle["task"], actual.iloc[pick].reset_index(drop=True), predicted.iloc[pick].reset_index(drop=True))[name])
        low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        interval = (low, high)
        degradation = loss(now)
        optimistic = loss(low if lower_is_better else high)
        passed = bool(optimistic <= max_drop)
        note = (f"{name} is {now:.4g} on the new data (95% range {low:.4g} to {high:.4g}) against {trained:.4g} in training: {abs(degradation):.0%} "
                f"{'worse' if degradation > 0 else 'better'}" + ("." if passed else f", and even the best case in that range is more than {max_drop:.0%} worse."))
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(data[[c for c in data.columns][:30]], index=False).values.tobytes()).hexdigest()[:16]
    result = {"ts": _now(), "actor": _actor(actor), "rows": int(len(data)), "metrics": {k: float(v) for k, v in metrics.items()}, "reference_metric": name, "interval": interval,
              "trained_value": trained, "new_value": now, "degradation": degradation, "max_drop": max_drop, "passed": passed, "note": note, "data": fingerprint}
    record["validations"].append(result)
    _save(record, directory)
    _log("validate", record["model"], result["actor"], {"rows": result["rows"], "passed": passed, "degradation": degradation}, directory)
    return result


def _latest_valid(record: dict) -> dict | None:
    good = [v for v in record["validations"] if v.get("passed")]
    return good[-1] if good else None


def _age_days(ts: str) -> float:
    return (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() / 86400


# ── Aprobaciones y promoción ─────────────────────────────────────────────────
def approve(reference: str, actor: str, comment: str = "", directory=None, models=None) -> dict:
    """Anota la aprobación de otra persona. Para modelos de riesgo alto es imprescindible y no puede ser la del responsable."""
    who = _actor(actor)
    record = get(reference, directory, models)
    if who.lower() == record["owner"].lower():
        raise ValueError("The owner cannot approve their own model: ask someone else.")
    if any(a["actor"].lower() == who.lower() and a.get("fingerprint") == record["lineage"]["sha256"] for a in record["approvals"]):
        raise ValueError(f"{who} has already approved this version.")
    record["approvals"].append({"actor": who, "ts": _now(), "comment": comment.strip(), "fingerprint": record["lineage"]["sha256"]})
    _save(record, directory)
    _log("approve", record["model"], who, {"comment": comment.strip()}, directory)
    return record


def _production_of(target, exclude: str, directory=None) -> dict | None:
    folder = governance_dir(directory)
    if not folder.exists():
        return None
    for path in sorted(folder.glob("*.json")):
        try:
            other = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if other.get("model") != exclude and other.get("target") == target and other.get("stage") == "production":
            return other
    return None


def checks(reference: str, to: str, data: pd.DataFrame | None = None, apply_recipe: bool = False, directory=None, models=None) -> list:
    """Las puertas para pasar a `to`: lista de {name, passed, detail}. No cambia nada."""
    if to not in STAGES:
        raise ValueError(f"The stage must be one of {', '.join(STAGES)}.")
    record = get(reference, directory, models)
    out = []
    if to in ("staging", "production"):
        missing = [label for key, label in (("owner", "an owner"), ("description", "a description"), ("intended_use", "the intended use"), ("limitations", "the known limitations"))
                   if not record.get(key)]
        out.append({"name": "documented", "passed": not missing, "detail": "Owner, description, intended use and limitations are filled in." if not missing
                    else f"Missing: {', '.join(missing)}."})
        check = integrity(reference, directory, models)
        out.append({"name": "file unchanged", "passed": check["ok"], "detail": check["message"]})
    if to == "production":
        good = _latest_valid(record)
        recent = bool(good) and _age_days(good["ts"]) <= VALIDATION_DAYS
        out.append({"name": "validated on new data", "passed": recent, "detail": (f"Passed on {good['rows']:,} rows {_age_days(good['ts']):.0f} days ago ({good['note']})" if recent else
                                                                                  f"No passed validation in the last {VALIDATION_DAYS} days: run validate() with rows the model has not seen.")})
        current = _production_of(record["target"], record["model"], directory)
        if current is None:
            out.append({"name": "not worse than production", "passed": True, "detail": "There is no model in production for this target yet."})
        elif data is None:
            out.append({"name": "not worse than production", "passed": False, "detail": f"'{current['model']}' is in production for '{record['target']}': give data with the real "
                                                                                      "results so both can be compared."})
        else:
            from kikolens.analyze.champion import compare_saved_models
            mine, theirs = load_bundle(record["file"]), load_bundle(current["file"])
            result = compare_saved_models({record["model"]: mine, current["model"]: theirs}, data, apply_recipe)
            worse = result["champion"] == current["model"] and result["reliable"]
            out.append({"name": "not worse than production", "passed": not worse, "detail": result["summary"]})
        if record["risk"] == "high":
            valid = [a for a in record["approvals"] if a.get("fingerprint") == record["lineage"]["sha256"] and a["actor"].lower() != record["owner"].lower()]
            out.append({"name": "approved by someone else", "passed": bool(valid), "detail": f"Approved by {', '.join(a['actor'] for a in valid)}." if valid else
                        "A high-risk model needs the approval of a person other than its owner (approve())."})
    if to == "archived" or to == "draft":
        out.append({"name": "allowed", "passed": True, "detail": "No conditions."})
    return out


def promote(reference: str, to: str, actor: str | None = None, data: pd.DataFrame | None = None, reason: str = "", force: bool = False, apply_recipe: bool = False,
            directory=None, models=None) -> dict:
    """Cambia la etapa si se cumplen las puertas. Con `force` se salta las que fallen, pero hace falta un motivo y queda anotado como excepción."""
    who = _actor(actor)
    record = get(reference, directory, models)
    current = record["stage"]
    if to not in STAGES:
        raise ValueError(f"The stage must be one of {', '.join(STAGES)}.")
    if to not in TRANSITIONS[current]:
        raise ValueError(f"A model in '{current}' cannot go to '{to}'. From there it can go to: {', '.join(sorted(TRANSITIONS[current]))}.")
    gates = checks(reference, to, data, apply_recipe, directory, models)
    failed = [g for g in gates if not g["passed"]]
    if failed and not force:
        raise ValueError("Not ready for " + to + ": " + " ".join(f"[{g['name']}] {g['detail']}" for g in failed))
    if failed and not reason.strip():
        raise ValueError("Overriding failed checks needs a reason, which is recorded in the audit log.")
    replaced = None
    if to == "production":
        previous = _production_of(record["target"], record["model"], directory)
        if previous:
            previous["stage"] = "archived"
            previous["stage_history"].append({"stage": "archived", "actor": who, "ts": _now(), "reason": f"replaced by {record['model']}"})
            _save(previous, directory)
            _log("archive", previous["model"], who, {"replaced_by": record["model"]}, directory)
            replaced = previous["model"]
    record["stage"] = to
    record["stage_history"].append({"stage": to, "actor": who, "ts": _now(), "reason": reason.strip() or "all checks passed"})
    _save(record, directory)
    _log("promote" if not failed else "promote_override", record["model"], who, {"from": current, "to": to, "reason": reason.strip(), "replaced": replaced,
                                                                                  "failed_checks": [g["name"] for g in failed]}, directory)
    return {"model": record["model"], "from": current, "to": to, "replaced": replaced, "overridden": [g["name"] for g in failed], "checks": gates}


def rollback(target: str, actor: str | None = None, reason: str = "", directory=None) -> dict:
    """Vuelve al modelo que estaba en producción antes del actual (el actual pasa a `staging`). Comprueba que su archivo no haya cambiado."""
    who = _actor(actor)
    folder = governance_dir(directory)
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))] if folder.exists() else []
    records = [r for r in records if r.get("target") == target]
    current = next((r for r in records if r["stage"] == "production"), None)
    candidates = []
    for r in records:
        if r["stage"] == "archived":
            for event in reversed(r["stage_history"]):
                if event["stage"] == "archived" and event["reason"].startswith("replaced by"):
                    candidates.append((event["ts"], r))
                    break
    if not candidates:
        raise ValueError(f"There is no earlier production model of '{target}' to go back to.")
    candidates.sort(key=lambda c: c[0])
    previous = candidates[-1][1]
    check = integrity(previous["model"], directory)
    if not check["ok"]:
        raise ValueError(f"Cannot go back to '{previous['model']}': {check['message']}")
    if current is not None:
        current["stage"] = "staging"
        current["stage_history"].append({"stage": "staging", "actor": who, "ts": _now(), "reason": f"rolled back to {previous['model']}: {reason.strip() or 'no reason given'}"})
        _save(current, directory)
    previous["stage"] = "production"
    previous["stage_history"].append({"stage": "production", "actor": who, "ts": _now(), "reason": f"rollback: {reason.strip() or 'no reason given'}"})
    _save(previous, directory)
    _log("rollback", previous["model"], who, {"target": target, "from": current["model"] if current else None, "reason": reason.strip()}, directory)
    return {"restored": previous["model"], "demoted": current["model"] if current else None}


# ── Vistas ───────────────────────────────────────────────────────────────────
def overview(directory=None, models=None) -> pd.DataFrame:
    """Todos los modelos dados de alta: etapa, responsable, riesgo, última validación y si toca revisarlo."""
    folder = governance_dir(directory)
    rows = []
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        good = _latest_valid(r)
        last = r["validations"][-1] if r["validations"] else None
        file_ok = Path(r["file"]).exists() and _sha256(Path(r["file"])) == r["lineage"]["sha256"]
        due = r["stage"] in ("staging", "production") and (good is None or _age_days(good["ts"]) > VALIDATION_DAYS)
        rows.append({"model": r["model"], "target": r["target"], "stage": r["stage"], "risk": r["risk"], "owner": r["owner"],
                     "last validation": last["ts"][:10] if last else "", "validation passed": (last or {}).get("passed"), "review due": due, "file unchanged": file_ok})
    columns = ["model", "target", "stage", "risk", "owner", "last validation", "validation passed", "review due", "file unchanged"]
    frame = pd.DataFrame(rows, columns=columns)
    order = {s: i for i, s in enumerate(("production", "staging", "draft", "archived"))}
    return frame.sort_values("stage", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)


def model_card(reference: str, directory=None, models=None) -> str:
    """Ficha en Markdown: para qué sirve, quién responde, cómo rinde, qué se comprobó y la historia de etapas."""
    r = get(reference, directory, models)
    check = integrity(reference, directory, models)
    lines = [f"# Model card: {r['name']}", "", "| | |", "| :--- | :--- |", f"| Stage | **{r['stage']}** |", f"| Owner | {r['owner']} |", f"| Risk | {r['risk']} |",
             f"| Predicts | {r['target']} ({r['task']}) |", f"| Algorithm | {r.get('label') or ''} |", f"| Trained on | {r['lineage'].get('rows_trained') or '?'} rows |",
             f"| Created | {r['lineage'].get('created') or '?'} with KikoLens {r['lineage'].get('kikolens_version') or '?'} |",
             f"| Cleaning steps included | {'yes' if r['lineage'].get('has_recipe') else 'no'} |", f"| File fingerprint | `{r['lineage']['sha256'][:16]}` ({'unchanged' if check['ok'] else 'CHANGED'}) |", ""]
    for title, key in (("Purpose", "description"), ("Intended use", "intended_use"), ("Limitations", "limitations")):
        lines += [f"## {title}", r.get(key) or "_Not written yet._", ""]
    metrics = r["lineage"].get("metrics") or {}
    lines += ["## Performance in training", ", ".join(f"{k} {v:.4g}" for k, v in metrics.items()) or "_Not recorded._", "", "## Validation on new data"]
    if r["validations"]:
        lines += ["| Date | By | Rows | Result | Detail |", "| :--- | :--- | ---: | :--- | :--- |"]
        for v in r["validations"]:
            lines.append(f"| {v['ts'][:10]} | {v['actor']} | {v['rows']:,} | {'passed' if v['passed'] else 'not passed' if v['passed'] is False else 'recorded'} | {v['note']} |")
    else:
        lines.append("_No validation yet._")
    lines += ["", "## Inputs", ", ".join(f"{k} ({v})" for k, v in (r["lineage"].get("inputs") or {}).items()) or "_Not recorded._", "", "## Approvals"]
    lines += [f"- {a['actor']} on {a['ts'][:10]}" + (f": {a['comment']}" if a["comment"] else "") for a in r["approvals"]] or ["_None._"]
    lines += ["", "## History"] + [f"- {e['ts'][:10]} -> **{e['stage']}** by {e['actor']} ({e['reason']})" for e in r["stage_history"]]
    return "\n".join(lines) + "\n"


def resolve_reference(reference: str, directory=None) -> Path:
    """Admite `objetivo@production` (o `modelo@staging`): el archivo del modelo que está en esa etapa."""
    base, _, stage = reference.rpartition("@")
    if stage not in STAGES or not base:
        raise ValueError(f"'{reference}': write it as NAME@STAGE with a stage of {', '.join(STAGES)}.")
    folder = governance_dir(directory)
    matches = []
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if r.get("stage") == stage and (base in (r.get("target"), r.get("model"), r.get("name")) or r.get("model") == slugify(base)):
            matches.append(r)
    if not matches:
        raise ValueError(f"No model called or predicting '{base}' is in '{stage}'. See the stages with: kikolens governance")
    chosen = max(matches, key=lambda r: r["stage_history"][-1]["ts"])
    check = integrity(chosen["model"], directory)
    if not check["ok"]:
        raise ValueError(f"'{chosen['model']}' is in {stage} but {check['message']}")
    return Path(chosen["file"])
