"""Servidor de API de KikoLens: el análisis completo por HTTP y JSON, con la biblioteca estándar y sin enviar datos a ningún sitio.

  GET    /health                      -> {"status": "ok"}                                  (sin clave)
  GET    /functions                   -> análisis disponibles y sus parámetros
  POST   /datasets                    -> {"rows": [...]} | {"csv": "texto"} | {"file": "nombre.csv"}  ->  {"id": ...}
  GET    /datasets, /datasets/{id}    -> lista, o forma y columnas de uno
  DELETE /datasets/{id}
  POST   /run/{función}               -> {"dataset": id, "args": {...}}  ->  resultado (resumen, tablas, números)
  POST   /jobs, GET /jobs/{id}        -> lo mismo, en segundo plano, para lo que tarda
  GET    /models, POST /models/{nombre}/predict   -> los modelos que se indicaron al arrancar
  GET    /openapi.json

Solo se ejecutan las funciones de una lista cerrada: nunca código, rutas ni consultas arbitrarias. Los archivos solo se leen
de la carpeta indicada con `data_dir`. Escucha solo en esta máquina salvo que se pida otra dirección, y entonces exige una clave."""
import hmac
import inspect
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from kikolens.analyze.predict import input_warnings, metadata, predict_frame

MAX_BODY = 25_000_000
MAX_ROWS = 1_000_000
MAX_DATASETS = 20
MAX_JOBS = 200
MAX_RESULT_ROWS = 200
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
ALLOWED = ("profile", "dictionary", "scan_privacy", "autopilot", "lab", "forecast", "anomalies", "spikes", "why", "regress", "survival", "cohorts", "rfm",
           "select_features", "segment", "ab_sequential", "granger", "forecast_drivers", "synthesize", "effect", "uplift", "discontinuity", "synthetic_control",
           "instrumental", "query")
ID = re.compile(r"[a-z0-9]{12}")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


# ── Resultados a JSON ────────────────────────────────────────────────────────
def _scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if value != value or value in (float("inf"), float("-inf")) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta, pd.Period)):
        return str(value)
    return value


def _frame(df: pd.DataFrame, limit: int) -> dict:
    shown = df.head(limit)
    if isinstance(shown.index, pd.MultiIndex) or not isinstance(shown.index, pd.RangeIndex):
        shown = shown.reset_index()
    shown = shown.copy()
    shown.columns = [str(c) for c in shown.columns]
    return {"columns": list(shown.columns), "rows": json.loads(shown.to_json(orient="records", date_format="iso", default_handler=str)), "total_rows": int(len(df))}


def to_json(value, figures: bool = False, limit: int = MAX_RESULT_ROWS, depth: int = 0):
    """Convierte un resultado de KikoLens en algo que cabe en JSON. Devuelve (valor, claves omitidas); lo que no se puede describir se omite y se dice."""
    omitted = []

    def convert(v, level):
        if isinstance(v, pd.DataFrame):
            return _frame(v, limit)
        if isinstance(v, pd.Series):
            return _frame(v.rename(v.name or "value").to_frame(), limit)
        if hasattr(v, "to_plotly_json"):
            return json.loads(v.to_json()) if figures else _SKIP
        if isinstance(v, (str, bool)) or v is None:
            return v
        if isinstance(v, (int, float, np.integer, np.floating, np.bool_, pd.Timestamp, pd.Timedelta)):
            return _scalar(v)
        if isinstance(v, np.ndarray):
            return [_scalar(x) for x in v.ravel()[:limit]] if v.ndim == 1 or v.size <= limit else _SKIP
        if isinstance(v, (list, tuple, set)) and level < 3:
            items = [convert(x, level + 1) for x in list(v)[:limit]]
            return [x for x in items if x is not _SKIP]
        if isinstance(v, dict) and level < 3:
            out = {}
            for k, x in v.items():
                c = convert(x, level + 1)
                if c is not _SKIP:
                    out[str(k)] = c
            return out
        return _SKIP

    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if str(key).startswith("_"):
                continue
            converted = convert(item, depth)
            if converted is _SKIP:
                omitted.append(str(key))
            else:
                out[str(key)] = converted
        return out, omitted
    converted = convert(value, depth)
    return ({"value": converted} if converted is not _SKIP else {}), ([] if converted is not _SKIP else ["value"])


_SKIP = object()


# ── Estado ───────────────────────────────────────────────────────────────────
class Store:
    """Conjuntos de datos y trabajos, en memoria y con límites: al cerrar el servidor no queda nada."""

    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir).resolve() if data_dir else None
        self.datasets, self.jobs = {}, {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=2)

    def add_dataset(self, df: pd.DataFrame, name: str) -> str:
        if df.empty or not len(df.columns):
            raise ApiError(400, "The dataset has no rows or no columns.")
        if len(df) > MAX_ROWS:
            raise ApiError(413, f"At most {MAX_ROWS:,} rows per dataset.")
        with self.lock:
            if len(self.datasets) >= MAX_DATASETS:
                raise ApiError(409, f"At most {MAX_DATASETS} datasets are kept at once; delete one first.")
            ident = uuid.uuid4().hex[:12]
            self.datasets[ident] = {"df": df, "name": name, "created": time.time()}
        return ident

    def dataset(self, ident: str) -> pd.DataFrame:
        if not isinstance(ident, str) or ident not in self.datasets:
            raise ApiError(404, f"No dataset '{ident}'. Create one with POST /datasets.")
        return self.datasets[ident]["df"]

    def read_file(self, name: str) -> pd.DataFrame:
        if self.data_dir is None:
            raise ApiError(400, "This server was not started with a data folder, so it cannot read files; send the rows or the csv text.")
        available = {entry.name: entry for entry in self.data_dir.iterdir() if entry.is_file()}      # lista permitida: el cliente elige entre lo que hay, no escribe una ruta
        target = available.get(name)
        if target is None:
            raise ApiError(404, "No such file in the data folder.")
        from kikolens.utils.loaders import load_data
        try:
            return load_data(str(target))
        except (ValueError, OSError) as e:
            raise ApiError(400, str(e)) from None


def functions() -> dict:
    """Las funciones que se pueden ejecutar: nombre -> (función, descripción, parámetros)."""
    from kikolens import api
    out = {}
    for name in ALLOWED:
        fn = getattr(api, name, None)
        if fn is None:
            continue
        params = {k: ("required" if p.default is inspect.Parameter.empty else repr(p.default)) for k, p in list(inspect.signature(fn).parameters.items())[1:]
                  if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)}
        out[name] = {"fn": fn, "description": (fn.__doc__ or "").strip().split("\n")[0], "params": params, "source": "built-in"}
    return out


def run_function(store: Store, name: str, dataset: str, args: dict, figures: bool = False) -> dict:
    catalog = functions()
    if name not in catalog:
        raise ApiError(404, f"No function '{name}'. See GET /functions.")
    if not isinstance(args, dict):
        raise ApiError(400, "'args' must be an object.")
    df = store.dataset(dataset)
    started = time.time()
    try:
        result = catalog[name]["fn"](df, **args)
    except ApiError:
        raise
    except TypeError as e:
        raise ApiError(400, f"Invalid arguments for {name}: {e}") from None
    except ValueError as e:
        raise ApiError(400, str(e)) from None
    except KeyError as e:                                            # una columna que no existe, en funciones que no la comprueban antes
        raise ApiError(400, f"Unknown column or key: {e}.") from None
    except Exception as e:                                           # no se filtran rutas ni trazas al cliente
        raise ApiError(500, f"{name} failed: {type(e).__name__}.") from None
    body, omitted = to_json(result, figures)
    return {"function": name, "dataset": dataset, "seconds": round(time.time() - started, 3), "result": body, "omitted": omitted}


def start_job(store: Store, name: str, dataset: str, args: dict) -> dict:
    if name not in functions():
        raise ApiError(404, f"No function '{name}'. See GET /functions.")
    store.dataset(dataset)
    with store.lock:
        if len(store.jobs) >= MAX_JOBS:
            for old in sorted(store.jobs, key=lambda j: store.jobs[j]["created"])[: MAX_JOBS // 2]:
                if store.jobs[old]["status"] in ("done", "failed"):
                    store.jobs.pop(old)
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "status": "queued", "function": name, "dataset": dataset, "created": time.time()}
        store.jobs[job_id] = job

    def work():
        job["status"] = "running"
        try:
            job["result"] = run_function(store, name, dataset, args)
            job["status"] = "done"
        except ApiError as e:
            job["error"], job["status"] = e.message, "failed"
        except Exception as e:                                       # el hilo no debe morir en silencio
            job["error"], job["status"] = f"{type(e).__name__}", "failed"

    store.pool.submit(work)
    return {"id": job_id, "status": "queued"}


# ── OpenAPI ──────────────────────────────────────────────────────────────────
def openapi(models: dict) -> dict:
    body = lambda props, required=(): {"required": True, "content": {"application/json": {"schema": {"type": "object", "properties": props, "required": list(required)}}}}   # noqa: E731
    ok = {"200": {"description": "OK"}, "400": {"description": "Invalid request"}, "401": {"description": "Missing or wrong token"}, "404": {"description": "Not found"}}
    return {"openapi": "3.0.3", "info": {"title": "KikoLens API", "version": "1", "description": "Local data analysis over HTTP. Nothing leaves the machine it runs on."},
            "paths": {
                "/health": {"get": {"summary": "Is the server up?", "responses": {"200": {"description": "ok"}}}},
                "/functions": {"get": {"summary": "Available analyses and their parameters", "responses": ok}},
                "/datasets": {"get": {"summary": "Datasets held in memory", "responses": ok},
                              "post": {"summary": "Create a dataset from rows, csv text or a file in the data folder", "responses": ok,
                                       "requestBody": body({"rows": {"type": "array", "items": {"type": "object"}}, "csv": {"type": "string"}, "file": {"type": "string"},
                                                            "name": {"type": "string"}})}},
                "/datasets/{id}": {"get": {"summary": "Shape, columns and first rows", "responses": ok}, "delete": {"summary": "Forget a dataset", "responses": ok}},
                "/run/{function}": {"post": {"summary": "Run an analysis and wait for it", "responses": ok,
                                             "requestBody": body({"dataset": {"type": "string"}, "args": {"type": "object"}, "figures": {"type": "boolean"}}, ["dataset"])}},
                "/jobs": {"post": {"summary": "Run an analysis in the background", "responses": ok,
                                   "requestBody": body({"function": {"type": "string"}, "dataset": {"type": "string"}, "args": {"type": "object"}}, ["function", "dataset"])}},
                "/jobs/{id}": {"get": {"summary": "Status and result of a job", "responses": ok}},
                "/models": {"get": {"summary": "Models being served", "responses": ok}},
                "/models/{name}/predict": {"post": {"summary": "Predict with a model", "responses": ok, "requestBody": body({"rows": {"type": "array", "items": {"type": "object"}},
                                                                                                                                  "raw": {"type": "boolean"}}, ["rows"])}},
            }, "components": {"securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}}}, "x-models": sorted(models)}


# ── Servidor ─────────────────────────────────────────────────────────────────
def make_server(host: str = "127.0.0.1", port: int = 8000, token: str | None = None, data_dir=None, models: dict | None = None,
                cors: str | None = None) -> ThreadingHTTPServer:
    """Crea el servidor sin arrancarlo (así se puede probar con `port=0`). `models` es {nombre: paquete} de modelos ya abiertos."""
    if host not in LOCAL_HOSTS and not token:
        raise ValueError("Listening on a network address needs a token (--token), so that not everyone on the network can use the server.")
    if data_dir is not None and not Path(data_dir).is_dir():
        raise ValueError(f"The data folder '{data_dir}' does not exist.")
    models = models or {}
    store = Store(data_dir)

    class Handler(BaseHTTPRequestHandler):
        server_version = "KikoLens"

        def log_message(self, *args):
            pass

        def _send(self, status: int, payload) -> None:
            if status == 204:                                        # respuesta a OPTIONS: sin cuerpo
                self.send_response(204)
                if cors:
                    self.send_header("Access-Control-Allow-Origin", cors)
                    self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if cors:
                self.send_header("Access-Control-Allow-Origin", cors)
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            return not token or hmac.compare_digest(self.headers.get("Authorization", "").encode(), f"Bearer {token}".encode())

        def _body(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.close_connection = True
                raise ApiError(400, "Invalid Content-Length.") from None
            if length > MAX_BODY:
                self.close_connection = True
                raise ApiError(413, f"Send at most {MAX_BODY:,} bytes of JSON.")
            raw = self.rfile.read(length) if length > 0 else b""
            if not raw:
                return {}
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                raise ApiError(400, "The body is not valid JSON.") from None
            if not isinstance(payload, dict):
                raise ApiError(400, "The body must be a JSON object.")
            return payload

        def _handle(self, method: str) -> None:
            try:
                path = self.path.split("?")[0].rstrip("/") or "/"
                query = self.path.partition("?")[2]
                if method == "GET" and path == "/health":
                    return self._send(200, {"status": "ok"})
                if method == "OPTIONS":
                    return self._send(204, {})
                body = self._body() if method in ("POST", "DELETE") else {}        # siempre se lee el cuerpo antes de responder un error
                if not self._authorized():
                    raise ApiError(401, "Missing or wrong token.")
                self._send(200, self._route(method, path, body, query))
            except ApiError as e:
                self._send(e.status, {"error": e.message})
            except Exception as e:
                self._send(500, {"error": f"Unexpected error: {type(e).__name__}."})

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

        def do_DELETE(self):
            self._handle("DELETE")

        def do_OPTIONS(self):
            self._handle("OPTIONS")

        def _route(self, method: str, path: str, body: dict, query: str):
            parts = path.strip("/").split("/")
            if method == "GET" and path == "/openapi.json":
                return openapi(models)
            if method == "GET" and path == "/functions":
                return {"functions": [{"name": n, "description": f["description"], "parameters": f["params"], "source": f["source"]} for n, f in functions().items()]}
            if parts[0] == "datasets":
                return self._datasets(method, parts, body)
            if method == "POST" and len(parts) == 2 and parts[0] == "run":
                return run_function(store, parts[1], body.get("dataset"), body.get("args") or {}, bool(body.get("figures", False)) or "figures=1" in query)
            if method == "POST" and path == "/jobs":
                return start_job(store, str(body.get("function")), body.get("dataset"), body.get("args") or {})
            if method == "GET" and len(parts) == 2 and parts[0] == "jobs":
                job = store.jobs.get(parts[1])
                if job is None:
                    raise ApiError(404, "No such job.")
                return {k: v for k, v in job.items() if k != "created"}
            if method == "GET" and path == "/models":
                return {"models": [{"name": n, **metadata(b)} for n, b in models.items()]}
            if method == "POST" and len(parts) == 3 and parts[0] == "models" and parts[2] == "predict":
                return self._predict(parts[1], body)
            raise ApiError(404, "Not found. See GET /openapi.json.")

        def _datasets(self, method: str, parts: list, body: dict):
            if len(parts) == 1 and method == "GET":
                return {"datasets": [{"id": i, "name": d["name"], "rows": len(d["df"]), "columns": len(d["df"].columns)} for i, d in store.datasets.items()]}
            if len(parts) == 1 and method == "POST":
                if "rows" in body:
                    rows = body["rows"]
                    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
                        raise ApiError(400, "'rows' must be a non-empty list of objects, one per row.")
                    df = pd.DataFrame(rows)
                elif "csv" in body:
                    try:
                        df = pd.read_csv(StringIO(str(body["csv"])))
                    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as e:
                        raise ApiError(400, f"The csv could not be read: {e}") from None
                elif "file" in body:
                    df = store.read_file(str(body["file"]))
                else:
                    raise ApiError(400, "Send 'rows', 'csv' or 'file'.")
                name = str(body.get("name") or body.get("file") or "dataset")[:80]
                ident = store.add_dataset(df, name)
                return {"id": ident, "rows": len(df), "columns": list(map(str, df.columns))}
            if len(parts) == 2 and ID.fullmatch(parts[1]):
                if method == "GET":
                    df = store.dataset(parts[1])
                    return {"id": parts[1], "name": store.datasets[parts[1]]["name"], "rows": len(df),
                            "columns": [{"name": str(c), "type": str(df[c].dtype), "missing": int(df[c].isna().sum())} for c in df.columns], "head": _frame(df, 5)["rows"]}
                if method == "DELETE":
                    store.dataset(parts[1])
                    with store.lock:
                        store.datasets.pop(parts[1], None)
                    return {"deleted": parts[1]}
            raise ApiError(404, "Not found. See GET /openapi.json.")

        def _predict(self, name: str, body: dict):
            if name not in models:
                raise ApiError(404, f"No model '{name}' is being served. See GET /models.")
            rows = body.get("rows")
            if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
                raise ApiError(400, "'rows' must be a non-empty list of objects, one per row.")
            if len(rows) > 10_000:
                raise ApiError(400, "At most 10,000 rows per request.")
            try:
                frame, report = predict_frame(models[name], pd.DataFrame(rows), apply_recipe=bool(body.get("raw", False)))
            except ValueError as e:
                raise ApiError(400, str(e)) from None
            new = [c for c in frame.columns if c not in set().union(*rows)]
            return {"model": name, "target": models[name]["target"], "predictions": json.loads(frame[new].to_json(orient="records", date_format="iso")),
                    "warnings": [message for _, message in input_warnings(report)]}

    server = ThreadingHTTPServer((host, port), Handler)
    server.store = store
    return server


def serve(host: str = "127.0.0.1", port: int = 8000, token: str | None = None, data_dir=None, models: dict | None = None, cors: str | None = None) -> None:
    server = make_server(host, port, token, data_dir, models, cors)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.store.pool.shutdown(wait=False, cancel_futures=True)
