"""Servidor local para puntuar con un modelo guardado: una API JSON mínima con la biblioteca estándar.
  GET  /health   -> {"status": "ok"}
  GET  /model    -> metadatos, entradas esperadas y métricas
  POST /predict  -> {"rows": [{...}, ...], "raw": false}  ->  {"predictions": [...], "warnings": [...]}
Solo escucha en la propia máquina salvo que se indique otra dirección, y entonces exige una clave (token)."""
import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

from kikolens.analyze.predict import input_warnings, metadata, predict_frame

MAX_BODY = 5_000_000
MAX_ROWS = 10_000
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _records(frame: pd.DataFrame) -> list:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def make_server(bundle: dict, host: str = "127.0.0.1", port: int = 8000, token: str | None = None) -> ThreadingHTTPServer:
    """Crea el servidor sin arrancarlo (así se puede probar con `port=0`)."""
    if host not in LOCAL_HOSTS and not token:
        raise ValueError("Listening on a network address needs a token (--token), so that not everyone on the network can use the model.")
    info = metadata(bundle)

    class Handler(BaseHTTPRequestHandler):
        server_version = "KikoLens"

        def log_message(self, *args):  # silencioso: la consola no es un registro de datos de clientes
            pass

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not token:
                return True
            given = self.headers.get("Authorization", "")
            return hmac.compare_digest(given.encode(), f"Bearer {token}".encode())

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"status": "ok"})
            if not self._authorized():
                return self._send(401, {"error": "Missing or wrong token."})
            if self.path == "/model":
                return self._send(200, info)
            self._send(404, {"error": "Not found. Use /health, /model or POST /predict."})

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, {"error": "Not found. Use POST /predict."})
            if not self._authorized():
                return self._send(401, {"error": "Missing or wrong token."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._send(400, {"error": "Invalid Content-Length."})
            if length <= 0 or length > MAX_BODY:
                return self._send(413 if length > MAX_BODY else 400, {"error": f"Send between 1 byte and {MAX_BODY:,} bytes of JSON."})
            try:
                payload = json.loads(self.rfile.read(length))
                rows = payload["rows"]
                if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
                    raise ValueError("'rows' must be a non-empty list of objects, one per row.")
                if len(rows) > MAX_ROWS:
                    raise ValueError(f"At most {MAX_ROWS:,} rows per request.")
                frame, report = predict_frame(bundle, pd.DataFrame(rows), apply_recipe=bool(payload.get("raw", False)))
            except (json.JSONDecodeError, KeyError, TypeError):
                return self._send(400, {"error": 'Send JSON like {"rows": [{"column": value, ...}]}.'})
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            given = set().union(*rows)
            new = [c for c in frame.columns if c not in given]
            self._send(200, {"target": bundle["target"], "predictions": _records(frame[new]),
                             "warnings": [message for _, message in input_warnings(report)]})

    return ThreadingHTTPServer((host, port), Handler)


def serve(bundle: dict, host: str = "127.0.0.1", port: int = 8000, token: str | None = None) -> None:
    server = make_server(bundle, host, port, token)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
