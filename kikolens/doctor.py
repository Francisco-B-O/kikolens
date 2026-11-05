"""Diagnóstico del entorno: `kikolens doctor` comprueba que todo lo necesario está instalado y en una versión compatible, qué extras
opcionales hay, si se puede escribir en la carpeta de usuario, si Ollama responde y si la configuración se lee bien.
Cada comprobación es un diccionario (`name`, `status`, `detail`) con estado `ok`, `warning` (opcional ausente) o `error` (hace falta arreglarlo)."""
import importlib
import importlib.metadata
import json
import platform
import sys
import tempfile
import urllib.error
import urllib.request

from kikolens import config

REQUIRED = {"pandas": "2.2", "numpy": "1.26", "scikit-learn": "1.4", "streamlit": "1.50", "plotly": "5.19", "duckdb": "0.10",
            "scipy": "1.11", "typer": "0.9", "openpyxl": "3.1", "matplotlib": "3.8"}
OPTIONAL = {"torch": "the PyTorch network (kikolens[torch])", "jupyterlab": "JupyterLab for `kikolens jupyter` (kikolens[jupyter])",
            "s3fs": "s3:// sources (kikolens[cloud])", "gcsfs": "gs:// sources (kikolens[cloud])",
            "psycopg2": "PostgreSQL (kikolens[postgres])", "pymysql": "MySQL (kikolens[mysql])",
            "playwright": "regenerating the documentation screenshots (kikolens[docs])"}


def _version_tuple(text: str) -> tuple:
    parts = []
    for piece in str(text).split("."):
        digits = ""
        for char in piece:                     # solo los dígitos del principio: "0rc1" es 0 y "post1" no es un número de versión
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _installed(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_python() -> dict:
    ok = sys.version_info >= (3, 12)
    return {"name": "Python", "status": "ok" if ok else "error",
            "detail": f"{platform.python_version()} on {platform.system()}" + ("" if ok else " (3.12 or newer is required)")}


def check_required() -> list:
    out = []
    for package, minimum in REQUIRED.items():
        version = _installed(package)
        if version is None:
            out.append({"name": package, "status": "error", "detail": "not installed"})
        elif _version_tuple(version) < _version_tuple(minimum):
            out.append({"name": package, "status": "error", "detail": f"{version} is older than the {minimum} KikoLens needs"})
        else:
            out.append({"name": package, "status": "ok", "detail": version})
    return out


def check_optional() -> list:
    out = []
    for module, purpose in OPTIONAL.items():
        distribution = {"psycopg2": "psycopg2-binary"}.get(module, module)
        version = _installed(distribution) or _installed(module)
        out.append({"name": module, "status": "ok" if version else "warning",
                    "detail": version if version else f"not installed: needed for {purpose}"})
    return out


def check_home() -> dict:
    home = config.home_dir()
    try:
        home.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=home, delete=True):
            pass
        return {"name": "User folder", "status": "ok", "detail": f"{home} is writable"}
    except OSError as e:
        return {"name": "User folder", "status": "error", "detail": f"cannot write to {home}: {e}"}


def check_ollama(timeout: float = 2.0) -> dict:
    host = config.ollama_host()
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as response:
            models = [m.get("name", "") for m in json.loads(response.read()).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return {"name": "Ollama", "status": "warning", "detail": f"not reachable at {host}: the language-model features are off (everything else works)"}
    wanted = config.ollama_model()
    if not any(m == wanted or m.split(":")[0] == wanted.split(":")[0] for m in models):
        return {"name": "Ollama", "status": "warning", "detail": f"running, but model '{wanted}' is not downloaded (ollama pull {wanted})"}
    return {"name": "Ollama", "status": "ok", "detail": f"{host} with model {wanted}"}


def check_config() -> dict:
    found = config.problems()
    files = [str(p) for p in config.config_files() if p.exists()]
    if found:
        return {"name": "Configuration", "status": "warning", "detail": "; ".join(found)}
    return {"name": "Configuration", "status": "ok", "detail": ("read " + ", ".join(files)) if files else "no configuration file (defaults in use)"}


def run_checks(network: bool = True) -> list:
    checks = [check_python(), *check_required(), *check_optional(), check_home(), check_config()]
    if network:
        checks.append(check_ollama())
    return checks


def summary(checks: list) -> str:
    errors = sum(c["status"] == "error" for c in checks)
    warnings = sum(c["status"] == "warning" for c in checks)
    if errors:
        return f"{errors} problem(s) must be fixed before KikoLens works fully."
    return "Everything required is in place." + (f" {warnings} optional item(s) are missing." if warnings else "")


def importable(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False
