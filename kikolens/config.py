"""Configuración compartida de KikoLens (se lee del entorno en cada llamada)."""
import os
import tomllib
from pathlib import Path

DEFAULT_MODEL = "llama3.2"
DEFAULT_HOST = "http://localhost:11434"
CONFIG_NAME = "kikolens.toml"
KNOWN = {"ollama": {"host", "model", "timeout"}, "dashboard": {"port", "language"}}


def config_files() -> list:
    """Archivos de configuración, del más prioritario al menos: `kikolens.toml` en la carpeta actual y `config.toml` en la de usuario."""
    return [Path.cwd() / CONFIG_NAME, home_dir() / "config.toml"]


def settings() -> dict:
    """Lo que dicen los archivos de configuración (las variables de entorno mandan sobre ello). Un archivo roto o con claves
    desconocidas se ignora con su aviso en `problems()`, nunca impide arrancar."""
    merged: dict = {}
    for path in reversed(config_files()):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        for section, values in data.items():
            if section in KNOWN and isinstance(values, dict):
                merged.setdefault(section, {}).update({k: v for k, v in values.items() if k in KNOWN[section]})
    return merged


def problems() -> list:
    """Avisos sobre los archivos de configuración encontrados: no son TOML válido o traen claves que no se conocen."""
    found = []
    for path in config_files():
        if not path.exists():
            continue
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as e:
            found.append(f"{path}: not valid TOML ({e})")
            continue
        for section, values in data.items():
            if section not in KNOWN:
                found.append(f"{path}: unknown section [{section}]")
            elif isinstance(values, dict):
                found += [f"{path}: unknown key '{k}' in [{section}]" for k in values if k not in KNOWN[section]]
    return found


def _setting(section: str, key: str, default):
    return settings().get(section, {}).get(key, default)


def ollama_host() -> str:
    return str(os.getenv("OLLAMA_HOST") or _setting("ollama", "host", DEFAULT_HOST)).rstrip("/")


def ollama_model() -> str:
    return str(os.getenv("OLLAMA_MODEL") or _setting("ollama", "model", DEFAULT_MODEL))


def ollama_generate_url() -> str:
    return f"{ollama_host()}/api/generate"


def ollama_timeout(default: int = 120) -> int:
    try:
        return int(os.getenv("OLLAMA_TIMEOUT") or _setting("ollama", "timeout", default))
    except (TypeError, ValueError):
        return default


def dashboard_port(default: int = 8501) -> int:
    try:
        return int(_setting("dashboard", "port", default))
    except (TypeError, ValueError):
        return default


def default_language() -> str:
    """`en` o `es`, según `[dashboard] language` (por defecto `en`)."""
    value = str(_setting("dashboard", "language", "en")).lower()
    return value if value in ("en", "es") else "en"


def home_dir() -> Path:
    """Carpeta de datos del usuario (registro de modelos). Se puede cambiar con KIKOLENS_HOME."""
    return Path(os.getenv("KIKOLENS_HOME") or Path.home() / ".kikolens")


# Opciones de Streamlit: las usa `kikolens start` y están reflejadas en .streamlit/config.toml.
STREAMLIT_OPTIONS = {
    "browser.gatherUsageStats": False,
    "client.toolbarMode": "minimal",
    "theme.light.primaryColor": "#9a3412",
    "theme.light.backgroundColor": "#faf8f3",
    "theme.light.secondaryBackgroundColor": "#f3f0e8",
    "theme.light.textColor": "#1c1b19",
    "theme.dark.primaryColor": "#e0865a",
    "theme.dark.backgroundColor": "#171614",
    "theme.dark.secondaryBackgroundColor": "#1f1e1b",
    "theme.dark.textColor": "#ece9e1",
}


def streamlit_flags() -> list:
    flags = []
    for key, value in STREAMLIT_OPTIONS.items():
        flags += [f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)]
    return flags
