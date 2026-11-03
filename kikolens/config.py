"""Configuración compartida de KikoLens (se lee del entorno en cada llamada)."""
import os

DEFAULT_MODEL = "llama3.2"
DEFAULT_HOST = "http://localhost:11434"


def ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")


def ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)


def ollama_generate_url() -> str:
    return f"{ollama_host()}/api/generate"


def ollama_timeout(default: int = 120) -> int:
    try:
        return int(os.getenv("OLLAMA_TIMEOUT", default))
    except ValueError:
        return default


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
