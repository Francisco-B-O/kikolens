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
