"""Cliente del modelo de lenguaje local (Ollama). Toda la IA de KikoLens pasa por aquí: el modelo nunca ejecuta código, solo devuelve texto o JSON
que el programa valida antes de usar. El «transporte» (la función que habla con Ollama) se puede sustituir, y así las pruebas no necesitan un modelo."""
import json
import re

import requests

from kikolens import config


class LLMError(RuntimeError):
    """El modelo no está disponible o no respondió."""


def ollama_transport(host: str | None = None, model: str | None = None, timeout: float | None = None, temperature: float = 0.1):
    """La función que envía un pedido a Ollama y devuelve el texto de la respuesta."""
    def send(payload: dict) -> str:
        body = {"model": model or config.ollama_model(), "stream": False, "options": {"temperature": temperature}, **payload}
        try:
            response = requests.post(f"{(host or config.ollama_host()).rstrip('/')}/api/generate", json=body,
                                     timeout=timeout or config.ollama_timeout())
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.Timeout:
            raise LLMError("The local model took too long to answer. Try a lighter model or raise the timeout in kikolens.toml.") from None
        except requests.exceptions.ConnectionError:
            raise LLMError("The local model is not running. Start Ollama and pull a model (for example: ollama pull llama3.2).") from None
        except (requests.exceptions.RequestException, ValueError) as e:
            raise LLMError(f"The local model failed: {e}") from None
    return send


class LLM:
    def __init__(self, transport=None, retries: int = 2):
        self.transport = transport or ollama_transport()
        self.retries = retries
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        self.calls += 1
        payload = {"prompt": prompt}
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        return self.transport(payload)

    def complete_json(self, prompt: str, validate=None, system: str | None = None):
        """Pide JSON, lo extrae y, si `validate(obj)` lanza ValueError, devuelve el error al modelo para que lo corrija (hasta `retries` veces)."""
        last = ""
        feedback = ""
        for attempt in range(self.retries + 1):
            text = self.complete(prompt + feedback, system, json_mode=True)
            try:
                obj = extract_json(text)
                return validate(obj) if validate else obj
            except ValueError as e:
                last = str(e)
                feedback = (f"\n\nYour previous answer was not valid: {last}\nPrevious answer:\n{text[:600]}\n"
                            "Answer again with ONLY the corrected JSON.")
        raise LLMError(f"The model did not return a usable answer after {self.retries + 1} tries ({last}).")

    def available(self, timeout: float = 2.0) -> bool:
        """¿Responde Ollama? Con un transporte propio (pruebas) se da por disponible."""
        if getattr(self.transport, "__name__", "") != "send":
            return True
        try:
            return requests.get(f"{config.ollama_host()}/api/tags", timeout=timeout).ok
        except requests.exceptions.RequestException:
            return False


def extract_json(text: str):
    """El primer objeto o lista JSON de un texto (quita bloques de código y charla alrededor). ValueError si no hay ninguno."""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        while start != -1:
            depth, in_string, escaped = 0, False, False
            for i in range(start, len(cleaned)):
                ch = cleaned[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                elif ch == '"':
                    in_string = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cleaned[start:i + 1])
                        except json.JSONDecodeError:
                            break
            start = cleaned.find(opener, start + 1)
    raise ValueError("the answer contains no valid JSON")
