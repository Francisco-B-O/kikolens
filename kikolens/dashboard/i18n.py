"""Idioma del panel. Los textos fijos se traducen con un diccionario (`es.py`); lo que no está en él se queda en inglés.
En vez de tocar cada vista, `install` envuelve las funciones de Streamlit que escriben texto y traduce su etiqueta."""
import functools
import re

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from kikolens.dashboard.es import ES

LANGUAGES = {"English": "en", "Español": "es"}
TABLES = {"es": ES}
PREFIXES = ("#### ", "### ", "## ")
# funciones que reciben un texto visible como primer argumento (o como `label`/`text`/`body`)
TEXT_FUNCTIONS = ("markdown", "caption", "subheader", "header", "title", "button", "download_button", "selectbox", "multiselect",
                  "checkbox", "slider", "select_slider", "radio", "text_input", "text_area", "number_input", "file_uploader",
                  "expander", "metric", "date_input", "toggle", "info", "warning", "success", "error", "spinner", "progress",
                  "segmented_control", "pills")
TEXT_KEYWORDS = ("label", "help", "text", "body", "placeholder")
_original = {}


def current() -> str:
    try:
        lang = st.session_state.get("lang", "en")
    except Exception:  # fuera de Streamlit
        return "en"
    return lang if lang in TABLES else "en"


def t(text, lang: str | None = None):
    """El texto en el idioma actual; si no hay traducción, el mismo texto."""
    lang = lang or current()
    if lang == "en" or not isinstance(text, str):
        return text
    table = TABLES[lang]
    if text in table:
        return table[text]
    for prefix in PREFIXES:
        if text.startswith(prefix) and text[len(prefix):] in table:
            return prefix + table[text[len(prefix):]]
    bold = re.fullmatch(r"\*\*(.+?)\*\*", text, flags=re.S)
    if bold and bold.group(1) in table:
        return f"**{table[bold.group(1)]}**"
    return text


def _translate_call(name: str, args: tuple, kwargs: dict, skip: int = 0):
    args = list(args)
    if len(args) > skip and isinstance(args[skip], str):
        args[skip] = t(args[skip])
    for key in TEXT_KEYWORDS:
        if isinstance(kwargs.get(key), str):
            kwargs[key] = t(kwargs[key])
    return args, kwargs


def _wrap_function(name: str, fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        args, kwargs = _translate_call(name, args, kwargs)
        return fn(*args, **kwargs)
    return inner


def _wrap_method(name: str, fn):
    @functools.wraps(fn)
    def inner(self, *args, **kwargs):
        args, kwargs = _translate_call(name, args, kwargs)
        return fn(self, *args, **kwargs)
    return inner


def _wrap_tabs(fn, method: bool):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        pos = 1 if method else 0
        args = list(args)
        if len(args) > pos and isinstance(args[pos], (list, tuple)):
            args[pos] = [t(a) if isinstance(a, str) else a for a in args[pos]]
        return fn(*args, **kwargs)
    return inner


def install(lang: str | None = None) -> None:
    """Activa el idioma: envuelve las funciones de Streamlit (las del módulo y las de cada contenedor) o las restaura."""
    lang = lang or current()
    if not _original:
        for name in TEXT_FUNCTIONS + ("tabs",):
            if hasattr(st, name):
                _original[("module", name)] = getattr(st, name)
            if hasattr(DeltaGenerator, name):
                _original[("class", name)] = getattr(DeltaGenerator, name)
    for (kind, name), fn in _original.items():
        target = st if kind == "module" else DeltaGenerator
        if lang == "en":
            setattr(target, name, fn)
        elif name == "tabs":
            setattr(target, name, _wrap_tabs(fn, kind == "class"))
        else:
            setattr(target, name, (_wrap_function if kind == "module" else _wrap_method)(name, fn))
