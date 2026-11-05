"""Consulta del glosario y de las guías de lectura, en inglés o en español."""
import difflib
import re

import pandas as pd

from kikolens.learn.glossary_data import TERMS
from kikolens.learn.guides_data import GUIDES

LANGUAGES = ("en", "es")
_BY_KEY = {row[0]: row for row in TERMS}


def _lang(lang: str) -> int:
    if lang not in LANGUAGES:
        raise ValueError("The language must be 'en' or 'es'.")
    return LANGUAGES.index(lang)


def _entry(row: tuple, lang: str) -> dict:
    i = _lang(lang)
    return {"key": row[0], "term": row[1 + i], "definition": row[3 + i], "watch_out": row[5 + i], "see_also": [{"key": k, "term": _BY_KEY[k][1 + i]} for k in row[7]]}


def get(term: str, lang: str = "en") -> dict:
    """Un término por su clave (`p_value`) o por su nombre en cualquiera de los dos idiomas."""
    _lang(lang)
    wanted = term.strip().lower()
    for row in TERMS:
        if wanted in (row[0], row[1].lower(), row[2].lower()):
            return _entry(row, lang)
    i = _lang(lang)
    names = {r[1 + i].lower(): r[1 + i] for r in TERMS}
    close = difflib.get_close_matches(wanted, list(names), n=3, cutoff=0.6) or [n.lower() for n in search(term, lang)["term"].head(3)]
    hint = f" Did you mean: {', '.join(names.get(c, c) for c in close)}?" if close else ""
    raise ValueError(f"'{term}' is not in the glossary.{hint}")


def search(query: str, lang: str = "en") -> pd.DataFrame:
    """Términos cuyo nombre o definición contienen el texto buscado, los de nombre primero."""
    i = _lang(lang)
    q = query.strip().lower()
    if not q:
        return glossary(lang)
    rows = []
    for row in TERMS:
        name, definition = row[1 + i].lower(), row[3 + i].lower()
        if q in name or q in row[0] or q in definition:
            rows.append((0 if q in name or q in row[0] else 1, row))
    rows.sort(key=lambda r: (r[0], r[1][1 + i].lower()))
    return pd.DataFrame([{"term": r[1][1 + i], "key": r[1][0], "definition": r[1][3 + i]} for r in rows], columns=["term", "key", "definition"])


def glossary(lang: str = "en") -> pd.DataFrame:
    """Todos los términos por orden alfabético."""
    i = _lang(lang)
    table = pd.DataFrame([{"term": r[1 + i], "key": r[0], "definition": r[3 + i]} for r in TERMS])
    return table.sort_values("term", key=lambda s: s.str.lower(), ignore_index=True)


def terms_in(text: str, lang: str = "en") -> list:
    """Claves de los términos del glosario que aparecen escritos en el texto (para enlazar explicaciones)."""
    i = _lang(lang)
    found = []
    for row in TERMS:
        name = re.escape(row[1 + i].lower())
        if re.search(rf"(?<!\w){name}(?!\w)", text.lower()):
            found.append(row[0])
    return found


def guide(page: str, lang: str = "en") -> dict | None:
    """La guía «cómo leer esta página», o None si esa sección no tiene."""
    i = _lang(lang)
    if page not in GUIDES:
        return None
    row = GUIDES[page]
    return {"page": page, "reads": row[i], "watch_out": row[2 + i], "terms": [_entry(_BY_KEY[k], lang) for k in row[4]]}


def pages_with_guide() -> list:
    return list(GUIDES)
