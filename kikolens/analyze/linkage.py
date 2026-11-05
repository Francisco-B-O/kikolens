"""Cruce aproximado de registros: une dos tablas por nombres que no coinciden exactamente («García, Ana» y «ana garcia»). Normaliza
(minúsculas, sin acentos ni signos, palabras ordenadas), busca candidatos con las palabras que comparten y puntúa cada par con
la similitud de caracteres. Cada fila de la izquierda recibe su mejor pareja si supera el umbral, y se avisa de los empates."""
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd

MAX_COMMON = 0.2         # una palabra presente en más del 20 % de las filas no sirve para encontrar candidatos
MAX_CANDIDATES = 50


def normalise(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii").lower()
    return " ".join(sorted(re.findall(r"[a-z0-9]+", text)))


def similarity(a: str, b: str) -> float:
    """0 a 1. Compara las cadenas normalizadas y, como alternativa más permisiva, sus iniciales y palabras comunes."""
    if not a or not b:
        return 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    overlap = len(ta & tb) / max(len(ta | tb), 1)
    return max(ratio, 0.5 * ratio + 0.5 * overlap)


def link_records(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str, threshold: float = 0.85,
                 ambiguity: float = 0.03) -> dict:
    """Mejor pareja de cada fila de `left` en `right`. Devuelve `matches` (con puntuación y aviso de ambigüedad) y recuentos."""
    for frame, col in ((left, left_on), (right, right_on)):
        if col not in frame.columns:
            raise ValueError(f"Column '{col}' not found.")
    if not 0.5 <= threshold <= 1:
        raise ValueError("The threshold must be between 0.5 and 1.")
    if len(left) == 0 or len(right) == 0:
        raise ValueError("Both tables need at least one row.")
    lnorm = left[left_on].map(normalise)
    rnorm = right[right_on].map(normalise)
    index = defaultdict(list)
    for pos, text in enumerate(rnorm):
        for token in set(text.split()):
            index[token].append(pos)
    limit = max(5, int(MAX_COMMON * len(right)))
    exact = defaultdict(list)
    for pos, text in enumerate(rnorm):
        if text:
            exact[text].append(pos)
    rows = []
    for lpos, text in enumerate(lnorm):
        if not text:
            continue
        candidates = list(exact.get(text, []))
        if not candidates:
            seen = defaultdict(int)
            for token in set(text.split()):
                postings = index.get(token, [])
                if len(postings) <= limit:
                    for r in postings:
                        seen[r] += 1
            candidates = [r for r, _ in sorted(seen.items(), key=lambda kv: -kv[1])[:MAX_CANDIDATES]]
        scored = sorted(((similarity(text, rnorm.iloc[r]), r) for r in candidates), reverse=True)
        if scored and scored[0][0] >= threshold:
            best, runner_up = scored[0], (scored[1][0] if len(scored) > 1 else 0.0)
            rows.append({"left row": lpos, "right row": best[1], "left value": left[left_on].iloc[lpos],
                         "right value": right[right_on].iloc[best[1]], "score": round(best[0], 4),
                         "exact": best[0] >= 0.9999, "ambiguous": runner_up >= best[0] - ambiguity})
    matches = pd.DataFrame(rows, columns=["left row", "right row", "left value", "right value", "score", "exact", "ambiguous"])
    n = len(matches)
    return {"matches": matches, "left rows": len(left), "matched": n, "unmatched": len(left) - n,
            "ambiguous": int(matches["ambiguous"].sum()) if n else 0,
            "summary": (f"{n:,} of {len(left):,} rows found a partner at similarity {threshold:.0%} or more"
                        + (f"; {int(matches['ambiguous'].sum())} of them had a close runner-up, so check them" if n and matches["ambiguous"].any() else "")
                        + (f". {int((~matches['exact']).sum())} are not exact matches." if n else "."))}


def merge_linked(left: pd.DataFrame, right: pd.DataFrame, matches: pd.DataFrame, suffix: str = "_right") -> pd.DataFrame:
    """Tabla izquierda con las columnas de la pareja encontrada (vacías si no la hubo) y la puntuación de cada cruce."""
    pairs = matches.set_index("left row")
    partner = right.iloc[pairs["right row"].to_numpy()].copy()
    partner.index = pairs.index
    partner = partner.add_suffix(suffix)
    out = left.reset_index(drop=True).join(partner)
    out["match score"] = pairs["score"].reindex(out.index)
    return out
