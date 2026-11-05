"""Minería de texto local, sin enviar nada fuera.

  * `topics`: temas con NMF o LDA y el número de temas elegido por coherencia; documentos representativos, solapes entre temas y reparto.
  * `topics_over_time`: cómo cambia el peso de cada tema por periodo.
  * `find_duplicates`: textos casi iguales (erratas, copias con cambios pequeños) agrupados.
  * `similar_texts`: los textos más parecidos a uno o a una frase.
  * `extract_patterns`: correos, enlaces, teléfonos, importes, porcentajes, fechas, etiquetas y menciones (patrones, no reconocimiento de entidades).
  * `keyphrases` (RAKE), `collocations` (PMI) y `readability` (Flesch / Fernández-Huerta aproximados).
Reutiliza el idioma y las palabras vacías de `text`."""
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from kikolens.analyze import text as tx

MAX_K = 15
EXCERPT = 160


def _prepare(series, language):
    docs = tx._docs(series)
    return docs, language or tx.detect_language(docs)


def _excerpt(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= EXCERPT else text[: EXCERPT - 1] + "…"


# ── Temas ────────────────────────────────────────────────────────────────────
def _coherence(binary, top_ids: list) -> float:
    """Coherencia UMass: los términos principales de un tema deberían aparecer en los mismos textos."""
    doc_freq = np.asarray(binary.sum(axis=0)).ravel()
    scores = []
    for ids in top_ids:
        total, pairs = 0.0, 0
        for a in range(1, len(ids)):
            for b in range(a):
                joint = float(binary[:, ids[a]].multiply(binary[:, ids[b]]).sum())
                total += math.log((joint + 1) / max(doc_freq[ids[b]], 1))
                pairs += 1
        scores.append(total / max(pairs, 1))
    return float(np.mean(scores))


def _fit_topics(matrix, k: int, method: str, seed: int):
    if method == "nmf":
        model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=400)
        doc_topic = model.fit_transform(matrix)
    else:
        model = LatentDirichletAllocation(n_components=k, random_state=seed, learning_method="batch", max_iter=30)
        doc_topic = model.fit_transform(matrix)
    return model, doc_topic


def topics(series, k: int | None = None, max_k: int = 8, language: str | None = None, method: str = "nmf", top_words: int = 8, seed: int = 0) -> dict:
    """Descubre temas. Sin `k` prueba de 2 a `max_k` y se queda con el de mejor coherencia; la tabla `coherence` deja ver las demás opciones."""
    if method not in ("nmf", "lda"):
        raise ValueError("method must be 'nmf' or 'lda'.")
    if k is not None and not 2 <= k <= MAX_K:
        raise ValueError(f"The number of topics must be between 2 and {MAX_K}.")
    if not 2 <= max_k <= MAX_K:
        raise ValueError(f"max_k must be between 2 and {MAX_K}.")
    docs, language = _prepare(series, language)
    stop = tx._stop(language)
    if method == "nmf":
        vectorizer = TfidfVectorizer(stop_words=stop, token_pattern=tx._TOKEN.pattern, min_df=2, max_df=0.6, max_features=4000)
    else:
        vectorizer = CountVectorizer(stop_words=stop, token_pattern=tx._TOKEN.pattern, min_df=2, max_df=0.6, max_features=4000)
    try:
        matrix = vectorizer.fit_transform(docs)
    except ValueError:
        raise ValueError("No term appears in at least two texts (or all are too common); the texts may be too short or too uniform.") from None
    terms = vectorizer.get_feature_names_out()
    if len(terms) < 10:
        raise ValueError("Fewer than 10 usable terms: the texts are too short or too similar to find topics.")
    binary = (CountVectorizer(vocabulary=terms, token_pattern=tx._TOKEN.pattern, binary=True).fit_transform(docs)).tocsc()
    upper = min(max_k, max(2, len(docs) // 8))
    rows, fits = [], {}
    candidates = [k] if k else list(range(2, upper + 1))
    for kk in candidates:
        model, doc_topic = _fit_topics(matrix, kk, method, seed)
        top_ids = [np.argsort(-row)[:10] for row in model.components_]
        fits[kk] = (model, doc_topic)
        rows.append({"topics": kk, "coherence": _coherence(binary, top_ids)})
    table = pd.DataFrame(rows)
    chosen = int(k or table.loc[table["coherence"].idxmax(), "topics"])
    model, doc_topic = fits[chosen]
    weights = doc_topic / np.maximum(doc_topic.sum(axis=1, keepdims=True), 1e-12)
    dominant = weights.argmax(axis=1)
    strength = weights.max(axis=1)
    overview = []
    top_sets = []
    for t in range(chosen):
        order = np.argsort(-model.components_[t])[:top_words]
        words = [str(terms[i]) for i in order]
        top_sets.append(set(words))
        mine = np.where(dominant == t)[0]
        best = mine[np.argsort(-strength[mine])][:3] if len(mine) else np.array([], dtype=int)
        overview.append({"topic": t + 1, "label": ", ".join(words[:3]), "terms": ", ".join(words), "texts": int(len(mine)), "share": len(mine) / len(docs),
                         "examples": [_excerpt(docs.iloc[i]) for i in best]})
    summary_table = pd.DataFrame(overview)
    notes = []
    for a in range(chosen):
        for b in range(a + 1, chosen):
            shared = len(top_sets[a] & top_sets[b]) / top_words
            if shared >= 0.5:
                notes.append(f"Topics {a + 1} and {b + 1} share {shared:.0%} of their main terms: they may be one theme; try fewer topics.")
    small = [int(r["topic"]) for r in overview if r["share"] < 0.03]
    if small:
        notes.append(f"Topic(s) {', '.join(map(str, small))} cover less than 3% of the texts: they may be noise.")
    if float(np.median(strength)) < 0.6:
        notes.append("Most texts mix several topics (the leading topic has less than 60% of their weight): the themes overlap in this data.")
    assigned = pd.DataFrame({"topic": dominant + 1, "weight": strength}, index=docs.index)
    fig = go.Figure(go.Bar(x=[f"{r['topic']}: {r['label']}" for r in overview], y=summary_table["share"]))
    fig.update_layout(title="Share of texts by main topic", yaxis_tickformat=".0%", yaxis_title="Share of texts")
    return {"table": summary_table, "assignments": assigned, "coherence": table, "k": chosen, "method": method, "language": language, "texts": int(len(docs)),
            "chosen_automatically": k is None, "doc_topic": pd.DataFrame(weights, index=docs.index, columns=[f"topic {i + 1}" for i in range(chosen)]),
            "notes": notes, "figure": fig, "_vectorizer": vectorizer, "_model": model,
            "summary": (f"{chosen} topics" + (" (chosen by coherence)" if k is None else "") + ": " + "; ".join(f"{r['label']} ({r['share']:.0%})" for r in overview) + ".")}


def topics_over_time(df: pd.DataFrame, text: str, date: str, k: int | None = None, freq: str = "M", **kwargs) -> dict:
    """Peso de cada tema en cada periodo (proporción de textos cuyo tema principal es ese)."""
    for column in (text, date):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
    frame = df[[text, date]].copy()
    frame[date] = pd.to_datetime(frame[date], errors="coerce")
    frame = frame.dropna()
    frame = frame[frame[text].astype(str).str.strip() != ""]
    if len(frame) < tx.MIN_DOCS:
        raise ValueError(f"At least {tx.MIN_DOCS} texts with a valid date are needed.")
    res = topics(frame[text], k, **kwargs)
    labelled = frame.loc[res["assignments"].index].assign(topic=res["assignments"]["topic"])
    periods = labelled[date].dt.to_period(freq)
    counts = labelled.groupby([periods, "topic"]).size().unstack(fill_value=0)
    share = counts.div(counts.sum(axis=1), axis=0)
    share.columns = [f"{c}: {res['table'].set_index('topic').loc[c, 'label']}" for c in share.columns]
    share.index = share.index.to_timestamp()
    fig = go.Figure()
    for column in share.columns:
        fig.add_trace(go.Scatter(x=share.index, y=share[column], stackgroup="one", name=column, mode="lines"))
    fig.update_layout(title="Share of texts by topic over time", yaxis_tickformat=".0%")
    thin = int((counts.sum(axis=1) < 10).sum())
    notes = list(res["notes"]) + ([f"{thin} period(s) have fewer than 10 texts, so their shares are unstable."] if thin else [])
    first, last = share.iloc[0], share.iloc[-1]
    moved = (last - first).abs().idxmax()
    return {**res, "over_time": share, "counts": counts, "notes": notes, "figure": fig,
            "summary": res["summary"] + f" The topic that changed most between the first and last period is '{moved}' ({first[moved]:.0%} to {last[moved]:.0%})."}


# ── Duplicados y parecidos ───────────────────────────────────────────────────
def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text).lower())).strip()


def find_duplicates(series, threshold: float = 0.9) -> dict:
    """Grupos de textos casi idénticos (similitud del coseno sobre fragmentos de 3 a 4 letras, que tolera erratas y cambios pequeños)."""
    if not 0.5 <= threshold <= 1:
        raise ValueError("The threshold must be between 0.5 and 1.")
    docs = tx._docs(series)
    clean = docs.map(_normalise)
    keep = clean[clean != ""]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True).fit_transform(keep)
    n = matrix.shape[0]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lowest = {}
    transpose = matrix.T.tocsr()
    for start in range(0, n, 400):
        block = (matrix[start: start + 400] @ transpose).tocoo()
        for i, j, value in zip(block.row + start, block.col, block.data):
            if j > i and value >= threshold:
                a, b = find(i), find(j)
                if a != b:
                    parent[b] = a
                lowest[i] = min(lowest.get(i, 1.0), value)
                lowest[j] = min(lowest.get(j, 1.0), value)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    rows = []
    for members in (m for m in groups.values() if len(m) > 1):
        exact = len({keep.iloc[m] for m in members}) == 1
        rows.append({"size": len(members), "exact": exact, "similarity": min(lowest.get(m, 1.0) for m in members), "example": _excerpt(docs.loc[keep.index[members[0]]]),
                     "rows": [keep.index[m] for m in members]})
    table = pd.DataFrame(rows, columns=["size", "exact", "similarity", "example", "rows"]).sort_values("size", ascending=False, ignore_index=True)
    table.insert(0, "group", range(1, len(table) + 1))
    extra = int(table["size"].sum() - len(table)) if len(table) else 0
    notes = []
    if len(table) and (~table["exact"]).any():
        notes.append("Some groups differ in small ways (typos, punctuation, one changed word): read them before deleting rows.")
    return {"table": table, "groups": len(table), "redundant": extra, "texts": int(n), "threshold": threshold, "notes": notes,
            "summary": (f"{len(table)} group(s) of near-identical texts, {extra} redundant text(s) out of {n:,} (similarity at least {threshold:.2f})." if len(table)
                        else f"No near-duplicate texts found at similarity {threshold:.2f}.")}


def similar_texts(series, query, n: int = 5, language: str | None = None) -> pd.DataFrame:
    """Los `n` textos más parecidos a una frase (`query` es un texto) o a la fila de ese índice (`query` es una etiqueta del índice)."""
    docs, language = _prepare(series, language)
    vectorizer = TfidfVectorizer(stop_words=tx._stop(language), token_pattern=tx._TOKEN.pattern, ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(docs)
    try:
        by_row = query in docs.index
    except TypeError:
        by_row = False
    if by_row:
        exclude = int(docs.index.get_loc(query))
        vec = matrix[exclude]
    elif isinstance(query, str):
        vec, exclude = vectorizer.transform([query]), None
    else:
        raise ValueError("The query must be a text or the index label of one of the texts.")
    if vec.nnz == 0:
        raise ValueError("None of the words of the query appear in the texts.")
    scores = np.asarray((matrix @ vec.T).todense()).ravel()
    if exclude is not None:
        scores[exclude] = -1
    order = np.argsort(-scores)[: max(1, min(n, 50))]
    return pd.DataFrame({"row": docs.index[order], "similarity": scores[order], "text": [_excerpt(docs.iloc[i]) for i in order]}).reset_index(drop=True)


# ── Patrones ─────────────────────────────────────────────────────────────────
PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    "url": r"https?://[^\s<>\"')]+|www\.[^\s<>\"')]+",
    "phone": r"(?<![\w.])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3}[\s.-]?\d{2,4}[\s.-]?\d{2,4}(?![\w.])",
    "money": r"(?:[$€£]\s?\d[\d.,]*|\d[\d.,]*\s?(?:EUR|USD|GBP|€|\$|£))",
    "percent": r"\d+(?:[.,]\d+)?\s?%",
    "date": r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",
    "hashtag": r"(?<!\w)#\w+",
    "mention": r"(?<![\w@])@\w+",
}
PERSONAL = ("email", "phone")


def extract_patterns(series, kinds: list | None = None) -> dict:
    """Encuentra patrones conocidos en cada texto. Son expresiones regulares: aciertan con formatos habituales y no entienden el contexto."""
    kinds = list(kinds or PATTERNS)
    unknown = [k for k in kinds if k not in PATTERNS]
    if unknown:
        raise ValueError(f"Unknown pattern(s) {', '.join(unknown)}. Available: {', '.join(PATTERNS)}.")
    s = series.dropna().astype(str)
    s = s[s.str.strip() != ""]
    if s.empty:
        raise ValueError("There are no texts.")
    compiled = {k: re.compile(PATTERNS[k], re.IGNORECASE) for k in kinds}
    rows = []
    for index, text in s.items():
        for kind, rx in compiled.items():
            for match in rx.finditer(text):
                value = match.group(0).strip().rstrip(".,;:")
                if kind == "phone" and sum(c.isdigit() for c in value) < 8:
                    continue
                rows.append({"row": index, "kind": kind, "value": value})
    found = pd.DataFrame(rows, columns=["row", "kind", "value"])
    summary = pd.DataFrame([{"kind": k, "matches": int((found["kind"] == k).sum()), "texts": int(found.loc[found["kind"] == k, "row"].nunique()),
                             "share of texts": found.loc[found["kind"] == k, "row"].nunique() / len(s), "distinct values": int(found.loc[found["kind"] == k, "value"].nunique())}
                            for k in kinds])
    notes = []
    personal = [k for k in PERSONAL if k in kinds and (found["kind"] == k).any()]
    if personal:
        notes.append(f"Personal data found ({', '.join(personal)}): review it in the Privacy section before sharing or exporting these texts.")
    return {"table": summary, "found": found, "texts": int(len(s)), "notes": notes,
            "summary": (", ".join(f"{int(r['texts'])} text(s) with {r['kind']}" for _, r in summary.iterrows() if r["matches"]) or "No known patterns were found") + "."}


# ── Frases clave, colocaciones y legibilidad ─────────────────────────────────
def keyphrases(series, n: int = 15, language: str | None = None, max_words: int = 4) -> pd.DataFrame:
    """Frases clave con RAKE: se parten los textos en las palabras vacías y la puntuación, y cada palabra pesa más cuanto más se junta con otras."""
    docs, language = _prepare(series, language)
    stop = set(tx._stop(language))
    splitter = re.compile(r"[.,;:!?()\[\]\"'\n¿¡]+")
    word_freq, word_degree, phrase_count, phrases_by_doc = Counter(), Counter(), Counter(), []
    for doc in docs:
        found = []
        for chunk in splitter.split(doc.lower()):
            current = []
            for word in re.findall(r"[^\W\d_]{2,}", chunk) + ["\0"]:
                if word == "\0" or word in stop:
                    if current:
                        found.append(tuple(current))
                    current = []
                else:
                    current.append(word)
        phrases_by_doc.append(found)
        for phrase in found:
            if len(phrase) > max_words:
                continue
            phrase_count[phrase] += 1
            for word in phrase:
                word_freq[word] += 1
                word_degree[word] += len(phrase)
    rows = []
    for phrase, count in phrase_count.items():
        if count < 2:
            continue
        score = sum(word_degree[w] / word_freq[w] for w in phrase)
        rows.append({"phrase": " ".join(phrase), "texts": count, "score": score})
    if not rows:
        raise ValueError("No phrase appears in at least two texts; the texts may be too short or too different.")
    out = pd.DataFrame(rows)
    out["rank"] = out["score"] * np.log1p(out["texts"])
    return out.sort_values("rank", ascending=False, ignore_index=True).head(n).drop(columns="rank")


def collocations(series, n: int = 15, min_count: int = 5, language: str | None = None) -> pd.DataFrame:
    """Pares de palabras seguidas que aparecen juntas más de lo que daría el azar (información mutua puntual normalizada)."""
    if min_count < 2:
        raise ValueError("min_count must be at least 2.")
    docs, language = _prepare(series, language)
    stop = set(tx._stop(language))
    unigrams, bigrams, total = Counter(), Counter(), 0
    for doc in docs:
        words = [w for w in tx._TOKEN.findall(doc.lower())]
        unigrams.update(words)
        total += len(words)
        for a, b in zip(words, words[1:]):
            if a not in stop and b not in stop:
                bigrams[(a, b)] += 1
    n_pairs = max(sum(bigrams.values()), 1)
    rows = []
    for (a, b), count in bigrams.items():
        if count < min_count:
            continue
        p_ab, p_a, p_b = count / n_pairs, unigrams[a] / total, unigrams[b] / total
        pmi = math.log2(p_ab / (p_a * p_b))
        rows.append({"words": f"{a} {b}", "count": count, "pmi": pmi, "npmi": pmi / -math.log2(p_ab)})
    if not rows:
        raise ValueError(f"No pair of words appears at least {min_count} times; lower min_count or use more texts.")
    return pd.DataFrame(rows).sort_values(["npmi", "count"], ascending=False, ignore_index=True).head(n)


def _syllables(word: str) -> int:
    return max(1, len(re.findall(r"[aeiouyáéíóúüàèìòù]+", word.lower())))


def readability(series, language: str | None = None) -> dict:
    """Facilidad de lectura por texto: Flesch (inglés) o Fernández-Huerta (español), de 0 (muy difícil) a 100 (muy fácil). Cuenta sílabas de forma aproximada."""
    docs, language = _prepare(series, language)
    rows = []
    for index, doc in docs.items():
        sentences = max(1, len([s for s in re.split(r"[.!?]+", doc) if s.strip()]))
        words = re.findall(r"[^\W\d_]+", doc)
        if not words:
            continue
        syllables = sum(_syllables(w) for w in words)
        if language == "spanish":
            score = 206.84 - 60 * syllables / len(words) - 1.02 * len(words) / sentences
        else:
            score = 206.835 - 1.015 * len(words) / sentences - 84.6 * syllables / len(words)
        rows.append({"row": index, "words": len(words), "sentences": sentences, "words per sentence": len(words) / sentences, "score": float(np.clip(score, 0, 100))})
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("The texts contain no words.")
    bands = pd.cut(table["score"], [-0.1, 30, 50, 60, 70, 80, 100.1], labels=["very difficult", "difficult", "fairly difficult", "standard", "fairly easy", "easy"])
    distribution = bands.value_counts().reindex(bands.cat.categories)
    median = float(table["score"].median())
    label = str(pd.cut([median], [-0.1, 30, 50, 60, 70, 80, 100.1], labels=bands.cat.categories)[0])
    return {"table": table, "language": language, "median": median, "distribution": distribution, "label": label,
            "notes": ["Syllables are counted from vowel groups, so scores are approximate; compare texts with each other rather than reading one number as exact."],
            "summary": f"Median reading ease {median:.0f} of 100 ({label}) with the {'Fernández-Huerta' if language == 'spanish' else 'Flesch'} formula; "
                       f"{table['words per sentence'].median():.1f} words per sentence."}
