"""Análisis de columnas de texto libre, todo en local: estadísticas, términos clave (TF-IDF), términos que distinguen
grupos, agrupación de textos, clasificación de texto, tono por léxico (aproximado) y resumen opcional con Ollama."""
import re

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from kikolens.config import ollama_generate_url, ollama_model, ollama_timeout
from kikolens.utils.dtypes import is_text

MIN_DOCS = 10
MAX_DOCS = 20000
SPANISH_STOP = frozenset("""a al algo algunos ante antes como con contra cual cuando de del desde donde durante e el ella ellas
ellos en entre era eran es esa esas ese eso esos esta estaba estan estar este esto estos fue fueron ha han has hasta hay la las le
les lo los me mi mis mucho muy más mas ni no nos nosotros o os otra otro para pero poco por porque que quien se sea ser si sin sobre
son su sus también tambien tan te tiene tienen todo todos tu tus un una uno unos y ya yo""".split())
STOPWORDS = {"english": ENGLISH_STOP_WORDS, "spanish": SPANISH_STOP}
_TOKEN = re.compile(r"(?u)\b[^\W\d_]{2,}\b")

# Léxico mínimo y abierto: da una idea del tono, no sustituye a un modelo entrenado.
POSITIVE = {"english": set("good great excellent love loved amazing perfect best happy fast easy nice wonderful fantastic "
                           "recommend helpful friendly useful reliable smooth satisfied awesome pleasant".split()),
            "spanish": set("bueno buena genial excelente encanta perfecto mejor feliz rapido rápido facil fácil bonito "
                           "recomiendo util útil amable fiable satisfecho increible increíble".split())}
NEGATIVE = {"english": set("bad terrible awful hate hated worst poor slow broken useless disappointed disappointing "
                           "problem problems difficult expensive late rude waste refund fail failed".split()),
            "spanish": set("malo mala terrible horrible odio peor lento roto inutil inútil decepcionado decepcionante "
                           "problema problemas dificil difícil caro tarde grosero fallo falla".split())}


def free_text_columns(df: pd.DataFrame) -> list:
    """Columnas de texto que parecen frases (no categorías ni identificadores)."""
    found = []
    for col in df.columns:
        if not is_text(df[col]):
            continue
        s = df[col].dropna().astype(str)
        if len(s) < MIN_DOCS:
            continue
        words = s.str.split().str.len().mean()
        if words >= 4 and s.nunique() / len(s) > 0.3:
            found.append(col)
    return found


def _docs(series: pd.Series) -> pd.Series:
    docs = series.dropna().astype(str).str.strip()
    docs = docs[docs != ""]
    if len(docs) < MIN_DOCS:
        raise ValueError(f"At least {MIN_DOCS} non-empty texts are needed.")
    if len(docs) > MAX_DOCS:
        docs = docs.sample(MAX_DOCS, random_state=0)
    return docs


def detect_language(series: pd.Series) -> str:
    sample = " ".join(series.dropna().astype(str).head(300)).lower()
    words = _TOKEN.findall(sample)
    if not words:
        return "english"
    scores = {lang: sum(w in stop for w in words) for lang, stop in STOPWORDS.items()}
    return max(scores, key=scores.get)


def _stop(language: str):
    if language not in STOPWORDS:
        raise ValueError(f"Language must be one of {list(STOPWORDS)}.")
    return list(STOPWORDS[language])


# ── Estadísticas y términos ──────────────────────────────────────────────────
def text_stats(series: pd.Series) -> dict:
    raw = series.astype(str).where(series.notna(), "")
    words = raw.str.split().str.len()
    clean = raw[raw.str.strip() != ""]
    return {"rows": int(len(series)), "empty": int(len(series) - len(clean)), "duplicates": int(clean.duplicated().sum()),
            "mean_words": float(words[raw.str.strip() != ""].mean()) if len(clean) else 0.0,
            "median_chars": float(clean.str.len().median()) if len(clean) else 0.0,
            "longest": int(clean.str.len().max()) if len(clean) else 0, "language": detect_language(series),
            "words": words[raw.str.strip() != ""]}


def top_terms(series: pd.Series, n: int = 20, ngram: int = 1, language: str | None = None) -> pd.DataFrame:
    """Términos más frecuentes (en cuántos textos aparecen) y su peso TF-IDF medio."""
    if ngram not in (1, 2, 3):
        raise ValueError("ngram must be 1, 2 or 3.")
    docs = _docs(series)
    language = language or detect_language(docs)
    counts = CountVectorizer(stop_words=_stop(language), token_pattern=_TOKEN.pattern, lowercase=True,
                             ngram_range=(ngram, ngram), min_df=2, binary=True)
    tfidf = TfidfVectorizer(stop_words=_stop(language), token_pattern=_TOKEN.pattern, ngram_range=(ngram, ngram), min_df=2)
    try:
        matrix = counts.fit_transform(docs)
        weights = np.asarray(tfidf.fit_transform(docs).mean(axis=0)).ravel()
    except ValueError:
        raise ValueError("No term appears in at least two texts; the texts may be too short or too different.") from None
    freq = np.asarray(matrix.sum(axis=0)).ravel()
    terms = counts.get_feature_names_out()
    out = pd.DataFrame({"term": terms, "texts": freq, "share": freq / len(docs), "tf-idf": weights})
    return out.sort_values(["texts", "tf-idf"], ascending=False, ignore_index=True).head(n)


def terms_by_group(df: pd.DataFrame, text: str, group: str, n: int = 10, language: str | None = None) -> pd.DataFrame:
    """Términos que más distinguen a cada grupo del resto (razón de probabilidades suavizada)."""
    for col in (text, group):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    data = df[[text, group]].dropna()
    data = data[data[text].astype(str).str.strip() != ""]
    if data[group].nunique() < 2 or data[group].nunique() > 12:
        raise ValueError("The group column needs between 2 and 12 distinct values.")
    language = language or detect_language(data[text])
    vec = CountVectorizer(stop_words=_stop(language), token_pattern=_TOKEN.pattern, min_df=3, binary=True)
    try:
        X = vec.fit_transform(data[text].astype(str))
    except ValueError:
        raise ValueError("Not enough repeated terms to compare groups.") from None
    terms, labels = vec.get_feature_names_out(), data[group].astype(str).to_numpy()
    total = np.asarray(X.sum(axis=0)).ravel()
    rows = []
    for g in sorted(set(labels)):
        mask = labels == g
        if mask.sum() < 3:
            continue
        inside = np.asarray(X[mask].sum(axis=0)).ravel()
        outside = total - inside
        odds = np.log((inside + 0.5) / (mask.sum() - inside + 0.5)) - np.log((outside + 0.5) / ((~mask).sum() - outside + 0.5))
        for i in np.argsort(-odds)[:n]:
            if inside[i] >= 3:
                rows.append({"group": g, "term": terms[i], "in group": int(inside[i]), "elsewhere": int(outside[i]),
                             "log odds": float(odds[i])})
    return pd.DataFrame(rows, columns=["group", "term", "in group", "elsewhere", "log odds"])


def search_texts(series: pd.Series, query: str, context: int = 40, limit: int = 30) -> pd.DataFrame:
    """Busca una palabra o frase y muestra el fragmento que la rodea."""
    query = query.strip()
    if len(query) < 2:
        raise ValueError("Type at least two characters to search.")
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    rows = []
    for idx, text in series.dropna().astype(str).items():
        m = pattern.search(text)
        if m:
            rows.append({"row": idx, "excerpt": ("..." if m.start() > context else "") +
                         text[max(0, m.start() - context): m.end() + context].replace("\n", " ") +
                         ("..." if m.end() + context < len(text) else "")})
            if len(rows) >= limit:
                break
    return pd.DataFrame(rows, columns=["row", "excerpt"])


# ── Tono aproximado ──────────────────────────────────────────────────────────
def lexicon_sentiment(series: pd.Series, language: str | None = None) -> pd.DataFrame:
    """Tono aproximado por léxico: (positivas - negativas) / palabras con tono. No entiende ironía ni negaciones."""
    language = language or detect_language(series)
    if language not in POSITIVE:
        raise ValueError(f"Language must be one of {list(POSITIVE)}.")
    rows = []
    for idx, text in series.items():
        words = _TOKEN.findall(str(text).lower()) if pd.notna(text) else []
        pos, neg = sum(w in POSITIVE[language] for w in words), sum(w in NEGATIVE[language] for w in words)
        score = (pos - neg) / (pos + neg) if pos + neg else 0.0
        rows.append({"positive words": pos, "negative words": neg, "score": score,
                     "tone": "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral" if pos + neg else "no signal"})
    return pd.DataFrame(rows, index=series.index)


# ── Agrupar ──────────────────────────────────────────────────────────────────
def cluster_texts(series: pd.Series, k: int = 5, language: str | None = None, seed: int = 0) -> dict:
    """Agrupa textos parecidos (TF-IDF + k-medias) y describe cada grupo por sus términos y ejemplos."""
    docs = _docs(series)
    if not 2 <= k <= min(15, len(docs) // 3):
        raise ValueError(f"The number of groups must be between 2 and {min(15, len(docs) // 3)} for this many texts.")
    language = language or detect_language(docs)
    vec = TfidfVectorizer(stop_words=_stop(language), token_pattern=_TOKEN.pattern, min_df=2, max_df=0.8, sublinear_tf=True)
    try:
        X = vec.fit_transform(docs)
    except ValueError:
        raise ValueError("The texts share too few terms to be grouped.") from None
    reduced = TruncatedSVD(n_components=min(100, X.shape[1] - 1, X.shape[0] - 1), random_state=seed).fit_transform(X) \
        if X.shape[1] > 400 else X.toarray()  # con pocos términos no hace falta reducir
    reduced = reduced / (np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-12)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(reduced)
    terms = vec.get_feature_names_out()
    summary = []
    for c in range(k):
        mask = labels == c
        centroid = np.asarray(X[mask].mean(axis=0)).ravel()
        top = [terms[i] for i in np.argsort(-centroid)[:6]]
        similarity = reduced[mask] @ reduced[mask].mean(axis=0)
        example = docs[mask].iloc[int(np.argmax(similarity))]
        summary.append({"group": c, "texts": int(mask.sum()), "share": float(mask.mean()), "key terms": ", ".join(top),
                        "example": example[:160]})
    return {"summary": pd.DataFrame(summary).sort_values("texts", ascending=False, ignore_index=True),
            "labels": pd.Series(labels, index=docs.index, name="text_group"), "language": language}


# ── Clasificar ───────────────────────────────────────────────────────────────
def train_text_classifier(df: pd.DataFrame, text: str, target: str, language: str | None = None, folds: int = 5,
                          seed: int = 0) -> dict:
    """Clasificador de texto (TF-IDF + regresión logística) evaluado con validación cruzada."""
    for col in (text, target):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    data = df[[text, target]].dropna()
    data = data[data[text].astype(str).str.strip() != ""]
    counts = data[target].astype(str).value_counts()
    if len(counts) < 2:
        raise ValueError("The target needs at least two different values.")
    if len(counts) > 20:
        raise ValueError("The target has more than 20 classes; choose a column with fewer categories.")
    data = data[data[target].astype(str).isin(counts[counts >= folds].index)]
    if len(data) < 30 or data[target].nunique() < 2:
        raise ValueError(f"Need at least 30 texts and {folds} examples of each class to train a classifier.")
    language = language or detect_language(data[text])
    X, y = data[text].astype(str), data[target].astype(str)
    pipe = Pipeline([("tfidf", TfidfVectorizer(stop_words=_stop(language), token_pattern=_TOKEN.pattern, min_df=2,
                                               ngram_range=(1, 2), sublinear_tf=True)),
                     ("clf", LogisticRegression(max_iter=1000, C=3.0, class_weight="balanced"))])
    pred = cross_val_predict(pipe, X, y, cv=min(folds, int(y.value_counts().min())))
    baseline = float(y.value_counts(normalize=True).iloc[0])
    pipe.fit(X, y)
    terms = pipe.named_steps["tfidf"].get_feature_names_out()
    coefs, classes, drivers = pipe.named_steps["clf"].coef_, list(pipe.classes_), []
    for i, c in enumerate(classes if len(classes) > 2 else classes[1:]):
        weights = coefs[i if len(classes) > 2 else 0]
        for j in np.argsort(-weights)[:8]:
            drivers.append({"class": c, "term": terms[j], "weight": float(weights[j])})
    return {"pipeline": pipe, "classes": classes, "language": language, "rows": len(data),
            "accuracy": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred, average="weighted")),
            "baseline": baseline, "drivers": pd.DataFrame(drivers, columns=["class", "term", "weight"]),
            "predicted": pd.Series(pred, index=data.index), "actual": y}


def classify_texts(result: dict, series: pd.Series) -> pd.DataFrame:
    pipe = result["pipeline"]
    texts = series.fillna("").astype(str)
    out = pd.DataFrame({"prediction": pipe.predict(texts), "confidence": pipe.predict_proba(texts).max(axis=1)}, index=series.index)
    return out


# ── Resumen con Ollama (opcional) ────────────────────────────────────────────
def summarize_with_llm(series: pd.Series, question: str = "What are the main themes and complaints?", n: int = 40,
                       model: str | None = None) -> str:
    """Pide al modelo local un resumen de una muestra de textos. Nada sale del equipo."""
    docs = _docs(series)
    sample = docs.sample(min(n, len(docs)), random_state=0).str.slice(0, 400).tolist()
    body = "\n".join(f"- {t}" for t in sample)
    prompt = (f"You analyze customer or survey comments. Answer in the language of the comments, with a short Markdown "
              f"list. Question: {question}\n\nComments (a random sample of {len(sample)} out of {len(docs)}):\n{body}")
    try:
        r = requests.post(ollama_generate_url(), json={"model": model or ollama_model(), "prompt": prompt, "stream": False},
                          timeout=ollama_timeout())
        r.raise_for_status()
        return r.json().get("response", "").strip() or "The model returned an empty answer."
    except requests.exceptions.Timeout:
        raise ValueError("The local model took too long. Try a lighter model or fewer texts.") from None
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Could not reach the local model ({e}). Is Ollama running?") from None


def association_with_numeric(series: pd.Series, values: pd.Series, n: int = 12) -> pd.DataFrame:
    """Términos cuya presencia se asocia con valores más altos o bajos de una columna numérica (Spearman)."""
    data = pd.concat([series, values], axis=1).dropna()
    data = data[data.iloc[:, 0].astype(str).str.strip() != ""]
    if len(data) < 30:
        raise ValueError("At least 30 rows with text and a value are needed.")
    language = detect_language(data.iloc[:, 0])
    vec = CountVectorizer(stop_words=_stop(language), token_pattern=_TOKEN.pattern, min_df=5, binary=True)
    try:
        X = vec.fit_transform(data.iloc[:, 0].astype(str))
    except ValueError:
        raise ValueError("Not enough repeated terms.") from None
    y = data.iloc[:, 1].astype(float).to_numpy()
    rows = []
    for j, term in enumerate(vec.get_feature_names_out()):
        present = X[:, j].toarray().ravel() > 0
        if 3 <= present.sum() <= len(y) - 3:
            rho, p = stats.spearmanr(present, y)
            rows.append({"term": term, "texts": int(present.sum()), "mean with": float(y[present].mean()),
                         "mean without": float(y[~present].mean()), "rho": float(rho), "p": float(p)})
    out = pd.DataFrame(rows, columns=["term", "texts", "mean with", "mean without", "rho", "p"])
    return out.reindex(out["rho"].abs().sort_values(ascending=False).index).head(n).reset_index(drop=True)
