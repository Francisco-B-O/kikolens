"""Recomendador por filtrado colaborativo entre artículos: dos artículos se parecen si los compran o puntúan las mismas personas
(similitud del coseno con una contracción que resta peso a los pares con pocas personas en común). Se evalúa de forma honesta,
ocultando una interacción de cada persona y viendo si el recomendador la devuelve entre las primeras, frente a recomendar solo lo más popular."""
import numpy as np
import pandas as pd
from scipy import sparse

MIN_INTERACTIONS = 100


class Recommender:
    def __init__(self, users, items, matrix: sparse.csr_matrix, similarity: np.ndarray, popularity: np.ndarray):
        self.users, self.items = list(users), list(items)
        self.user_index = {u: i for i, u in enumerate(self.users)}
        self.item_index = {it: i for i, it in enumerate(self.items)}
        self.matrix, self.similarity, self.popularity = matrix, similarity, popularity

    def recommend(self, user, n: int = 10, exclude_seen: bool = True) -> pd.DataFrame:
        """Los `n` artículos con mayor puntuación para `user`: suma de la similitud con lo que ya ha usado."""
        if user not in self.user_index:
            return self.popular(n)
        row = self.matrix[self.user_index[user]]
        scores = np.asarray(row @ self.similarity).ravel()
        if exclude_seen:
            scores[row.indices] = -np.inf
        top = np.argsort(-scores)[:n]
        top = [t for t in top if np.isfinite(scores[t])]
        return pd.DataFrame({"item": [self.items[t] for t in top], "score": [float(scores[t]) for t in top]})

    def popular(self, n: int = 10) -> pd.DataFrame:
        top = np.argsort(-self.popularity)[:n]
        return pd.DataFrame({"item": [self.items[t] for t in top], "score": [float(self.popularity[t]) for t in top]})

    def similar_items(self, item, n: int = 10) -> pd.DataFrame:
        if item not in self.item_index:
            raise ValueError(f"'{item}' is not one of the items.")
        i = self.item_index[item]
        scores = self.similarity[i].copy()
        scores[i] = -np.inf
        top = np.argsort(-scores)[:n]
        return pd.DataFrame({"item": [self.items[t] for t in top], "similarity": [float(scores[t]) for t in top],
                             "people in common": [int(self.matrix[:, i].multiply(self.matrix[:, t]).nnz) for t in top]})


def _matrix(df: pd.DataFrame, user: str, item: str, rating: str | None) -> tuple:
    for c in (user, item) + ((rating,) if rating else ()):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    data = df[[user, item] + ([rating] if rating else [])].dropna()
    data[user], data[item] = data[user].astype(str), data[item].astype(str)
    if rating:
        if not pd.api.types.is_numeric_dtype(data[rating]):
            raise ValueError(f"'{rating}' must be numeric.")
        data = data.groupby([user, item], as_index=False)[rating].mean()
        weight = data[rating].astype(float).clip(lower=0).to_numpy()
    else:
        data = data.drop_duplicates([user, item])
        weight = np.ones(len(data))
    if len(data) < MIN_INTERACTIONS:
        raise ValueError(f"At least {MIN_INTERACTIONS} interactions are needed; there are {len(data)}.")
    users, u = np.unique(data[user], return_inverse=True)
    items, i = np.unique(data[item], return_inverse=True)
    if len(users) < 10 or len(items) < 5:
        raise ValueError("At least 10 people and 5 items are needed.")
    return users, items, sparse.csr_matrix((weight, (u, i)), shape=(len(users), len(items))), data, u, i


def fit(df: pd.DataFrame, user: str, item: str, rating: str | None = None, shrinkage: float = 10.0) -> Recommender:
    """Ajusta el recomendador. `rating` es opcional: sin él, cada interacción pesa igual (compra, clic...)."""
    users, items, matrix, *_ = _matrix(df, user, item, rating)
    binary = matrix.copy()
    binary.data = np.ones_like(binary.data)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel()) + 1e-12
    dots = (matrix.T @ matrix).toarray()
    common = (binary.T @ binary).toarray()
    similarity = dots / np.outer(norms, norms) * (common / (common + shrinkage))
    np.fill_diagonal(similarity, 0.0)
    popularity = np.asarray(binary.sum(axis=0)).ravel().astype(float)
    return Recommender(users, items, matrix.tocsr(), similarity, popularity)


def evaluate(df: pd.DataFrame, user: str, item: str, rating: str | None = None, k: int = 10, seed: int = 0, shrinkage: float = 10.0) -> dict:
    """Oculta una interacción de cada persona con al menos 3 y mide con qué frecuencia aparece entre las `k` primeras recomendaciones,
    frente a recomendar lo más popular."""
    users, items, matrix, data, u, i = _matrix(df, user, item, rating)
    rng = np.random.default_rng(seed)
    counts = np.bincount(u, minlength=len(users))
    eligible = np.where(counts >= 3)[0]
    if len(eligible) < 10:
        raise ValueError("At least 10 people with 3 or more interactions are needed to evaluate.")
    hide = {}
    for person in eligible:
        positions = np.where(u == person)[0]
        hide[person] = int(rng.choice(positions))
    keep = np.ones(len(u), bool)
    keep[list(hide.values())] = False
    train = data[keep].reset_index(drop=True)
    model = fit(train, user, item, rating, shrinkage)
    hits = pop_hits = 0
    for person, pos in hide.items():
        target = data[item].iloc[pos]
        name = users[person]
        recs = model.recommend(name, k)["item"].tolist()
        popular = model.recommend("__nobody__", k + int(counts[person]))["item"].tolist()
        seen = set(train.loc[train[user] == name, item])
        popular = [p for p in popular if p not in seen][:k]
        hits += target in recs
        pop_hits += target in popular
    n = len(hide)
    result = {"hit_rate": hits / n, "popular_hit_rate": pop_hits / n, "people": n, "k": k}
    lift = result["hit_rate"] / result["popular_hit_rate"] if result["popular_hit_rate"] else np.inf
    result["summary"] = (f"Hiding one item per person, the recommender brings it back in its top {k} {result['hit_rate']:.1%} of the time, "
                         f"against {result['popular_hit_rate']:.1%} for recommending the most popular items ({lift:.1f} times better).")
    return result
