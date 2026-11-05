"""Búsqueda de variables nuevas: prueba productos, cocientes, sumas, diferencias, logaritmos y cuadrados de las columnas
numéricas y conserva solo las que mejoran el modelo en validación cruzada. Cada una sale como una fórmula que se aplica
con la operación `add_column`, así queda en la receta y se puede repetir con datos nuevos."""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from kikolens.analyze.modellab import make_pipeline, prepare_xy, _splitter
from kikolens.analyze.transform import evaluate_expression, add_column

MAX_BASE = 8
MAX_SCREENED = 24


def _tick(name: str) -> str:
    return f"`{name}`"


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_")[:40]


def candidates(X: pd.DataFrame, y_numeric: pd.Series | None = None, base: list | None = None) -> list:
    """Las fórmulas candidatas como (nombre, expresión, tipo)."""
    nums = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
            and X[c].nunique() > 5]
    if base is None:
        base = nums
    base = [c for c in base if c in nums][:MAX_BASE]
    out = []
    for c in base:
        s = X[c].dropna()
        if len(s) and s.min() >= 0 and abs(s.skew()) > 1:
            out.append((f"log_{_slug(c)}", f"log1p({_tick(c)})", "log"))
        out.append((f"{_slug(c)}_squared", f"{_tick(c)} ** 2", "square"))
    for i, a in enumerate(base):
        for b in base[i + 1:]:
            ta, tb = _tick(a), _tick(b)
            out += [(f"{_slug(a)}_x_{_slug(b)}", f"{ta} * {tb}", "product"),
                    (f"{_slug(a)}_per_{_slug(b)}", f"{ta} / {tb}", "ratio"),
                    (f"{_slug(b)}_per_{_slug(a)}", f"{tb} / {ta}", "ratio"),
                    (f"{_slug(a)}_plus_{_slug(b)}", f"{ta} + {tb}", "sum"),
                    (f"{_slug(a)}_minus_{_slug(b)}", f"{ta} - {tb}", "difference")]
    return out


def _model(task: str):
    return (HistGradientBoostingRegressor(max_iter=60, random_state=0) if task == "regression"
            else HistGradientBoostingClassifier(max_iter=60, random_state=0))


def _cv(X, y, task, folds, seed) -> float:
    cv = _splitter(task, y, folds, seed)
    scoring = "r2" if task == "regression" else "f1_weighted"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.mean(cross_val_score(make_pipeline(X, _model(task)), X, y, cv=cv, scoring=scoring, error_score=np.nan)))


def suggest_features(df: pd.DataFrame, target: str, drop=(), max_new: int = 5, folds: int = 3, min_gain: float = 0.01,
                     seed: int = 42, progress=None) -> dict:
    """Devuelve `suggestions` (lista con nombre, fórmula, tipo y mejora acumulada), el resultado antes y después."""
    if max_new < 1:
        raise ValueError("max_new must be at least 1.")
    X, y, info = prepare_xy(df, target, drop=drop)
    task = info["task"]
    if len(X) > 4000:
        sample = X.sample(4000, random_state=seed).index
        X, y = X.loc[sample], y.loc[sample]
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    pool = candidates(X)
    if not pool:
        raise ValueError("There are no numeric columns to combine (at least two with more than 5 distinct values).")

    # cribado: se conservan las candidatas más relacionadas con el objetivo
    y_num = pd.Series(pd.factorize(y)[0] if task == "classification" else y.to_numpy(float))
    scored = []
    for name, expr, kind in pool:
        try:
            values = evaluate_expression(X, expr)
        except ValueError:
            continue
        if values.nunique() < 5 or values.isna().mean() > 0.2:
            continue
        r = abs(values.rank().corr(y_num.rank())) if task == "regression" or y_num.nunique() > 1 else 0
        scored.append((0.0 if pd.isna(r) else r, name, expr, kind, values))
    scored.sort(key=lambda t: -t[0])
    scored = scored[:MAX_SCREENED]
    if not scored:
        raise ValueError("No candidate feature could be computed from these columns.")

    baseline = _cv(X, y, task, folds, seed)
    current, base_frame, chosen = baseline, X.copy(), []
    remaining = list(scored)
    for step in range(max_new):
        best = None
        for i, (_, name, expr, kind, values) in enumerate(remaining):
            if progress:
                progress((step + i / max(len(remaining), 1)) / max_new, name)
            trial = base_frame.assign(**{name: values.to_numpy()})
            score = _cv(trial, y, task, folds, seed)
            if best is None or score > best[0]:
                best = (score, i)
        if best is None or best[0] - current < min_gain:
            break
        score, i = best
        _, name, expr, kind, values = remaining.pop(i)
        base_frame[name] = values.to_numpy()
        chosen.append({"name": name, "expression": expr, "kind": kind, "score_after": score, "gain": score - current})
        current = score
        if not remaining:
            break
    metric = "R2" if task == "regression" else "F1"
    summary = (f"{len(chosen)} new column(s) raise the cross-validated {metric} from {baseline:.3f} to {current:.3f}."
               if chosen else f"None of the {len(scored)} ideas improved the {metric} ({baseline:.3f}) by at least {min_gain}.")
    return {"suggestions": chosen, "baseline": baseline, "final": current, "metric": metric, "task": task,
            "tested": len(scored), "summary": summary}


def apply_suggestions(df: pd.DataFrame, suggestions: list):
    """Añade las columnas elegidas. Devuelve (nuevo_df, pasos) con pasos (operación, parámetros, mensaje) para la receta."""
    steps = []
    for s in suggestions:
        df, message = add_column(df, s["name"], s["expression"])
        steps.append(("add_column", {"name": s["name"], "expression": s["expression"]}, message))
    return df, steps
