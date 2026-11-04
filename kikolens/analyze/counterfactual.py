"""Contrafactuales: ¿qué es lo mínimo que habría que cambiar en una fila para que el modelo diera otro resultado?
Busca cambios de una y de dos variables sobre valores que existen en los datos de entrenamiento, y ordena por
número de cambios y por lo lejos que van. Sirve para explicar una decisión y para ver qué palancas existen."""
from itertools import combinations

import numpy as np
import pandas as pd

from kikolens.utils.dtypes import categorical_columns

GRID = 15


def _values(series: pd.Series, grid: int) -> list:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        q = series.dropna().quantile(np.linspace(0, 1, grid)).unique()
        return [float(v) for v in q]
    return sorted(series.dropna().astype(str).unique())[:30]


def _score(pipe, frame: pd.DataFrame, task: str, desired):
    """Cuánto se acerca cada fila a lo deseado: probabilidad de la clase, o valor predicho (regresión)."""
    if task == "regression":
        return np.asarray(pipe.predict(frame), float)
    classes = list(pipe.classes_)
    return pipe.predict_proba(frame)[:, classes.index(desired)]


def find(pipe, X_reference: pd.DataFrame, row: pd.DataFrame, task: str, desired=None, target_range=None,
         fixed=(), max_changes: int = 2, top: int = 5, grid: int = GRID) -> dict:
    """`desired`: la clase que se quiere obtener. En regresión, `target_range=(min, max)` o, si se omite el máximo, un
    mínimo. `fixed`: variables que no se pueden cambiar (edad, sexo...)."""
    if len(row) != 1:
        raise ValueError("Pass exactly one row.")
    if task == "classification":
        if desired is None or str(desired) not in [str(c) for c in pipe.classes_]:
            raise ValueError("Say which class you would like to obtain.")
        desired = next(c for c in pipe.classes_ if str(c) == str(desired))
    elif not target_range or len(target_range) != 2 or (target_range[0] is None and target_range[1] is None):
        raise ValueError("Give a range for the value you would like, such as (100, None) for at least 100.")
    if max_changes not in (1, 2):
        raise ValueError("max_changes must be 1 or 2.")

    row = row.reset_index(drop=True)
    unknown = [c for c in fixed if c not in row.columns]
    if unknown:
        raise ValueError(f"Unknown columns to keep fixed: {unknown}.")
    movable = [c for c in row.columns if c not in fixed]
    cats = set(categorical_columns(X_reference))
    options = {c: [v for v in _values(X_reference[c], grid) if str(v) != str(row.at[0, c])] for c in movable}

    def success(frame):
        if task == "classification":
            return np.asarray(pipe.predict(frame)).astype(str) == str(desired)
        pred = _score(pipe, frame, task, None)
        lo, hi = target_range
        return (pred >= (lo if lo is not None else -np.inf)) & (pred <= (hi if hi is not None else np.inf))

    current = float(_score(pipe, row, task, desired)[0])
    if bool(success(row)[0]):
        return {"already": True, "current": current, "found": pd.DataFrame(),
                "summary": "The model already gives the result you want for this row."}

    def build(changes: dict) -> pd.DataFrame:
        frame = pd.concat([row] * len(next(iter(changes.values()))), ignore_index=True)
        for col, vals in changes.items():
            frame[col] = vals
        return frame

    found = []
    for col in movable:
        vals = options[col]
        if not vals:
            continue
        frame = build({col: vals})
        ok = success(frame)
        score = _score(pipe, frame, task, desired)
        for v, good, s in zip(vals, ok, score):
            if good:
                found.append({"changes": {col: v}, "score": float(s)})
    if max_changes == 2:
        singles_done = {next(iter(f["changes"])) for f in found}
        influential = sorted(movable, key=lambda c: -_influence(pipe, row, options[c], c, task, desired))[:6]
        for a, b in combinations([c for c in influential if c not in singles_done], 2):
            va, vb = options[a], options[b]
            if not va or not vb:
                continue
            pairs = [(x, y) for x in va for y in vb]
            frame = build({a: [p[0] for p in pairs], b: [p[1] for p in pairs]})
            ok, score = success(frame), _score(pipe, frame, task, desired)
            for (x, y), good, s in zip(pairs, ok, score):
                if good:
                    found.append({"changes": {a: x, b: y}, "score": float(s)})

    spans = {c: float(X_reference[c].std() or 1.0) for c in movable
             if pd.api.types.is_numeric_dtype(X_reference[c]) and c not in cats}

    def distance(changes: dict) -> float:
        total = 0.0
        for col, v in changes.items():
            total += abs(float(v) - float(row.at[0, col])) / spans[col] if col in spans else 1.0
        return total

    for f in found:
        f["distance"] = distance(f["changes"])
    found.sort(key=lambda f: (len(f["changes"]), f["distance"]))
    best, seen = [], set()
    for f in found:
        key = tuple(sorted(f["changes"]))
        if key in seen and len(seen) >= top:
            continue
        seen.add(key)
        best.append(f)
        if len(best) >= top:
            break
    rows = [{"change": " and ".join(f"{c}: {_show(row.at[0, c])} -> {_show(v)}" for c, v in f["changes"].items()),
             "variables": len(f["changes"]), "effort": round(f["distance"], 2), "result": round(f["score"], 3)} for f in best]
    summary = (f"{len(found)} way(s) found; the smallest is " + rows[0]["change"] + "." if rows else
               "No change of up to " + str(max_changes) + " variable(s) reaches that result. Try letting more variables move.")
    return {"already": False, "current": current, "found": pd.DataFrame(rows), "summary": summary}


def _influence(pipe, row, values, col, task, desired) -> float:
    if not values:
        return 0.0
    frame = pd.concat([row] * len(values), ignore_index=True)
    frame[col] = values
    scores = _score(pipe, frame, task, desired)
    return float(np.nanmax(scores) - np.nanmin(scores))


def _show(value) -> str:
    return f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)
