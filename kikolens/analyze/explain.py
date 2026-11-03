"""Explicabilidad avanzada de modelos entrenados con el laboratorio: valores de Shapley por muestreo, análisis de
errores por segmento, umbrales de decisión con coste y comparación fila a fila de dos modelos."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import f1_score, r2_score

from kikolens.analyze.modellab import candidates, make_pipeline

MAX_FEATURES_SHAP = 40
BINS = 4


# ── Shapley por muestreo ─────────────────────────────────────────────────────
def _score(pipe, frame: pd.DataFrame, task: str, target_class) -> np.ndarray:
    if task == "classification" and hasattr(pipe, "predict_proba"):
        classes = list(pipe.classes_)
        return pipe.predict_proba(frame)[:, classes.index(target_class)]
    return np.asarray(pipe.predict(frame), dtype=float)


def shapley_row(pipe, X_background: pd.DataFrame, row: pd.DataFrame, task: str, n_permutations: int = 40,
                seed: int = 0, target_class=None) -> dict:
    """Contribución de cada variable a la predicción de una fila (Shapley por permutaciones). La suma de las
    contribuciones más el valor base es la predicción (en probabilidad de `target_class` si es clasificación)."""
    columns = list(X_background.columns)
    if len(columns) > MAX_FEATURES_SHAP:
        raise ValueError(f"Shapley values are limited to {MAX_FEATURES_SHAP} features; leave some out first.")
    if len(row) != 1:
        raise ValueError("Pass exactly one row.")
    if task == "classification" and target_class is None:
        target_class = pipe.classes_[-1]
    rng = np.random.default_rng(seed)
    n_perm = int(max(5, min(n_permutations, 300)))
    background = X_background.sample(n_perm, replace=len(X_background) < n_perm, random_state=seed).reset_index(drop=True)
    row = row[columns].reset_index(drop=True)

    frames, orders = [], []
    for i in range(n_perm):
        order = rng.permutation(len(columns))
        current = background.iloc[[i]].copy().reset_index(drop=True)
        frames.append(current.copy())
        for j in order:
            current[columns[j]] = row[columns[j]].iloc[0]
            frames.append(current.copy())
        orders.append(order)
    scores = _score(pipe, pd.concat(frames, ignore_index=True).astype(X_background.dtypes.to_dict(), errors="ignore"),
                    task, target_class).reshape(n_perm, len(columns) + 1)
    contrib = np.zeros(len(columns))
    for i, order in enumerate(orders):
        deltas = np.diff(scores[i])
        contrib[order] += deltas
    contrib /= n_perm
    table = pd.DataFrame({"feature": columns, "value": [row[c].iloc[0] for c in columns], "shapley": contrib})
    table = table.sort_values("shapley", key=abs, ascending=False, ignore_index=True)
    return {"table": table, "base": float(scores[:, 0].mean()), "prediction": float(scores[:, -1].mean()),
            "target_class": target_class}


def shapley_global(pipe, X_train: pd.DataFrame, X_test: pd.DataFrame, task: str, n_rows: int = 30,
                   n_permutations: int = 20, seed: int = 0, target_class=None) -> pd.DataFrame:
    """Importancia global: media del valor absoluto de Shapley sobre una muestra de filas de prueba."""
    sample = X_test.sample(min(n_rows, len(X_test)), random_state=seed)
    totals = []
    for i in range(len(sample)):
        res = shapley_row(pipe, X_train, sample.iloc[[i]], task, n_permutations, seed + i, target_class)
        totals.append(res["table"].set_index("feature")["shapley"])
    frame = pd.concat(totals, axis=1)
    out = pd.DataFrame({"feature": frame.index, "mean |shapley|": frame.abs().mean(axis=1).to_numpy(),
                        "mean shapley": frame.mean(axis=1).to_numpy()})
    return out.sort_values("mean |shapley|", ascending=False, ignore_index=True)


def shapley_figure(result: dict, top: int = 12) -> go.Figure:
    t = result["table"].head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=t["shapley"], y=t["feature"], orientation="h",
                           marker_color=["#2f5d62" if v >= 0 else "#9a3412" for v in t["shapley"]],
                           text=[f"{v}" if not isinstance(v, float) else f"{v:.4g}" for v in t["value"]],
                           hovertext=t["value"].astype(str)))
    fig.update_layout(title=f"Starting from {result['base']:.4g}, each feature moves the prediction to {result['prediction']:.4g}",
                      xaxis_title="Contribution to the prediction")
    return fig


# ── Análisis de errores ──────────────────────────────────────────────────────
def _errors(pipe, X: pd.DataFrame, y: pd.Series, task: str) -> pd.Series:
    pred = pipe.predict(X)
    if task == "regression":
        return pd.Series(np.abs(np.asarray(y, dtype=float) - pred), index=X.index)
    return pd.Series((np.asarray(y) != pred).astype(float), index=X.index)


def _segments(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) and series.nunique() > 8:
        try:
            return pd.qcut(series, BINS, duplicates="drop").astype(str)
        except ValueError:
            return None
    if series.nunique() <= 20:
        return series.astype(str)
    return None


def error_analysis(pipe, X_test: pd.DataFrame, y_test: pd.Series, task: str, min_rows: int = 8, top: int = 12) -> dict:
    """Dónde falla el modelo: segmentos (de una variable) con un error claramente mayor que el global."""
    if len(X_test) < 2 * min_rows:
        raise ValueError(f"At least {2 * min_rows} test rows are needed to analyze errors.")
    err = _errors(pipe, X_test, y_test, task)
    overall, spread = float(err.mean()), float(err.std() or 1e-12)
    rows = []
    for col in X_test.columns:
        seg = _segments(X_test[col])
        if seg is None:
            continue
        for value, idx in seg.groupby(seg).groups.items():
            n = len(idx)
            if n < min_rows:
                continue
            mean = float(err.loc[idx].mean())
            z = (mean - overall) / (spread / np.sqrt(n))
            if mean > overall * 1.15 and z >= 2:
                rows.append({"feature": col, "segment": value, "rows": n, "error": mean, "overall": overall,
                             "times worse": mean / overall if overall else float("inf"), "z": float(z)})
    segments = pd.DataFrame(rows, columns=["feature", "segment", "rows", "error", "overall", "times worse", "z"])
    segments = segments.sort_values("z", ascending=False, ignore_index=True).head(top)
    worst = X_test.assign(actual=np.asarray(y_test), predicted=pipe.predict(X_test), error=err.to_numpy())
    worst = worst.sort_values("error", ascending=False).head(15)
    metric = "mean absolute error" if task == "regression" else "error rate"
    return {"metric": metric, "overall": overall, "segments": segments, "worst": worst,
            "summary": (f"Overall {metric}: {overall:.4g}. " + (f"{len(segments)} segment(s) are clearly worse, the first being "
                        f"{segments.loc[0, 'feature']} = {segments.loc[0, 'segment']} "
                        f"({segments.loc[0, 'times worse']:.1f} times the average)." if len(segments)
                        else "No segment is clearly worse than the rest."))}


# ── Umbral de decisión ───────────────────────────────────────────────────────
def threshold_analysis(pipe, X: pd.DataFrame, y: pd.Series, cost_fp: float = 1.0, cost_fn: float = 1.0, positive=None) -> dict:
    """Para clasificación binaria: qué pasa con precisión, recall y coste según el umbral de probabilidad."""
    if not hasattr(pipe, "predict_proba") or len(pipe.classes_) != 2:
        raise ValueError("Threshold analysis needs a binary classifier that gives probabilities.")
    if cost_fp < 0 or cost_fn < 0:
        raise ValueError("Costs cannot be negative.")
    classes = list(pipe.classes_)
    positive = classes[1] if positive is None else positive
    if positive not in classes:
        raise ValueError(f"'{positive}' is not one of the classes {classes}.")
    prob = pipe.predict_proba(X)[:, classes.index(positive)]
    truth = (np.asarray(y) == positive)
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = prob >= t
        tp, fp = int((pred & truth).sum()), int((pred & ~truth).sum())
        fn, tn = int((~pred & truth).sum()), int((~pred & ~truth).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        rows.append({"threshold": t, "precision": precision, "recall": recall, "f1": f1, "accuracy": (tp + tn) / len(truth),
                     "false positives": fp, "false negatives": fn, "cost": fp * cost_fp + fn * cost_fn})
    table = pd.DataFrame(rows)
    best_cost = table.loc[table["cost"].idxmin()]
    best_f1 = table.loc[table["f1"].idxmax()]
    default = table.loc[(table["threshold"] - 0.5).abs().idxmin()]
    return {"table": table, "positive": positive, "best_cost": best_cost, "best_f1": best_f1, "default": default,
            "summary": (f"Treating '{positive}' as the positive class, the cheapest threshold is {best_cost['threshold']:.2f} "
                        f"(cost {best_cost['cost']:.4g} against {default['cost']:.4g} at 0.50); the best F1 is at "
                        f"{best_f1['threshold']:.2f}.")}


def threshold_figure(result: dict) -> go.Figure:
    t = result["table"]
    fig = go.Figure()
    for col in ("precision", "recall", "f1"):
        fig.add_trace(go.Scatter(x=t["threshold"], y=t[col], mode="lines", name=col))
    fig.add_trace(go.Scatter(x=t["threshold"], y=t["cost"], mode="lines", name="cost", yaxis="y2", line=dict(dash="dot")))
    fig.add_vline(x=float(result["best_cost"]["threshold"]), line_dash="dash", annotation_text="cheapest")
    fig.update_layout(title="What each threshold gives", xaxis_title="Probability threshold", yaxis_title="Score",
                      yaxis2=dict(overlaying="y", side="right", title="Cost"))
    return fig


# ── Comparar dos modelos ─────────────────────────────────────────────────────
def compare_pair(X_train, y_train, X_test, y_test, task: str, name_a: str, name_b: str, seed: int = 42) -> dict:
    """Entrena dos modelos del laboratorio con los mismos datos y los compara en las filas de prueba."""
    pool = candidates(task, seed)
    for name in (name_a, name_b):
        if name not in pool:
            raise ValueError(f"Unknown model '{name}'. Choose from {list(pool)}.")
    if name_a == name_b:
        raise ValueError("Choose two different models.")
    fitted = {n: clone(make_pipeline(X_train, pool[n][0])).fit(X_train, y_train) for n in (name_a, name_b)}
    pa, pb = (fitted[n].predict(X_test) for n in (name_a, name_b))
    y = np.asarray(y_test)
    if task == "regression":
        ea, eb = np.abs(y - pa), np.abs(y - pb)
        score = {name_a: r2_score(y, pa), name_b: r2_score(y, pb)}
        metric = "R2"
        diff = ea - eb
        p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
        agreement = float(np.corrcoef(pa, pb)[0, 1])
        gap = np.abs(pa - pb)
        extra = f"Predictions correlate at {agreement:.3f}."
    else:
        ea, eb = (pa != y).astype(float), (pb != y).astype(float)
        score = {name_a: f1_score(y, pa, average="weighted"), name_b: f1_score(y, pb, average="weighted")}
        metric = "F1 (weighted)"
        only_a, only_b = int(((pa == y) & (pb != y)).sum()), int(((pb == y) & (pa != y)).sum())
        p = float(stats.binomtest(only_a, only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0
        agreement = float((pa == pb).mean())
        gap = (pa != pb).astype(float)
        extra = f"They agree on {agreement:.1%} of the rows; {only_a} rows only {name_a} gets right and {only_b} only {name_b}."
    wins_a, wins_b = int((ea < eb).sum()), int((eb < ea).sum())
    better = name_a if score[name_a] > score[name_b] else name_b
    verdict = (f"{better} scores higher ({metric} {score[name_a]:.4g} against {score[name_b]:.4g}), "
               + ("and the difference on these rows is statistically reliable. " if p < 0.05 else
                  "but the difference on these rows is not statistically reliable, so treat them as equivalent. ") + extra)
    table = X_test.assign(actual=y, **{name_a: pa, name_b: pb}, gap=gap)
    return {"scores": score, "metric": metric, "p_value": p, "wins": {name_a: wins_a, name_b: wins_b},
            "agreement": agreement, "verdict": verdict, "disagreements": table.sort_values("gap", ascending=False).head(15),
            "models": fitted, "predictions": {name_a: pa, name_b: pb}}
