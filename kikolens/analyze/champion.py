"""Campeón y aspirante: compara varios modelos guardados con los mismos datos de los que ya se conoce el resultado y dice cuál
acierta más y si la diferencia es fiable o puede ser casualidad."""
import numpy as np
import pandas as pd
from scipy import stats

from kikolens.analyze.perfwatch import _metrics
from kikolens.analyze.predict import predict_frame


def compare_saved_models(bundles: dict, df: pd.DataFrame, apply_recipe: bool = False) -> dict:
    """`bundles`: nombre -> paquete de modelo. Todos deben predecir la misma columna, que debe estar en `df`."""
    if len(bundles) < 2:
        raise ValueError("Choose at least two models to compare.")
    targets = {b["target"] for b in bundles.values()}
    if len(targets) != 1:
        raise ValueError(f"These models predict different columns ({', '.join(sorted(targets))}); only models for the same target can be compared.")
    target = targets.pop()
    tasks = {b["task"] for b in bundles.values()}
    if len(tasks) != 1:
        raise ValueError("The models mix regression and classification.")
    task = tasks.pop()
    if target not in df.columns:
        raise ValueError(f"The data needs the real values in a column named '{target}'.")
    data = df.dropna(subset=[target]).reset_index(drop=True)
    if len(data) < 30:
        raise ValueError("At least 30 rows with a known result are needed.")
    actual, predicted, rows = data[target], {}, []
    for name, bundle in bundles.items():
        scored, _ = predict_frame(bundle, data, apply_recipe=apply_recipe, interval=False)
        predicted[name] = scored[f"predicted_{target}"]
        rows.append({"model": name, "algorithm": bundle.get("label", ""), **_metrics(task, actual, predicted[name])})
    key = "MAE" if task == "regression" else "F1"
    board = pd.DataFrame(rows).sort_values(key, ascending=task == "regression").reset_index(drop=True)
    board.insert(0, "rank", range(1, len(board) + 1))
    first, second = board["model"].iloc[0], board["model"].iloc[1]
    p, how = _paired(task, actual, predicted[first], predicted[second])
    reliable = p < 0.05
    gap = abs(board[key].iloc[0] - board[key].iloc[1])
    return {"table": board, "task": task, "target": target, "metric": key, "champion": first, "challenger": second,
            "p_value": p, "reliable": reliable, "test": how,
            "summary": (f"{first} comes first on {key} ({board[key].iloc[0]:.4g} against {board[key].iloc[1]:.4g} for {second}, a gap of {gap:.3g}). "
                        + (f"The difference is reliable ({how}, p = {p:.3g})." if reliable else
                           f"But the gap could easily be chance ({how}, p = {p:.3g}): on these rows they are not clearly different."))}


def _paired(task: str, actual: pd.Series, a: pd.Series, b: pd.Series) -> tuple:
    if task == "regression":
        ea = np.abs(actual.to_numpy(float) - a.to_numpy(float))
        eb = np.abs(actual.to_numpy(float) - b.to_numpy(float))
        if np.allclose(ea, eb):
            return 1.0, "Wilcoxon test of the absolute errors"
        return float(stats.wilcoxon(ea, eb).pvalue), "Wilcoxon test of the absolute errors"
    right_a = (actual.astype(str) == a.astype(str)).to_numpy()
    right_b = (actual.astype(str) == b.astype(str)).to_numpy()
    only_a, only_b = int((right_a & ~right_b).sum()), int((~right_a & right_b).sum())
    if only_a + only_b == 0:
        return 1.0, "exact test on the rows where only one is right"
    return float(stats.binomtest(only_a, only_a + only_b, 0.5).pvalue), "exact test on the rows where only one is right"
