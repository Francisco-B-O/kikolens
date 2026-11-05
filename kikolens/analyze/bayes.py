"""A/B bayesiano: en vez de un p-valor, la probabilidad de que una variante sea mejor y cuánto se pierde si se elige mal."""
import numpy as np
import pandas as pd
from scipy import stats


def _rng(seed):
    return np.random.default_rng(seed)


def _report(a: np.ndarray, b: np.ndarray, labels, coverage: float, kind: str) -> dict:
    diff = b - a
    lift = diff / np.where(a == 0, np.nan, a)
    lo, hi = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    prob = float(np.mean(diff > 0))
    loss_b = float(np.mean(np.maximum(-diff, 0)))     # lo que se pierde de media al elegir B si A era mejor
    loss_a = float(np.mean(np.maximum(diff, 0)))
    interval = tuple(float(v) for v in np.quantile(diff, [lo, hi]))
    lift_interval = tuple(float(v) for v in np.nanquantile(lift, [lo, hi]))
    leader = labels[1] if prob >= 0.5 else labels[0]
    certainty = max(prob, 1 - prob)
    return {"probability_b_better": prob, "expected_loss_b": loss_b, "expected_loss_a": loss_a,
            "difference": float(diff.mean()), "difference_interval": interval, "lift": float(np.nanmean(lift)),
            "lift_interval": lift_interval, "leader": leader, "kind": kind,
            "summary": (f"{labels[1]} is better than {labels[0]} with probability {prob:.1%}. "
                        f"The most likely difference is {diff.mean():+.4g} ({int(coverage * 100)}% range "
                        f"{interval[0]:+.4g} to {interval[1]:+.4g}). Choosing {labels[1]} wrongly would cost "
                        f"{loss_b:.4g} on average; choosing {labels[0]} wrongly, {loss_a:.4g}. "
                        + ("The evidence favours " + leader + (" strongly." if certainty >= 0.95 else ", but not yet convincingly.")))}


def bayesian_rates(successes_a: int, total_a: int, successes_b: int, total_b: int, labels=("A", "B"), prior=(1.0, 1.0),
                   coverage: float = 0.95, draws: int = 200_000, seed: int = 0) -> dict:
    """Tasas de conversión con prior Beta (por defecto uniforme)."""
    for s, n in ((successes_a, total_a), (successes_b, total_b)):
        if n <= 0 or s < 0 or s > n:
            raise ValueError("Successes must be between 0 and the total, and totals must be positive.")
    if min(prior) <= 0:
        raise ValueError("The prior parameters must be positive.")
    rng = _rng(seed)
    a = rng.beta(prior[0] + successes_a, prior[1] + total_a - successes_a, draws)
    b = rng.beta(prior[0] + successes_b, prior[1] + total_b - successes_b, draws)
    out = _report(a, b, labels, coverage, "rate")
    out["rates"] = {labels[0]: successes_a / total_a, labels[1]: successes_b / total_b}
    return out


def bayesian_means(values_a, values_b, labels=("A", "B"), coverage: float = 0.95, draws: int = 200_000, seed: int = 0) -> dict:
    """Medias de una variable numérica (ingresos, tiempos...) con la distribución t de la media de cada grupo."""
    a_vals = pd.to_numeric(pd.Series(values_a), errors="coerce").dropna().to_numpy(float)
    b_vals = pd.to_numeric(pd.Series(values_b), errors="coerce").dropna().to_numpy(float)
    if len(a_vals) < 5 or len(b_vals) < 5:
        raise ValueError("Each group needs at least 5 numeric values.")
    rng = _rng(seed)

    def posterior(v):
        return v.mean() + stats.t.rvs(len(v) - 1, size=draws, random_state=rng) * v.std(ddof=1) / np.sqrt(len(v))

    out = _report(posterior(a_vals), posterior(b_vals), labels, coverage, "mean")
    out["means"] = {labels[0]: float(a_vals.mean()), labels[1]: float(b_vals.mean())}
    return out


def bayesian_ab(df: pd.DataFrame, group: str, outcome: str, control, variant=None, success=None, **kwargs) -> dict:
    """Atajo sobre una tabla: `outcome` binario (con `success` como valor de éxito) o numérico."""
    for col in (group, outcome):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    data = df[[group, outcome]].dropna()
    data[group] = data[group].astype(str)
    levels = sorted(data[group].unique())
    control = str(control)
    if control not in levels:
        raise ValueError(f"'{control}' is not a value of '{group}'.")
    others = [v for v in levels if v != control]
    if variant is None:
        if len(others) != 1:
            raise ValueError("Say which variant to compare with the control.")
        variant = others[0]
    variant = str(variant)
    if variant not in levels:
        raise ValueError(f"'{variant}' is not a value of '{group}'.")
    a, b = data[data[group] == control][outcome], data[data[group] == variant][outcome]
    labels = (control, variant)
    if success is not None or not pd.api.types.is_numeric_dtype(data[outcome]) or set(data[outcome].unique()) <= {0, 1, True, False}:
        mark = success if success is not None else 1
        ya, yb = (a.astype(str) == str(mark)), (b.astype(str) == str(mark))
        if success is None and pd.api.types.is_numeric_dtype(data[outcome]):
            ya, yb = a == 1, b == 1
        return bayesian_rates(int(ya.sum()), len(a), int(yb.sum()), len(b), labels=labels, **kwargs)
    return bayesian_means(a, b, labels=labels, **kwargs)
