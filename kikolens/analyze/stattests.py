"""Pruebas estadísticas explicadas en lenguaje llano: comparar grupos, asociaciones, correlaciones, pruebas A/B y
cribado de columnas relacionadas con un objetivo. Cada prueba elige el método adecuado según los datos, devuelve el
tamaño del efecto (no solo el p-valor) y una conclusión que dice qué se hizo y qué no se puede concluir."""
import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

MAX_GROUPS = 20
MAX_CATEGORIES = 50
LOW_CARDINALITY = 10  # un número con tan pocos valores distintos se trata como categoría al cribar


def _numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _label(value: float, cuts, names=("negligible", "small", "medium", "large")) -> str:
    for cut, name in zip(cuts, names):
        if value < cut:
            return name
    return names[-1]


def _p_text(p: float) -> str:
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"


def benjamini_hochberg(p_values) -> np.ndarray:
    """Corrige la tasa de falsos descubrimientos cuando se hacen muchas pruebas a la vez."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out


# ── Comparar grupos ──────────────────────────────────────────────────────────
def _is_normal(sample: pd.Series) -> bool:
    if len(sample) >= 30:  # con muestras grandes el teorema central del límite protege a las pruebas paramétricas
        return True
    if len(sample) < 3 or sample.nunique() < 3:
        return False
    return stats.shapiro(sample.sample(min(len(sample), 500), random_state=0)).pvalue > 0.05


def compare_groups(df: pd.DataFrame, value: str, group: str, alpha: float = 0.05) -> dict:
    """¿Difiere una variable numérica entre grupos? Elige t de Welch, Mann-Whitney, ANOVA o Kruskal-Wallis."""
    for col in (value, group):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    if not _numeric(df[value]):
        raise ValueError(f"'{value}' must be numeric to compare it between groups.")
    data = df[[value, group]].dropna()
    data[group] = data[group].astype(str)
    sizes = data[group].value_counts()
    usable = sizes[sizes >= 2].index.tolist()
    if len(usable) < 2:
        raise ValueError("At least two groups with two or more rows each are needed.")
    if len(usable) > MAX_GROUPS:
        raise ValueError(f"'{group}' has {len(usable)} groups; choose a column with at most {MAX_GROUPS}.")
    data = data[data[group].isin(usable)]
    samples = {g: data.loc[data[group] == g, value].astype(float) for g in usable}
    arrays = list(samples.values())

    normal = all(_is_normal(s) for s in arrays)
    equal_var = stats.levene(*arrays).pvalue > 0.05 if min(len(a) for a in arrays) > 2 else False
    n_total, k = len(data), len(arrays)
    post_hoc = None

    if k == 2:
        a, b = arrays
        if normal:
            res = stats.ttest_ind(a, b, equal_var=False)
            pooled = math.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
            effect = abs(a.mean() - b.mean()) / pooled if pooled else 0.0
            test, effect_name = "Welch's t-test", "Cohen's d"
            size = _label(effect, (0.2, 0.5, 0.8))
        else:
            res = stats.mannwhitneyu(a, b, alternative="two-sided")
            effect = abs(1 - 2 * res.statistic / (len(a) * len(b)))
            test, effect_name = "Mann-Whitney U test", "rank-biserial r"
            size = _label(effect, (0.1, 0.3, 0.5))
    elif normal and equal_var:
        res = stats.f_oneway(*arrays)
        grand = data[value].mean()
        ss_between = sum(len(a) * (a.mean() - grand) ** 2 for a in arrays)
        effect = ss_between / ((data[value] - grand) ** 2).sum()
        test, effect_name = "One-way ANOVA", "eta squared"
        size = _label(effect, (0.01, 0.06, 0.14))
        if res.pvalue < alpha:
            tukey = stats.tukey_hsd(*arrays)
            post_hoc = pd.DataFrame([{"group A": usable[i], "group B": usable[j], "difference": arrays[i].mean() - arrays[j].mean(),
                                      "p (Tukey)": float(tukey.pvalue[i, j])}
                                     for i, j in itertools.combinations(range(k), 2)])
    else:
        res = stats.kruskal(*arrays)
        effect = max(0.0, (res.statistic - k + 1) / (n_total - k))
        test, effect_name = "Kruskal-Wallis test", "epsilon squared"
        size = _label(effect, (0.01, 0.08, 0.26))
        if res.pvalue < alpha:
            pairs = list(itertools.combinations(range(k), 2))
            raw = [stats.mannwhitneyu(arrays[i], arrays[j], alternative="two-sided").pvalue for i, j in pairs]
            post_hoc = pd.DataFrame([{"group A": usable[i], "group B": usable[j],
                                      "difference": arrays[i].median() - arrays[j].median(), "p (Holm-adjusted)": p}
                                     for (i, j), p in zip(pairs, _holm(raw))])

    p = float(res.pvalue)
    summary = pd.DataFrame({"group": usable, "rows": [len(a) for a in arrays], "mean": [a.mean() for a in arrays],
                            "median": [a.median() for a in arrays], "std": [a.std() for a in arrays]})
    significant = p < alpha
    why = ("The data are roughly normal" if normal else "The data are not clearly normal") + \
        (" and the spread is similar" if k > 2 and normal and equal_var else "")
    verdict = (f"{test} ({why.lower()}): {_p_text(p)}, {effect_name} = {effect:.2f} ({size} effect). "
               + (f"'{value}' differs between the groups of '{group}'." if significant else
                  f"No reliable difference in '{value}' between the groups of '{group}' was found; this does not prove there is none."))
    if significant and size == "negligible":
        verdict += " The difference is statistically detectable but too small to matter in practice."
    return {"test": test, "p_value": p, "significant": significant, "effect": float(effect), "effect_name": effect_name,
            "effect_size": size, "summary": summary, "post_hoc": post_hoc, "normal": normal, "equal_variance": equal_var,
            "conclusion": verdict, "alpha": alpha, "groups": usable, "samples": samples}


def _holm(p_values) -> list:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    n = len(p)
    adjusted = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()


# ── Asociación entre categorías ──────────────────────────────────────────────
def association(df: pd.DataFrame, a: str, b: str, alpha: float = 0.05) -> dict:
    """¿Están relacionadas dos variables categóricas? Chi-cuadrado (o Fisher en tablas 2x2 pequeñas) y V de Cramér."""
    for col in (a, b):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    if a == b:
        raise ValueError("Choose two different columns.")
    data = df[[a, b]].dropna().astype(str)
    table = pd.crosstab(data[a], data[b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Each column needs at least two different values.")
    if max(table.shape) > MAX_CATEGORIES:
        raise ValueError(f"Too many categories (limit {MAX_CATEGORIES} per column).")
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=table.shape == (2, 2))
    small = float((expected < 5).mean())
    test = "Chi-squared test"
    if table.shape == (2, 2) and expected.min() < 5:
        p, test = float(stats.fisher_exact(table.to_numpy())[1]), "Fisher's exact test"
    n = int(table.to_numpy().sum())
    v = math.sqrt(chi2 / (n * (min(table.shape) - 1))) if n else 0.0
    size = _label(v, (0.1, 0.3, 0.5))
    residuals = pd.DataFrame((table.to_numpy() - expected) / np.sqrt(np.maximum(expected, 1e-12)),
                             index=table.index, columns=table.columns)
    significant = p < alpha
    verdict = (f"{test}: {_p_text(p)}, Cramér's V = {v:.2f} ({size} association). "
               + (f"'{a}' and '{b}' are related." if significant else
                  f"No reliable relationship between '{a}' and '{b}' was found."))
    if small > 0.2 and test == "Chi-squared test":
        verdict += f" {small:.0%} of the cells have fewer than 5 expected rows, so the result is approximate."
    return {"test": test, "p_value": float(p), "significant": significant, "cramers_v": float(v), "effect_size": size,
            "table": table, "expected": pd.DataFrame(expected, index=table.index, columns=table.columns),
            "residuals": residuals, "conclusion": verdict, "alpha": alpha, "n": n}


# ── Correlación ──────────────────────────────────────────────────────────────
def correlation_test(df: pd.DataFrame, x: str, y: str, alpha: float = 0.05) -> dict:
    for col in (x, y):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
        if not _numeric(df[col]):
            raise ValueError(f"'{col}' must be numeric.")
    if x == y:
        raise ValueError("Choose two different columns.")
    data = df[[x, y]].dropna()
    n = len(data)
    if n < 5:
        raise ValueError("At least 5 rows with both values are needed.")
    if data[x].nunique() < 2 or data[y].nunique() < 2:
        raise ValueError("Both columns need some variation.")
    r, p = stats.pearsonr(data[x], data[y])
    rho, p_rho = stats.spearmanr(data[x], data[y])
    z, se = math.atanh(min(max(r, -0.999999), 0.999999)), 1 / math.sqrt(max(n - 3, 1))
    ci = (math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se))
    size = _label(abs(r), (0.1, 0.3, 0.5))
    direction = "positive" if r > 0 else "negative"
    verdict = (f"Pearson r = {r:.3f} (95% CI {ci[0]:.2f} to {ci[1]:.2f}), {_p_text(p)}: a {size} {direction} linear relationship. "
               if p < alpha else f"Pearson r = {r:.3f}, {_p_text(p)}: no reliable linear relationship. ")
    if abs(rho) - abs(r) > 0.15:
        verdict += f"Spearman rho = {rho:.2f} is stronger, so the relationship is curved or driven by outliers."
    elif abs(r) - abs(rho) > 0.15:
        verdict += f"Spearman rho = {rho:.2f} is weaker: a few extreme points may be creating the correlation."
    return {"pearson_r": float(r), "pearson_p": float(p), "spearman_rho": float(rho), "spearman_p": float(p_rho),
            "ci": ci, "n": n, "effect_size": size, "significant": p < alpha, "conclusion": verdict,
            "data": data}


# ── Pruebas A/B ──────────────────────────────────────────────────────────────
def sample_size_proportions(baseline: float, minimum_effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Filas necesarias por grupo para detectar un cambio absoluto `minimum_effect` sobre una tasa `baseline`."""
    if not 0 < baseline < 1:
        raise ValueError("The baseline rate must be between 0 and 1.")
    p2 = baseline + minimum_effect
    if not 0 < p2 < 1 or minimum_effect == 0:
        raise ValueError("The change to detect must be non-zero and keep the rate between 0 and 1.")
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pooled = (baseline + p2) / 2
    n = ((za * math.sqrt(2 * pooled * (1 - pooled)) + zb * math.sqrt(baseline * (1 - baseline) + p2 * (1 - p2))) ** 2
         / (p2 - baseline) ** 2)
    return int(math.ceil(n))


def sample_size_means(std: float, minimum_effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    if std <= 0 or minimum_effect == 0:
        raise ValueError("The standard deviation and the change to detect must be non-zero.")
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(math.ceil(2 * ((za + zb) * std / minimum_effect) ** 2))


def ab_test(df: pd.DataFrame, group: str, outcome: str, control, success=None, alpha: float = 0.05, seed: int = 0) -> dict:
    """Compara un grupo de control con otro de tratamiento. Si `outcome` es binario compara tasas (z de dos
    proporciones y probabilidad bayesiana de que el tratamiento gane); si es numérico compara medias (t de Welch)."""
    for col in (group, outcome):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    data = df[[group, outcome]].dropna()
    data[group] = data[group].astype(str)
    control = str(control)
    levels = data[group].unique().tolist()
    if control not in levels:
        raise ValueError(f"'{control}' is not a value of '{group}'.")
    others = [g for g in levels if g != control]
    if len(others) != 1:
        raise ValueError(f"A/B tests need exactly two groups; '{group}' has {len(levels)}. Filter the data first.")
    treatment = others[0]
    a, b = data[data[group] == control][outcome], data[data[group] == treatment][outcome]
    if min(len(a), len(b)) < 2:
        raise ValueError("Each group needs at least two rows.")

    binary = data[outcome].nunique() == 2
    if binary:
        values = sorted(data[outcome].astype(str).unique())
        win = str(success) if success is not None else values[-1]
        if win not in values:
            raise ValueError(f"'{win}' is not a value of '{outcome}'.")
        xa, xb = int((a.astype(str) == win).sum()), int((b.astype(str) == win).sum())
        na, nb = len(a), len(b)
        pa, pb = xa / na, xb / nb
        pooled = (xa + xb) / (na + nb)
        se0 = math.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
        z = (pb - pa) / se0 if se0 else 0.0
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
        se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
        diff = pb - pa
        ci = (diff - 1.96 * se, diff + 1.96 * se)
        rng = np.random.default_rng(seed)
        draws_a, draws_b = rng.beta(1 + xa, 1 + na - xa, 20000), rng.beta(1 + xb, 1 + nb - xb, 20000)
        prob_better = float((draws_b > draws_a).mean())
        lift = diff / pa if pa else float("nan")
        kind, measure = "proportion", f"rate of '{win}'"
        stat_a, stat_b = pa, pb
        extra = {"successes": (xa, xb), "prob_treatment_better": prob_better, "success_value": win}
        needed = None
        try:
            needed = sample_size_proportions(pa, diff) if 0 < pa < 1 and 0 < pb < 1 and diff else None
        except ValueError:
            needed = None
    else:
        if not _numeric(data[outcome]):
            raise ValueError(f"'{outcome}' must be numeric or have exactly two values.")
        res = stats.ttest_ind(b.astype(float), a.astype(float), equal_var=False)
        diff = float(b.mean() - a.mean())
        se = math.sqrt(a.var() / len(a) + b.var() / len(b))
        ci = (diff - 1.96 * se, diff + 1.96 * se)
        p = float(res.pvalue)
        lift = diff / a.mean() if a.mean() else float("nan")
        kind, measure = "mean", f"mean of '{outcome}'"
        stat_a, stat_b = float(a.mean()), float(b.mean())
        pooled_std = math.sqrt((a.var() + b.var()) / 2)
        extra = {"std": pooled_std}
        needed = sample_size_means(pooled_std, diff) if diff and pooled_std else None
        na, nb = len(a), len(b)

    significant = p < alpha
    direction = "higher" if diff > 0 else "lower"
    verdict = (f"The {measure} is {stat_b:.4g} for '{treatment}' against {stat_a:.4g} for '{control}' "
               f"({diff:+.4g}, {lift:+.1%} relative; 95% CI {ci[0]:+.4g} to {ci[1]:+.4g}), {_p_text(p)}. "
               + (f"'{treatment}' is reliably {direction}." if significant else "The difference is not statistically reliable."))
    if kind == "proportion":
        verdict += f" Probability that '{treatment}' is better: {extra['prob_treatment_better']:.1%}."
    if not significant and needed and needed > min(na, nb):
        verdict += f" You would need about {needed:,} rows per group to confirm a gap this size (you have {min(na, nb):,})."
    return {"kind": kind, "control": control, "treatment": treatment, "n": (na, nb), "values": (stat_a, stat_b),
            "difference": float(diff), "relative_lift": float(lift), "ci": ci, "p_value": p, "significant": significant,
            "conclusion": verdict, "needed_per_group": needed, "alpha": alpha, **extra}


# ── Cribado ──────────────────────────────────────────────────────────────────
def _as_category(s: pd.Series) -> bool:
    return not _numeric(s) or s.nunique(dropna=True) <= LOW_CARDINALITY


def _pair_test(df: pd.DataFrame, col: str, target: str):
    """(prueba, estadístico, p, efecto 0-1) entre una columna y el objetivo, según el tipo de cada una."""
    data = df[[col, target]].dropna()
    if len(data) < 8 or data[col].nunique() < 2 or data[target].nunique() < 2:
        return None
    cat_col, cat_target = _as_category(data[col]), _as_category(data[target])
    if not cat_col and not cat_target:
        rho, p = stats.spearmanr(data[col], data[target])
        return "Spearman correlation", float(rho), float(p), abs(float(rho))
    if cat_col and cat_target:
        table = pd.crosstab(data[col].astype(str), data[target].astype(str))
        if max(table.shape) > MAX_CATEGORIES:
            return None
        chi2, p, _, _ = stats.chi2_contingency(table)
        n = table.to_numpy().sum()
        return "Chi-squared test", float(chi2), float(p), math.sqrt(chi2 / (n * (min(table.shape) - 1)))
    numeric, categories = (data[target], data[col]) if cat_col else (data[col], data[target])
    groups = [g.astype(float).to_numpy() for _, g in numeric.groupby(categories.astype(str)) if len(g) >= 2]
    if len(groups) < 2 or len(groups) > MAX_CATEGORIES:
        return None
    h, p = stats.kruskal(*groups)
    return "Kruskal-Wallis test", float(h), float(p), max(0.0, (h - len(groups) + 1) / (len(data) - len(groups)))


def screen_associations(df: pd.DataFrame, target: str, alpha: float = 0.05, max_columns: int = 80) -> pd.DataFrame:
    """Prueba cada columna contra el objetivo y corrige por pruebas múltiples (Benjamini-Hochberg)."""
    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found.")
    if df[target].notna().sum() < 8:
        raise ValueError("The target needs at least 8 known values.")
    rows = []
    for col in [c for c in df.columns if c != target][:max_columns]:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        try:
            result = _pair_test(df, col, target)
        except ValueError:
            result = None
        if result:
            rows.append({"column": col, "test": result[0], "statistic": result[1], "p_value": result[2], "strength": result[3]})
    if not rows:
        raise ValueError("No column could be tested against the target.")
    out = pd.DataFrame(rows)
    out["p_adjusted"] = benjamini_hochberg(out["p_value"])
    out["related"] = out["p_adjusted"] < alpha
    return out.sort_values(["p_adjusted", "strength"], ascending=[True, False], ignore_index=True)
