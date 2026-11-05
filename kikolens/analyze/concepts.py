"""Laboratorio de conceptos: simulaciones que enseñan por qué la estadística funciona como funciona.

Cada concepto simula miles de casos con una semilla fija, devuelve las cifras, un gráfico, una conclusión escrita con esos números y una sugerencia de qué cambiar
para aprender más. No usa tus datos: sirve para entender los resultados que sí salen de ellos.

  clt                  el teorema central del límite: las medias se vuelven una campana aunque los datos no lo sean
  p_values             qué es un p-valor, cuántos «hallazgos» salen por azar y cómo mirar antes de tiempo los infla
  confidence_intervals qué significa «95 % de confianza»
  power                cuánta gente hace falta para ver un efecto
  overfitting          sesgo, varianza y por qué un modelo más complejo puede predecir peor
  simpson              la paradoja de Simpson: la tendencia global contradice la de cada grupo
  regression_to_mean   por qué «el mejor del año pasado» suele bajar
  multiple_testing     probar muchas cosas siempre encuentra algo, y cómo corregirlo"""
import numpy as np
import plotly.graph_objects as go
from scipy import stats

from kikolens.analyze.stattests import benjamini_hochberg


# ── Piezas comunes ───────────────────────────────────────────────────────────
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _ttest_p(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """p-valores de Student de dos muestras (varianzas iguales), vectorizado sobre el último eje."""
    na, nb = a.shape[-1], b.shape[-1]
    va, vb = a.var(axis=-1, ddof=1), b.var(axis=-1, ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    t = (b.mean(axis=-1) - a.mean(axis=-1)) / np.sqrt(pooled * (1 / na + 1 / nb))
    return 2 * stats.t.sf(np.abs(t), na + nb - 2)


def _result(title: str, summary: str, lesson: str, try_this: str, figure, table=None, notes=None, **extra) -> dict:
    return {"title": title, "summary": summary, "lesson": lesson, "try": try_this, "figure": figure, "table": table, "notes": notes or [], **extra}


# ── 1. Teorema central del límite ────────────────────────────────────────────
POPULATIONS = {
    "uniform": ("Uniform (flat)", lambda rng, size: rng.uniform(0, 1, size)),
    "exponential": ("Exponential (skewed)", lambda rng, size: rng.exponential(1.0, size)),
    "lognormal": ("Log-normal (very skewed)", lambda rng, size: rng.lognormal(0, 1.0, size)),
    "bimodal": ("Two humps", lambda rng, size: np.where(rng.random(size) < 0.5, rng.normal(-2, 0.5, size), rng.normal(2, 0.5, size))),
    "coin": ("Coin with 10% heads", lambda rng, size: (rng.random(size) < 0.1).astype(float)),
    "heavy": ("Heavy tails (t with 3 degrees of freedom)", lambda rng, size: rng.standard_t(3, size)),
}


def clt(distribution: str = "exponential", n: int = 30, reps: int = 5000, seed: int = 0) -> dict:
    """Toma `reps` muestras de tamaño `n` de una población que no es una campana y mira cómo se distribuyen sus medias."""
    if distribution not in POPULATIONS:
        raise ValueError(f"distribution must be one of {', '.join(POPULATIONS)}.")
    if not 1 <= n <= 5000 or not 200 <= reps <= 50000:
        raise ValueError("n must be between 1 and 5000 and reps between 200 and 50,000.")
    label, draw = POPULATIONS[distribution]
    rng = _rng(seed)
    population = draw(rng, 200_000)
    mu, sigma = float(population.mean()), float(population.std())
    means = draw(rng, (reps, n)).mean(axis=1)
    predicted = sigma / np.sqrt(n)
    skew_pop, skew_means = float(stats.skew(population)), float(stats.skew(means))
    ks = float(stats.kstest((means - means.mean()) / means.std(), "norm").statistic)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=means, histnorm="probability density", nbinsx=60, name=f"Means of {n} values"))
    grid = np.linspace(means.min(), means.max(), 200)
    fig.add_trace(go.Scatter(x=grid, y=stats.norm.pdf(grid, mu, predicted), name="Bell curve predicted by theory", line=dict(width=3)))
    fig.update_layout(title=f"Averages of {n} values from: {label}", xaxis_title="Average", yaxis_title="Density")
    pop_fig = go.Figure(go.Histogram(x=population[:20000], histnorm="probability density", nbinsx=60, name="One value"))
    pop_fig.update_layout(title=f"The population itself: {label}", xaxis_title="Value", showlegend=False)
    close = ks < 0.03
    notes = []
    if distribution == "heavy" and n < 100:
        notes.append("With heavy tails the bell takes much longer to appear: a few extreme values still move the average. Try n = 500.")
    if n == 1:
        notes.append("With n = 1 each 'average' is just one value, so you are looking at the population again.")
    return _result(
        "The central limit theorem",
        (f"One value from '{label}' is far from a bell (skewness {skew_pop:+.2f}), but the average of {n} values has skewness {skew_means:+.2f} and spread {means.std():.4f}, "
         f"almost exactly the {predicted:.4f} that theory predicts (population spread {sigma:.3f} divided by the square root of {n}). "
         + ("The averages already look like a bell." if close else "The averages are not yet a clean bell: more values per average would fix that.")),
        "Averages behave more regularly than the data they come from: they cluster around the true mean with a spread that shrinks as 1 over the square root of n. "
        "That is why so many tests can assume a bell shape even when the data are not one, as long as the samples are not tiny.",
        "Change the population to 'Log-normal' or 'Heavy tails' and watch how many values per average it takes; then set n to 1, 5, 30, 300.",
        fig, table=None, notes=notes, population_figure=pop_fig, population_mean=mu, population_sd=sigma, predicted_sd=float(predicted), observed_sd=float(means.std()),
        skew_population=skew_pop, skew_means=skew_means, normality_gap=ks)


# ── 2. p-valores ─────────────────────────────────────────────────────────────
def p_values(effect: float = 0.0, n: int = 30, experiments: int = 4000, alpha: float = 0.05, looks: int = 1, seed: int = 0) -> dict:
    """Repite `experiments` experimentos con dos grupos de `n` personas. `effect` es la diferencia real en desviaciones típicas; `looks` es cuántas veces se mira
    el resultado por el camino (se para en cuanto sale significativo)."""
    if not 4 <= n <= 2000 or not 200 <= experiments <= 20000 or not 0 < alpha < 0.5 or not 1 <= looks <= 20 or not -3 <= effect <= 3:
        raise ValueError("n must be 4 to 2000, experiments 200 to 20,000, alpha between 0 and 0.5, looks 1 to 20 and effect between -3 and 3.")
    rng = _rng(seed)
    a = rng.normal(0, 1, (experiments, n))
    b = rng.normal(effect, 1, (experiments, n))
    sizes = sorted({max(4, int(round(n * (k + 1) / looks))) for k in range(looks)})
    significant = np.zeros(experiments, bool)
    first_p = None
    for m in sizes:
        p = _ttest_p(a[:, :m], b[:, :m])
        if first_p is None:
            first_p = p
        significant |= p < alpha
    final_p = _ttest_p(a, b)
    rate = float(significant.mean())
    fig = go.Figure(go.Histogram(x=final_p, xbins=dict(start=0, end=1, size=0.05), histnorm="probability", name="p-values"))
    fig.add_vline(x=alpha, line_dash="dash", annotation_text=f"alpha = {alpha:g}")
    fig.update_layout(title=f"p-values of {experiments:,} experiments (true effect = {effect:g} standard deviations)", xaxis_title="p-value", yaxis_title="Share of experiments", yaxis_tickformat=".0%")
    notes = []
    if effect == 0:
        summary = (f"There is no real effect in any of the {experiments:,} experiments, yet {rate:.1%} of them came out 'significant'. "
                   + (f"That is the {alpha:.0%} the test promises: the p-values are spread evenly between 0 and 1, so one in {round(1 / alpha)} experiments looks like a discovery by pure chance."
                      if looks == 1 else f"Looking {looks} times and stopping at the first success pushed the false alarms from {alpha:.0%} to {rate:.1%}."))
        lesson = ("A p-value is the probability of seeing a difference at least this big if there were truly nothing going on. When nothing is going on, p-values are uniformly spread, "
                  "so a significant result is not proof of an effect, it is a surprise worth checking. Checking the result repeatedly and stopping when it looks good breaks that promise.")
    else:
        summary = (f"The real effect is {effect:g} standard deviations. With {n} per group the test detects it in {rate:.1%} of experiments (its power); in the other "
                   f"{1 - rate:.1%} a real effect goes unnoticed. The p-values pile up near 0 instead of spreading evenly.")
        lesson = ("A non-significant result does not mean 'no effect': with too few people a real effect is missed most of the time. "
                  "Power is the chance of seeing an effect that is really there, and it depends on the size of the effect, the sample and the noise.")
        if rate < 0.5:
            notes.append(f"This experiment would miss the effect more often than not (power {rate:.0%}). Try a larger n.")
    return _result("What a p-value is", summary, lesson,
                   "Set the effect to 0 and looks to 1, then to 10 and see how peeking inflates the false alarms; then set the effect to 0.5 and change n.",
                   fig, notes=notes, significant_rate=rate, alpha=alpha, looks=looks, first_look_rate=float((first_p < alpha).mean()), p_values=final_p)


# ── 3. Intervalos de confianza ───────────────────────────────────────────────
def confidence_intervals(reps: int = 100, n: int = 30, level: float = 0.95, population: str = "uniform", seed: int = 0) -> dict:
    """Calcula `reps` intervalos de confianza para la media, cada uno con una muestra nueva, y cuenta cuántos contienen el valor verdadero."""
    if not 10 <= reps <= 1000 or not 2 <= n <= 5000 or not 0.5 <= level < 1:
        raise ValueError("reps must be 10 to 1000, n 2 to 5000 and level between 0.5 and 1.")
    if population not in POPULATIONS:
        raise ValueError(f"population must be one of {', '.join(POPULATIONS)}.")
    rng = _rng(seed)
    _, draw = POPULATIONS[population]
    truth = float(draw(_rng(seed + 1), 400_000).mean())
    data = draw(rng, (reps, n))
    means, sems = data.mean(axis=1), data.std(axis=1, ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(0.5 + level / 2, n - 1)
    low, high = means - crit * sems, means + crit * sems
    hit = (low <= truth) & (truth <= high)
    # cobertura estable con muchas réplicas, para mostrar que el nivel se cumple (o no) a la larga
    many = draw(_rng(seed + 2), (4000, n))
    m2, s2 = many.mean(axis=1), many.std(axis=1, ddof=1) / np.sqrt(n)
    long_run = float(((m2 - crit * s2 <= truth) & (truth <= m2 + crit * s2)).mean())
    fig = go.Figure()
    order = np.arange(reps)
    for ok, color, name in ((True, "#2F6FB3", "Contains the true mean"), (False, "#c0392b", "Misses the true mean")):
        idx = order[hit == ok]
        fig.add_trace(go.Scatter(x=means[idx], y=idx, mode="markers", name=name, marker=dict(color=color, size=5),
                                 error_x=dict(type="data", symmetric=False, array=(high - means)[idx], arrayminus=(means - low)[idx], color=color, thickness=1)))
    fig.add_vline(x=truth, line_dash="dash", annotation_text="true mean")
    fig.update_layout(title=f"{reps} intervals of {level:.0%} confidence, each from its own sample of {n}", xaxis_title="Value", yaxis_title="Sample number", yaxis=dict(autorange="reversed"))
    misses = int((~hit).sum())
    notes = []
    if long_run < level - 0.02:
        notes.append(f"With samples of {n} from this population the intervals cover the truth only {long_run:.1%} of the time over thousands of repetitions, below the {level:.0%} promised: "
                     "the data are too skewed for so few values. A larger sample restores it.")
    return _result(
        "What 'confidence' means",
        f"Of {reps} intervals, {int(hit.sum())} contain the true mean ({hit.mean():.0%}) and {misses} miss it. Over 4,000 repetitions the share is {long_run:.1%}, "
        f"against the {level:.0%} the method promises.",
        "'95% confidence' is a statement about the method, not about one interval: if you repeated the study many times, about 95% of the intervals built this way would contain the truth. "
        "Any single interval either contains it or does not; you just do not know which.",
        "Raise the level to 99% (wider intervals, fewer misses), reduce n to 5 with a skewed population, and see the coverage fall.",
        fig, notes=notes, covered=int(hit.sum()), missed=misses, long_run_coverage=long_run, true_value=truth)


# ── 4. Potencia ──────────────────────────────────────────────────────────────
def _power(effect: float, n, alpha: float) -> np.ndarray:
    n = np.asarray(n, float)
    dof = 2 * n - 2
    crit = stats.t.ppf(1 - alpha / 2, dof)
    nc = effect * np.sqrt(n / 2)
    return 1 - stats.nct.cdf(crit, dof, nc) + stats.nct.cdf(-crit, dof, nc)


def power(effect: float = 0.5, alpha: float = 0.05, target: float = 0.8, simulate_n: int = 30, seed: int = 0) -> dict:
    """Probabilidad de detectar un efecto de `effect` desviaciones típicas según los participantes por grupo; cuántos hacen falta para llegar a `target`;
    y una comprobación por simulación para `simulate_n`."""
    if not 0.05 <= effect <= 3 or not 0.001 <= alpha <= 0.3 or not 0.5 <= target < 0.999 or not 4 <= simulate_n <= 2000:
        raise ValueError("effect must be 0.05 to 3, alpha 0.001 to 0.3, target 0.5 to 0.999 and simulate_n 4 to 2000.")
    grid = np.unique(np.round(np.geomspace(3, 5000, 120)).astype(int))
    curve = _power(effect, grid, alpha)
    reached = grid[curve >= target]
    needed = int(reached[0]) if len(reached) else None
    rng = _rng(seed)
    runs = 4000
    simulated = float((_ttest_p(rng.normal(0, 1, (runs, simulate_n)), rng.normal(effect, 1, (runs, simulate_n))) < alpha).mean())
    theory = float(_power(effect, simulate_n, alpha))
    fig = go.Figure(go.Scatter(x=grid, y=curve, mode="lines", name="Power"))
    fig.add_hline(y=target, line_dash="dot", annotation_text=f"target {target:.0%}")
    if needed:
        fig.add_vline(x=needed, line_dash="dash", annotation_text=f"n = {needed}")
    fig.update_layout(title=f"Chance of detecting an effect of {effect:g} standard deviations", xaxis_title="Participants per group", yaxis_title="Power", xaxis_type="log", yaxis_tickformat=".0%")
    half = int(_ceil_n(effect * 2, alpha, target))
    return _result(
        "How many people are enough",
        (f"To detect an effect of {effect:g} standard deviations with {target:.0%} probability you need about {needed if needed else 'more than 5,000'} participants per group. "
         f"With {simulate_n} per group the power is {theory:.1%} by formula and {simulated:.1%} by simulation of {runs:,} experiments. "
         f"An effect twice as big ({effect * 2:g}) would need only about {half}."),
        "Power rises with the sample, with the size of the effect and with a looser threshold, and it falls with noise. Needing four times the people to detect an effect half as big is the rule "
        "that explains why small effects are so expensive to study.",
        "Halve the effect and watch the required sample quadruple; then lower alpha to 0.01.",
        fig, table=None, needed_per_group=needed, power_at_n=theory, simulated_power=simulated, effect=effect)


def _ceil_n(effect: float, alpha: float, target: float) -> int:
    grid = np.arange(3, 100000)
    curve = _power(effect, grid, alpha)
    return int(grid[np.argmax(curve >= target)]) if (curve >= target).any() else 100000


# ── 5. Sobreajuste ───────────────────────────────────────────────────────────
def overfitting(max_degree: int = 12, n: int = 30, noise: float = 0.3, reps: int = 200, seed: int = 0) -> dict:
    """Ajusta polinomios de grado creciente a datos de `sin(2 pi x)` con ruido, muchas veces, y mide el error con los datos de ajuste y con datos nuevos,
    descompuesto en sesgo al cuadrado, varianza y ruido."""
    if not 2 <= max_degree <= 20 or not 8 <= n <= 500 or not 0.01 <= noise <= 2 or not 20 <= reps <= 1000:
        raise ValueError("max_degree must be 2 to 20, n 8 to 500, noise 0.01 to 2 and reps 20 to 1000.")
    if max_degree > n - 2:
        raise ValueError("max_degree must be at most n - 2.")
    rng = _rng(seed)
    f = lambda x: np.sin(2 * np.pi * x)                                         # noqa: E731
    grid = np.linspace(0, 1, 100)
    rows = []
    preds_by_degree = {}
    for d in range(1, max_degree + 1):
        preds, train, test = [], [], []
        for _ in range(reps):
            x = rng.random(n)
            y = f(x) + rng.normal(0, noise, n)
            p = np.polynomial.Polynomial.fit(x, y, d)
            train.append(np.mean((p(x) - y) ** 2))
            xt = rng.random(n)
            test.append(np.mean((p(xt) - (f(xt) + rng.normal(0, noise, n))) ** 2))
            preds.append(p(grid))
        preds = np.array(preds)
        preds_by_degree[d] = preds
        rows.append({"degree": d, "training error": float(np.mean(train)), "error on new data": float(np.mean(test)), "bias squared": float(np.mean((preds.mean(axis=0) - f(grid)) ** 2)),
                     "variance": float(np.mean(preds.var(axis=0))), "noise": noise ** 2})
    best = min(rows, key=lambda r: r["error on new data"])
    last = rows[-1]
    fig = go.Figure()
    for key, name in (("training error", "Error on the data it was fitted to"), ("error on new data", "Error on new data"), ("bias squared", "Bias squared"), ("variance", "Variance")):
        fig.add_trace(go.Scatter(x=[r["degree"] for r in rows], y=[r[key] for r in rows], mode="lines+markers", name=name))
    fig.add_vline(x=best["degree"], line_dash="dash", annotation_text="best on new data")
    fig.update_layout(title="Complexity against error", xaxis_title="Complexity of the model (polynomial degree)", yaxis_title="Mean squared error", yaxis_type="log")
    curves = go.Figure()
    for d in sorted({1, best["degree"], max_degree}):
        shown = preds_by_degree[d]
        curves.add_trace(go.Scatter(x=grid, y=shown[0], mode="lines", name=f"Degree {d}: one fit"))
        curves.add_trace(go.Scatter(x=grid, y=shown.mean(axis=0), mode="lines", line=dict(dash="dot"), name=f"Degree {d}: average of {reps} fits"))
    curves.add_trace(go.Scatter(x=grid, y=f(grid), mode="lines", line=dict(color="black", width=3), name="Truth"))
    curves.update_layout(title="What the models actually draw", xaxis_title="x", yaxis_title="y", yaxis_range=[-2.5, 2.5])
    notes = []
    if last["error on new data"] > best["error on new data"] * 1.5:
        notes.append(f"Past degree {best['degree']} the training error keeps falling while the error on new data rises {last['error on new data'] / best['error on new data']:.1f} times: that gap is overfitting.")
    return _result(
        "Bias, variance and overfitting",
        (f"Degree {best['degree']} predicts new data best (error {best['error on new data']:.3f}, of which {best['noise']:.3f} is unavoidable noise). "
         f"A degree {max_degree} curve fits its own data almost perfectly (error {last['training error']:.3f}) but errs {last['error on new data']:.3f} on new data, "
         f"with variance {last['variance']:.3f}: it memorised the noise of this sample."),
        "A simple model is stable but can be wrong in a systematic way (bias). A flexible model can follow the truth but also follows the noise (variance). "
        "The error on the data used for fitting always improves with complexity, so it cannot tell you when to stop; only data the model has not seen can.",
        "Increase the noise or reduce n and see the best degree fall; then try a very large n and see the flexible model become safe.",
        fig, table=_frame(rows), notes=notes, curves_figure=curves, best_degree=best["degree"])


def _frame(rows):
    import pandas as pd
    return pd.DataFrame(rows)


# ── 6. Paradoja de Simpson ───────────────────────────────────────────────────
def simpson(strength: float = 1.0, groups: int = 3, n: int = 150, seed: int = 0) -> dict:
    """Datos en los que dentro de cada grupo `y` sube con `x`, pero los grupos con más `x` tienen menos `y` de base, así que en conjunto parece que baja."""
    if not 0.05 <= strength <= 3 or not 2 <= groups <= 6 or not 30 <= n <= 2000:
        raise ValueError("strength must be 0.05 to 3, groups 2 to 6 and n 30 to 2000.")
    rng = _rng(seed)
    xs, ys, gs = [], [], []
    for g in range(groups):
        x = rng.normal(2 * g, 0.7, n)
        y = 10 - 3 * strength * g + 1.0 * (x - 2 * g) + rng.normal(0, 0.7, n)
        xs.append(x)
        ys.append(y)
        gs.append(np.full(n, g))
    x, y, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)
    overall = stats.linregress(x, y)
    within = [stats.linregress(x[g == k], y[g == k]) for k in range(groups)]
    fig = go.Figure()
    for k in range(groups):
        fig.add_trace(go.Scatter(x=x[g == k], y=y[g == k], mode="markers", name=f"Group {k + 1}", marker=dict(size=4, opacity=0.6)))
        line = np.array([x[g == k].min(), x[g == k].max()])
        fig.add_trace(go.Scatter(x=line, y=within[k].intercept + within[k].slope * line, mode="lines", showlegend=False, line=dict(width=3)))
    ends = np.array([x.min(), x.max()])
    fig.add_trace(go.Scatter(x=ends, y=overall.intercept + overall.slope * ends, mode="lines", name="All together", line=dict(color="black", dash="dash", width=3)))
    fig.update_layout(title="The same data, two stories", xaxis_title="x", yaxis_title="y")
    import pandas as pd
    table = pd.DataFrame([{"group": "All together", "slope": overall.slope, "correlation": overall.rvalue}] +
                         [{"group": f"Group {k + 1}", "slope": r.slope, "correlation": r.rvalue} for k, r in enumerate(within)])
    reversed_ = overall.slope < 0 < np.mean([r.slope for r in within])
    return _result(
        "Simpson's paradox",
        (f"Within every group, y goes up with x (slopes about {np.mean([r.slope for r in within]):+.2f}). Putting everybody together, y goes "
         f"{'down' if overall.slope < 0 else 'up'} with x (slope {overall.slope:+.2f}). " + ("The overall trend is the opposite of what happens inside each group." if reversed_ else
                                                                                                "Here the overall trend still agrees with the groups; raise the strength to reverse it.")),
        "A trend in the whole data can be the opposite of the trend in every part when groups differ in both variables. The data alone do not say which story is right: "
        "you need to know how the groups arose (is the group a confounder you should adjust for, or part of the effect you want to measure?).",
        "Raise the strength to reverse the trend, then ask: would you give a treatment based on the overall line or the group lines?",
        fig, table=table, overall_slope=float(overall.slope), within_slopes=[float(r.slope) for r in within], reversed=bool(reversed_))


# ── 7. Regresión a la media ──────────────────────────────────────────────────
def regression_to_mean(reliability: float = 0.5, n: int = 5000, top: float = 0.1, seed: int = 0) -> dict:
    """Cada resultado es habilidad más suerte; `reliability` es la parte de la variación que es habilidad. Mira cómo le va el año siguiente al `top` mejor."""
    if not 0 <= reliability <= 1 or not 200 <= n <= 100000 or not 0.01 <= top <= 0.5:
        raise ValueError("reliability must be 0 to 1, n 200 to 100,000 and top 0.01 to 0.5.")
    rng = _rng(seed)
    skill = rng.normal(0, np.sqrt(reliability), n)
    luck = np.sqrt(1 - reliability)
    year1, year2 = skill + rng.normal(0, luck, n), skill + rng.normal(0, luck, n)
    cut = np.quantile(year1, 1 - top)
    best = year1 >= cut
    m1, m2 = float(year1[best].mean()), float(year2[best].mean())
    expected = reliability * m1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=year1[~best][:3000], y=year2[~best][:3000], mode="markers", name="Others", marker=dict(size=3, opacity=0.4)))
    fig.add_trace(go.Scatter(x=year1[best], y=year2[best], mode="markers", name=f"Top {top:.0%} in year 1", marker=dict(size=4, color="crimson")))
    line = np.array([year1.min(), year1.max()])
    fig.add_trace(go.Scatter(x=line, y=line, mode="lines", name="Same result both years", line=dict(dash="dash", color="black")))
    fig.add_trace(go.Scatter(x=line, y=reliability * line, mode="lines", name="What actually happens on average", line=dict(color="green", width=3)))
    fig.update_layout(title="Year 1 against year 2", xaxis_title="Year 1 result", yaxis_title="Year 2 result")
    kept = m2 / m1 if m1 else float("nan")
    return _result(
        "Regression to the mean",
        (f"The top {top:.0%} in year 1 averaged {m1:+.2f}. The same people averaged {m2:+.2f} in year 2 ({kept:.0%} of their lead), close to the {expected:+.2f} expected when "
         f"{reliability:.0%} of the variation is skill and the rest luck. Nobody got worse: the luck simply did not repeat."),
        "Extreme results are a mix of skill and luck, and the luck does not come back. The more luck matters (low reliability), the more the best fall back and the worst recover. "
        "This explains the 'sophomore slump', why punishing a bad result and rewarding a good one seems to work, and why a treatment given to the worst cases seems to help even if it does nothing.",
        "Set the reliability to 1 (pure skill, nothing falls back) and to 0 (pure luck, everyone falls back to the average).",
        fig, notes=[], first_year_mean=m1, second_year_mean=m2, expected_second_year=float(expected), share_kept=float(kept))


# ── 8. Comparaciones múltiples ───────────────────────────────────────────────
def multiple_testing(tests: int = 20, real_effects: int = 0, n: int = 30, effect: float = 0.8, alpha: float = 0.05, studies: int = 1000, seed: int = 0) -> dict:
    """En cada «estudio» se hacen `tests` comparaciones, `real_effects` de ellas con un efecto verdadero. Compara qué pasa sin corregir, con Bonferroni y con Benjamini-Hochberg."""
    if not 2 <= tests <= 200 or not 0 <= real_effects <= tests or not 8 <= n <= 500 or not 0 < alpha < 0.3 or not 200 <= studies <= 5000:
        raise ValueError("tests must be 2 to 200, real_effects 0 to tests, n 8 to 500, alpha between 0 and 0.3 and studies 200 to 5000.")
    rng = _rng(seed)
    a = rng.normal(0, 1, (studies, tests, n))
    shift = np.zeros(tests)
    shift[:real_effects] = effect
    b = rng.normal(0, 1, (studies, tests, n)) + shift[None, :, None]
    p = _ttest_p(a, b)
    truth = np.zeros(tests, bool)
    truth[:real_effects] = True
    methods = {"No correction": p < alpha, "Bonferroni": p < alpha / tests, "Benjamini-Hochberg": np.array([benjamini_hochberg(row) < alpha for row in p])}
    import pandas as pd
    rows = []
    for name, found in methods.items():
        false = found & ~truth
        true = found & truth
        discoveries = found.sum(axis=1)
        rows.append({"method": name, "studies with at least one false alarm": float((false.sum(axis=1) > 0).mean()), "false alarms per study": float(false.sum(axis=1).mean()),
                     "real effects found": float(true[:, :real_effects].mean()) if real_effects else float("nan"),
                     "share of discoveries that are false": float(false.sum() / max(discoveries.sum(), 1))})
    table = pd.DataFrame(rows)
    fig = go.Figure(go.Bar(x=table["method"], y=table["studies with at least one false alarm"]))
    fig.update_layout(title=f"Share of studies with at least one false alarm ({tests} tests each)", yaxis_tickformat=".0%", yaxis_title="Studies")
    uncorrected = table.iloc[0]
    expected = 1 - (1 - alpha) ** (tests - real_effects)
    return _result(
        "Why testing many things finds something",
        (f"With {tests - real_effects} comparisons where nothing is real, {uncorrected['studies with at least one false alarm']:.0%} of studies still announce at least one 'discovery' "
         f"(theory: {expected:.0%}). Bonferroni brings that to {table.iloc[1]['studies with at least one false alarm']:.1%} and Benjamini-Hochberg to {table.iloc[2]['studies with at least one false alarm']:.1%}"
         + (f", while still finding {table.iloc[2]['real effects found']:.0%} of the real effects (Bonferroni finds {table.iloc[1]['real effects found']:.0%})." if real_effects else ".")),
        "Every test has a chance of a false alarm, and the chances add up: with 20 tests at 5%, about two out of three studies find something that is not there. "
        "Corrections trade a few missed real effects for far fewer false ones; Benjamini-Hochberg controls the share of false discoveries and is usually the better compromise.",
        "Set the number of tests to 1, then 100; then add 3 real effects and compare how many each method finds.",
        fig, table=table, notes=[], expected_any_false=float(expected))


# ── Registro ─────────────────────────────────────────────────────────────────
def _p(kind, default, low=None, high=None, label="", choices=None):
    return {"type": kind, "default": default, "min": low, "max": high, "label": label, "choices": choices}


CONCEPTS = {
    "clt": {"title": "The central limit theorem", "question": "Why do averages look like a bell even when the data do not?", "fn": clt,
            "params": {"distribution": _p("choice", "exponential", label="Shape of the data", choices=list(POPULATIONS)), "n": _p("int", 30, 1, 2000, "Values per average"),
                       "reps": _p("int", 5000, 200, 50000, "Averages to draw")}},
    "p_values": {"title": "What a p-value is", "question": "If nothing is going on, how often does a test still say something is?", "fn": p_values,
                 "params": {"effect": _p("float", 0.0, -3, 3, "Real effect (standard deviations)"), "n": _p("int", 30, 4, 2000, "People per group"),
                            "looks": _p("int", 1, 1, 20, "Times you peek at the result"), "alpha": _p("float", 0.05, 0.001, 0.4, "Threshold (alpha)"),
                            "experiments": _p("int", 4000, 200, 20000, "Experiments")}},
    "confidence_intervals": {"title": "What 'confidence' means", "question": "What does 95% confidence mean for one interval?", "fn": confidence_intervals,
                             "params": {"population": _p("choice", "uniform", label="Shape of the data", choices=list(POPULATIONS)), "n": _p("int", 30, 2, 2000, "Sample size"),
                                        "level": _p("float", 0.95, 0.5, 0.999, "Confidence level"), "reps": _p("int", 100, 10, 1000, "Intervals to draw")}},
    "power": {"title": "How many people are enough", "question": "How many people do I need to see an effect?", "fn": power,
              "params": {"effect": _p("float", 0.5, 0.05, 3, "Effect to detect (standard deviations)"), "alpha": _p("float", 0.05, 0.001, 0.3, "Threshold (alpha)"),
                         "target": _p("float", 0.8, 0.5, 0.99, "Power wanted"), "simulate_n": _p("int", 30, 4, 2000, "Check by simulation with n per group")}},
    "overfitting": {"title": "Bias, variance and overfitting", "question": "Why can a more complex model predict worse?", "fn": overfitting,
                    "params": {"max_degree": _p("int", 12, 2, 20, "Most complex model"), "n": _p("int", 30, 8, 500, "Data points"), "noise": _p("float", 0.3, 0.01, 2, "Noise"),
                               "reps": _p("int", 150, 20, 1000, "Repetitions")}},
    "simpson": {"title": "Simpson's paradox", "question": "Can a trend reverse when you split the data into groups?", "fn": simpson,
                "params": {"strength": _p("float", 1.0, 0.05, 3, "Difference between groups"), "groups": _p("int", 3, 2, 6, "Groups"), "n": _p("int", 150, 30, 2000, "Points per group")}},
    "regression_to_mean": {"title": "Regression to the mean", "question": "Why does the best of last year usually do worse this year?", "fn": regression_to_mean,
                           "params": {"reliability": _p("float", 0.5, 0.0, 1.0, "Share of the result that is skill"), "top": _p("float", 0.1, 0.01, 0.5, "Top share to follow"),
                                      "n": _p("int", 5000, 200, 100000, "People")}},
    "multiple_testing": {"title": "Why testing many things finds something", "question": "What happens if I test twenty things?", "fn": multiple_testing,
                         "params": {"tests": _p("int", 20, 2, 200, "Comparisons per study"), "real_effects": _p("int", 0, 0, 200, "Of them, real effects"),
                                    "effect": _p("float", 0.8, 0.1, 3, "Size of the real effects"), "n": _p("int", 30, 8, 500, "People per group"), "alpha": _p("float", 0.05, 0.001, 0.3, "Threshold (alpha)"),
                                    "studies": _p("int", 1000, 200, 5000, "Studies to simulate")}},
}


def run(name: str, seed: int = 0, **params) -> dict:
    """Ejecuta un concepto con parámetros comprobados contra sus límites (así un valor absurdo da un mensaje claro y no una espera larga)."""
    if name not in CONCEPTS:
        raise ValueError(f"Unknown concept '{name}'. Available: {', '.join(CONCEPTS)}.")
    spec = CONCEPTS[name]["params"]
    unknown = set(params) - set(spec)
    if unknown:
        raise ValueError(f"{name} does not take {sorted(unknown)}. It takes: {', '.join(spec)}.")
    clean = {}
    for key, value in params.items():
        s = spec[key]
        if s["type"] == "choice":
            if value not in s["choices"]:
                raise ValueError(f"'{key}' must be one of {', '.join(s['choices'])}.")
            clean[key] = value
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be a number.") from None
        if not np.isfinite(number) or not s["min"] <= number <= s["max"]:
            raise ValueError(f"'{key}' must be between {s['min']} and {s['max']}.")
        clean[key] = int(number) if s["type"] == "int" else number
    result = CONCEPTS[name]["fn"](seed=int(seed), **clean)
    result["concept"] = name
    result["params"] = {**{k: v["default"] for k, v in spec.items()}, **clean}
    return result
