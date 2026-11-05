"""Simulación Monte Carlo y optimización.

  * `monte_carlo`: propaga la incertidumbre de varias entradas (con sus distribuciones y correlaciones) por una fórmula y devuelve el abanico de resultados:
    percentiles, probabilidad de pasar un umbral, riesgo en la cola y qué entrada pesa más.
  * `inputs_from_data`: construye las entradas a partir de columnas reales (remuestreo de lo observado y su correlación de rangos).
  * `linear_program`: programación lineal y entera (reparto de presupuesto, mezcla de productos, turnos) con los precios sombra de cada límite.
  * `optimize_expression`: maximiza o minimiza una fórmula cualquiera con límites y restricciones escritos con palabras.
Las fórmulas pasan por el intérprete seguro de `transform`: nunca se ejecuta código arbitrario."""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import Bounds, LinearConstraint, linprog, milp, minimize

from kikolens.analyze.transform import evaluate_expression

DISTRIBUTIONS = ("normal", "uniform", "triangular", "lognormal", "beta", "poisson", "fixed", "empirical")
MAX_DRAWS = 1_000_000
MIN_DRAWS = 200


# ── Distribuciones ───────────────────────────────────────────────────────────
def _need(spec: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in spec]
    if missing:
        raise ValueError(f"The '{spec['dist']}' distribution needs: {', '.join(missing)}.")


def _ppf(name: str, spec: dict, u: np.ndarray) -> np.ndarray:
    """Cuantil de cada distribución: así todas se pueden correlacionar a través de uniformes."""
    dist = spec.get("dist")
    if dist not in DISTRIBUTIONS:
        raise ValueError(f"'{name}': unknown distribution '{dist}'. Use one of {', '.join(DISTRIBUTIONS)}.")
    u = np.clip(u, 1e-9, 1 - 1e-9)
    if dist == "normal":
        _need(spec, "mean", "sd")
        if spec["sd"] < 0:
            raise ValueError(f"'{name}': the standard deviation cannot be negative.")
        return stats.norm.ppf(u, spec["mean"], spec["sd"])
    if dist == "uniform":
        _need(spec, "low", "high")
        if spec["low"] > spec["high"]:
            raise ValueError(f"'{name}': low is above high.")
        return spec["low"] + u * (spec["high"] - spec["low"])
    if dist == "triangular":
        _need(spec, "low", "mode", "high")
        if not spec["low"] <= spec["mode"] <= spec["high"] or spec["low"] == spec["high"]:
            raise ValueError(f"'{name}': it must hold that low <= mode <= high and low < high.")
        return stats.triang.ppf(u, (spec["mode"] - spec["low"]) / (spec["high"] - spec["low"]), spec["low"], spec["high"] - spec["low"])
    if dist == "lognormal":
        _need(spec, "mean", "sd")
        if spec["mean"] <= 0 or spec["sd"] < 0:
            raise ValueError(f"'{name}': a lognormal needs a positive mean and a non-negative standard deviation.")
        sigma2 = np.log(1 + (spec["sd"] / spec["mean"]) ** 2)                # mean y sd son los de la variable, no los de su logaritmo
        return stats.lognorm.ppf(u, np.sqrt(sigma2), scale=np.exp(np.log(spec["mean"]) - sigma2 / 2))
    if dist == "beta":
        _need(spec, "a", "b")
        if spec["a"] <= 0 or spec["b"] <= 0:
            raise ValueError(f"'{name}': beta parameters must be positive.")
        return stats.beta.ppf(u, spec["a"], spec["b"], loc=spec.get("low", 0.0), scale=spec.get("high", 1.0) - spec.get("low", 0.0))
    if dist == "poisson":
        _need(spec, "mean")
        if spec["mean"] < 0:
            raise ValueError(f"'{name}': the mean cannot be negative.")
        return stats.poisson.ppf(u, spec["mean"]).astype(float)
    if dist == "fixed":
        _need(spec, "value")
        return np.full(len(u), float(spec["value"]))
    values = np.asarray(spec.get("values", []), dtype=float)                  # empirical
    values = values[~np.isnan(values)]
    if len(values) < 5:
        raise ValueError(f"'{name}': an empirical distribution needs at least 5 observed values.")
    return np.quantile(values, u)


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    """Si la matriz que se pide no es posible (no es definida positiva) se acerca a la más parecida que sí lo es."""
    vals, vecs = np.linalg.eigh((matrix + matrix.T) / 2)
    if vals.min() > 1e-8:
        return matrix
    fixed = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


def _correlation_matrix(names: list, correlations) -> np.ndarray:
    k = len(names)
    out = np.eye(k)
    if correlations is None:
        return out
    if isinstance(correlations, pd.DataFrame):
        items = {(a, b): correlations.loc[a, b] for a in names for b in names if a in correlations.index and b in correlations.columns and a < b}
    else:
        items = dict(correlations)
    for (a, b), rho in items.items():
        if a not in names or b not in names:
            raise ValueError(f"Correlation between '{a}' and '{b}': unknown input.")
        if not -1 <= rho <= 1:
            raise ValueError(f"Correlation between '{a}' and '{b}' must be between -1 and 1.")
        i, j = names.index(a), names.index(b)
        out[i, j] = out[j, i] = rho
    return out


def draw(inputs: dict, n: int = 10000, correlations=None, seed: int = 0) -> pd.DataFrame:
    """Simula `n` valores de cada entrada; con `correlations` las entradas se mueven juntas (cópula gaussiana)."""
    if not inputs:
        raise ValueError("Give at least one input.")
    if not MIN_DRAWS <= n <= MAX_DRAWS:
        raise ValueError(f"The number of simulations must be between {MIN_DRAWS:,} and {MAX_DRAWS:,}.")
    names = list(inputs)
    corr = _nearest_correlation(_correlation_matrix(names, correlations))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, len(names))) @ np.linalg.cholesky(corr).T
    u = stats.norm.cdf(z)
    return pd.DataFrame({name: _ppf(name, inputs[name], u[:, i]) for i, name in enumerate(names)})


def inputs_from_data(df: pd.DataFrame, columns: list | None = None) -> tuple:
    """Entradas a partir de columnas numéricas: cada una remuestrea lo observado y la correlación de rangos se conserva. Devuelve (entradas, correlaciones)."""
    columns = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if not columns:
        raise ValueError("There are no numeric columns to simulate.")
    for c in columns:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"'{c}' must be a numeric column.")
    inputs = {c: {"dist": "empirical", "values": df[c].dropna().to_numpy(dtype=float)} for c in columns}
    corr = df[columns].corr(method="spearman").fillna(0.0) if len(columns) > 1 else None
    return inputs, corr


# ── Monte Carlo ──────────────────────────────────────────────────────────────
def monte_carlo(inputs: dict, expression: str, n: int = 10000, correlations=None, threshold: float | None = None, higher_is_better: bool = True,
                seed: int = 0) -> dict:
    """Pasa la incertidumbre de las entradas por una fórmula (por ejemplo `price * volume - cost`) y describe el resultado."""
    samples = draw(inputs, n, correlations, seed)
    try:
        result = evaluate_expression(samples, expression).astype(float)
    except KeyError as e:
        raise ValueError(f"The formula uses an unknown name: {e}.") from None
    if result.isna().all():
        raise ValueError("The formula gives no valid number for any simulation (check divisions by zero and the names used).")
    invalid = float(result.isna().mean())
    values = result.dropna().to_numpy()
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    pct = {f"p{p}": float(np.percentile(values, p)) for p in (5, 25, 50, 75, 95)}
    tail = np.percentile(values, 5 if higher_is_better else 95)
    tail_values = values[values <= tail] if higher_is_better else values[values >= tail]
    se = sd / np.sqrt(len(values))
    out = {"samples": samples.assign(**{"result": result}), "mean": mean, "sd": sd, "percentiles": pct, "standard_error": float(se), "n": len(values),
           "value_at_risk": float(tail), "expected_shortfall": float(tail_values.mean()), "invalid_share": invalid}
    notes = []
    if invalid > 0.01:
        notes.append(f"{invalid:.1%} of the simulations gave no number (division by zero or similar) and were left out.")
    if len(values) < 2000:
        notes.append("Few simulations: the percentiles in the tails will move from one run to the next. Use 10,000 or more.")
    if threshold is not None:
        above = float((values > threshold).mean())
        out["p_above"], out["p_below"], out["threshold"] = above, 1 - above, threshold
    sens = sensitivity(samples, result)
    out["sensitivity"] = sens
    out["notes"] = notes
    out["figure"] = distribution_figure(values, mean, pct, threshold)
    out["tornado_figure"] = tornado_figure(sens)
    out["convergence_figure"] = convergence_figure(values)
    top = sens.iloc[0] if len(sens) else None
    out["summary"] = (f"The result averages {mean:,.4g} (standard deviation {sd:,.4g}); 90% of the simulations fall between {pct['p5']:,.4g} and {pct['p95']:,.4g}."
                      + (f" The chance of going above {threshold:g} is {out['p_above']:.0%}." if threshold is not None else "")
                      + (f" What moves it most is '{top['input']}' ({top['share of variance']:.0%} of the variation)." if top is not None and top["share of variance"] > 0 else ""))
    return out


def sensitivity(samples: pd.DataFrame, result: pd.Series) -> pd.DataFrame:
    """Qué entradas explican la variación del resultado (correlación de rangos y reparto aproximado de la varianza)."""
    rows = []
    for c in samples.columns:
        x = samples[c]
        if x.nunique() < 2:
            rows.append({"input": c, "rank correlation": 0.0})
            continue
        rho = stats.spearmanr(x, result, nan_policy="omit")[0]
        rows.append({"input": c, "rank correlation": 0.0 if np.isnan(rho) else float(rho)})
    table = pd.DataFrame(rows)
    sq = table["rank correlation"] ** 2
    table["share of variance"] = sq / sq.sum() if sq.sum() > 0 else 0.0
    return table.sort_values("share of variance", ascending=False).reset_index(drop=True)


def distribution_figure(values: np.ndarray, mean: float, pct: dict, threshold: float | None) -> go.Figure:
    fig = go.Figure(go.Histogram(x=values, nbinsx=60, histnorm="probability", name="Simulations"))
    fig.add_vline(x=mean, line_dash="dash", annotation_text="mean")
    fig.add_vrect(x0=pct["p5"], x1=pct["p95"], opacity=0.08, line_width=0)
    if threshold is not None:
        fig.add_vline(x=threshold, line_color="crimson", annotation_text="threshold")
    fig.update_layout(title="Range of results (the shaded band holds 90%)", xaxis_title="Result", yaxis_title="Share of simulations", showlegend=False)
    return fig


def tornado_figure(sens: pd.DataFrame) -> go.Figure:
    ordered = sens.sort_values("share of variance")
    fig = go.Figure(go.Bar(x=ordered["rank correlation"], y=ordered["input"], orientation="h"))
    fig.update_layout(title="What moves the result (rank correlation with each input)", xaxis_title="Correlation", xaxis_range=[-1, 1])
    return fig


def convergence_figure(values: np.ndarray) -> go.Figure:
    steps = np.arange(1, len(values) + 1)
    running = np.cumsum(values) / steps
    keep = np.unique(np.linspace(0, len(values) - 1, min(400, len(values))).astype(int))
    fig = go.Figure(go.Scatter(x=steps[keep], y=running[keep], mode="lines"))
    fig.update_layout(title="The average settles as simulations are added", xaxis_title="Simulations", yaxis_title="Running mean")
    return fig


# ── Optimización lineal y entera ─────────────────────────────────────────────
def linear_program(objective: dict, constraints: list, bounds: dict | None = None, maximize: bool = True, integer: list | None = None) -> dict:
    """Maximiza o minimiza `sum(coef * variable)` con restricciones `{"coefs": {var: coef}, "op": "<=" | ">=" | "==", "rhs": número, "name": opcional}`.
    `bounds` da (mínimo, máximo) por variable (por defecto 0 a infinito); `integer` lista las variables que deben ser enteras."""
    names = list(objective)
    if not names:
        raise ValueError("The objective has no variables.")
    integer = list(integer or [])
    for v in integer:
        if v not in names:
            raise ValueError(f"'{v}' is marked as integer but is not in the objective.")
    sign = -1.0 if maximize else 1.0
    c = sign * np.array([float(objective[v]) for v in names])
    rows, rhs, ops, labels = [], [], [], []
    for i, con in enumerate(constraints):
        if con.get("op") not in ("<=", ">=", "=="):
            raise ValueError(f"Constraint {i + 1}: the operator must be <=, >= or ==.")
        unknown = [v for v in con.get("coefs", {}) if v not in names]
        if unknown:
            raise ValueError(f"Constraint {i + 1} uses unknown variable(s): {', '.join(unknown)}.")
        rows.append([float(con["coefs"].get(v, 0.0)) for v in names])
        rhs.append(float(con["rhs"]))
        ops.append(con["op"])
        labels.append(con.get("name") or f"constraint {i + 1}")
    bnds = []
    for v in names:
        lo, hi = (bounds or {}).get(v, (0.0, None))
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"'{v}': the minimum is above the maximum.")
        bnds.append((lo, hi))
    A = np.array(rows) if rows else np.zeros((0, len(names)))
    b = np.array(rhs)
    shadow = {}
    if integer:
        lo = np.array([-np.inf if b_[0] is None else b_[0] for b_ in bnds])
        hi = np.array([np.inf if b_[1] is None else b_[1] for b_ in bnds])
        cons = []
        if rows:
            low = np.array([x if o in (">=", "==") else -np.inf for x, o in zip(b, ops)])
            high = np.array([x if o in ("<=", "==") else np.inf for x, o in zip(b, ops)])
            cons = [LinearConstraint(A, low, high)]
        res = milp(c, constraints=cons, bounds=Bounds(lo, hi), integrality=np.array([1 if v in integer else 0 for v in names]))
        ok, x, value, message = res.status == 0, res.x, (res.fun if res.status == 0 else None), res.message
    else:
        ub = [i for i, o in enumerate(ops) if o == "<="]
        ge = [i for i, o in enumerate(ops) if o == ">="]
        eq = [i for i, o in enumerate(ops) if o == "=="]
        A_ub = np.vstack([A[ub], -A[ge]]) if (ub or ge) else None
        b_ub = np.concatenate([b[ub], -b[ge]]) if (ub or ge) else None
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A[eq] if eq else None, b_eq=b[eq] if eq else None, bounds=bnds, method="highs")
        ok, x, value, message = res.status == 0, res.x, (res.fun if res.status == 0 else None), res.message
        if ok:
            marg = list(res.ineqlin.marginals) if A_ub is not None else []
            eqm = list(res.eqlin.marginals) if eq else []
            for pos, i in enumerate(ub):
                shadow[labels[i]] = sign * marg[pos]
            for pos, i in enumerate(ge):
                shadow[labels[i]] = -sign * marg[len(ub) + pos]
            for pos, i in enumerate(eq):
                shadow[labels[i]] = sign * eqm[pos]
    if not ok:
        reason = "no solution satisfies every constraint" if res.status == 2 else "the problem is unbounded: something can grow without limit" if res.status == 3 else str(message)
        raise ValueError(f"No optimum found: {reason}.")
    solution = pd.Series(x, index=names).round(10)
    used = A @ x if rows else np.array([])
    table = pd.DataFrame({"constraint": labels, "used": used, "limit": b, "operator": ops})
    table["binding"] = np.isclose(table["used"], table["limit"], rtol=1e-7, atol=1e-7)
    table["worth of one more unit"] = [shadow.get(lbl, np.nan) for lbl in labels] if shadow else np.nan
    objective_value = float(sign * value)
    binding = table[table["binding"]]
    note = ""
    if shadow and len(binding):
        best = binding.assign(w=binding["worth of one more unit"].abs()).sort_values("w", ascending=False).iloc[0]
        if best["w"] > 0:
            note = f" The limit worth relaxing first is '{best['constraint']}': one more unit changes the objective by {best['worth of one more unit']:+.4g}."
    return {"solution": solution, "objective": objective_value, "constraints": table, "integer": integer, "figure": _solution_figure(solution),
            "summary": f"The best {'maximum' if maximize else 'minimum'} is {objective_value:,.6g}, reached with " + ", ".join(f"{k} = {v:,.4g}" for k, v in solution.items()) + "." + note}


def _solution_figure(solution: pd.Series) -> go.Figure:
    fig = go.Figure(go.Bar(x=solution.index.astype(str), y=solution.values))
    fig.update_layout(title="Optimal values", yaxis_title="Value")
    return fig


# ── Optimización de una fórmula cualquiera ───────────────────────────────────
_CONSTRAINT = re.compile(r"^(.*?)(<=|>=|==|<|>|=)(.*)$")


def _value(expression: str, point: dict) -> float:
    frame = pd.DataFrame({k: [float(v)] for k, v in point.items()})
    value = float(evaluate_expression(frame, expression).iloc[0])
    return value if np.isfinite(value) else np.nan


def optimize_expression(expression: str, bounds: dict, constraints: list | None = None, maximize: bool = True, starts: int = 12, seed: int = 0) -> dict:
    """Busca los valores de las variables (con sus límites) que más suben o bajan la fórmula, respetando restricciones como `a + b <= 100`.
    Prueba varios puntos de partida porque la fórmula puede tener varios picos."""
    names = list(bounds)
    if not names:
        raise ValueError("Give at least one variable with its bounds.")
    lo = np.array([float(bounds[v][0]) for v in names])
    hi = np.array([float(bounds[v][1]) for v in names])
    if (lo > hi).any():
        raise ValueError("A variable has its minimum above its maximum.")
    parsed = []
    for text in constraints or []:
        m = _CONSTRAINT.match(text.replace(" ", "")) if text.strip() else None
        if not m:
            raise ValueError(f"Cannot read the constraint '{text}'. Write it like 'a + b <= 100'.")
        left, op, right = m.groups()
        parsed.append((left, "==" if op == "=" else {"<": "<=", ">": ">="}.get(op, op), right))
    point = lambda x: dict(zip(names, x))                      # noqa: E731
    _value(expression, point(lo))                              # falla ya con un mensaje claro si la fórmula no es válida

    def objective(x):
        v = _value(expression, point(x))
        return 1e12 if np.isnan(v) else (-v if maximize else v)

    cons = []
    for left, op, right in parsed:
        def gap(x, left=left, right=right):
            return _value(left, point(x)) - _value(right, point(x))
        if op == "==":
            cons.append({"type": "eq", "fun": gap})
        elif op == "<=":
            cons.append({"type": "ineq", "fun": lambda x, gap=gap: -gap(x)})
        else:
            cons.append({"type": "ineq", "fun": gap})
    rng = np.random.default_rng(seed)
    candidates = [(lo + hi) / 2] + [lo + rng.random(len(names)) * (hi - lo) for _ in range(max(starts - 1, 0))]
    best, tried = None, 0
    for x0 in candidates:
        try:
            res = minimize(objective, x0, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
        except (ValueError, FloatingPointError):
            continue
        tried += 1
        feasible = all((abs(c["fun"](res.x)) < 1e-5 if c["type"] == "eq" else c["fun"](res.x) > -1e-5) for c in cons)
        if feasible and np.all(res.x >= lo - 1e-9) and np.all(res.x <= hi + 1e-9) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise ValueError("No point satisfying every constraint was found: check that they do not contradict each other or the bounds.")
    x = np.clip(best.x, lo, hi)
    value = _value(expression, point(x))
    at_edge = [v for v, xi, l_, h_ in zip(names, x, lo, hi) if np.isclose(xi, l_, atol=1e-6) or np.isclose(xi, h_, atol=1e-6)]
    notes = []
    if at_edge:
        notes.append(f"{', '.join(at_edge)} ended at a bound: widening it could improve the result.")
    return {"solution": pd.Series(x, index=names), "objective": float(value), "starts": tried, "at_bound": at_edge, "notes": notes,
            "summary": f"The {'highest' if maximize else 'lowest'} value found is {value:,.6g}, with " + ", ".join(f"{k} = {v:,.4g}" for k, v in zip(names, x)) + "."}


# ── Lectura de texto (terminal y panel) ──────────────────────────────────────
_PARAMETERS = {"normal": ("mean", "sd"), "uniform": ("low", "high"), "triangular": ("low", "mode", "high"), "lognormal": ("mean", "sd"), "beta": ("a", "b"),
               "poisson": ("mean",), "fixed": ("value",)}


def parse_input(text: str) -> tuple:
    """`precio=normal(10, 2)` -> ("precio", {"dist": "normal", "mean": 10.0, "sd": 2.0})."""
    m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*=\s*([a-z]+)\s*\(([^)]*)\)\s*", text or "")
    if not m or m.group(2) not in _PARAMETERS:
        raise ValueError(f"Cannot read '{text}'. Write it like price=normal(10, 2); distributions: {', '.join(_PARAMETERS)}.")
    name, dist, raw = m.groups()
    try:
        numbers = [float(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"'{text}': the parameters must be numbers.") from None
    if len(numbers) != len(_PARAMETERS[dist]):
        raise ValueError(f"'{dist}' takes {len(_PARAMETERS[dist])} number(s): {', '.join(_PARAMETERS[dist])}.")
    return name, {"dist": dist, **dict(zip(_PARAMETERS[dist], numbers))}


def parse_variable(text: str) -> tuple:
    """`x=0:10` -> ("x", (0.0, 10.0))."""
    m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*=\s*(-?[\d.eE+-]+)\s*:\s*(-?[\d.eE+-]+)\s*", text or "")
    if not m:
        raise ValueError(f"Cannot read '{text}'. Write it like x=0:10 (minimum:maximum).")
    try:
        return m.group(1), (float(m.group(2)), float(m.group(3)))
    except ValueError:
        raise ValueError(f"'{text}': the bounds must be numbers.") from None


_TERM = re.compile(r"([+-]?)\s*(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+)?\s*\*?\s*([A-Za-z_]\w*)?")


def parse_linear(text: str) -> dict:
    """`3x + 5*y - 2z` -> {"x": 3, "y": 5, "z": -2}. Solo términos lineales (nada de x*y ni x**2)."""
    source = (text or "").replace(" ", "")
    if not source:
        raise ValueError("The expression is empty.")
    if re.search(r"[A-Za-z_]\w*\*[A-Za-z_]|\*\*|/|\(|\)", source):
        raise ValueError(f"'{text}' is not linear: use sums of numbers times variables, like 3*x + 5*y.")
    out, pos = {}, 0
    while pos < len(source):
        m = _TERM.match(source, pos)
        if not m or m.end() == pos or (m.group(2) is None and m.group(3) is None):
            raise ValueError(f"Cannot read '{text}' near '{source[pos:pos + 8]}'.")
        sign, number, name = m.groups()
        if name is None:
            raise ValueError(f"'{text}' has a constant term; move it to the other side of the constraint.")
        out[name] = out.get(name, 0.0) + (-1 if sign == "-" else 1) * (float(number) if number else 1.0)
        pos = m.end()
    return out


def parse_linear_constraint(text: str, name: str | None = None) -> dict:
    """`x + 2y <= 12` -> {"coefs": {...}, "op": "<=", "rhs": 12.0}."""
    m = re.fullmatch(r"(.+?)(<=|>=|==|=)\s*(-?[\d.eE+-]+)\s*", (text or "").replace(" ", ""))
    if not m:
        raise ValueError(f"Cannot read the constraint '{text}'. Write it like 'x + 2*y <= 12' with a number on the right.")
    try:
        rhs = float(m.group(3))
    except ValueError:
        raise ValueError(f"'{text}': the right-hand side must be a number.") from None
    return {"coefs": parse_linear(m.group(1)), "op": "==" if m.group(2) == "=" else m.group(2), "rhs": rhs, "name": name or text.strip()}
