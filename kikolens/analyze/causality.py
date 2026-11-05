"""Inferencia causal con datos observacionales. Compara la misma pregunta («¿qué efecto tuvo X sobre Y?») con varios métodos, para que se vea si
coinciden, y dice cuándo no se puede fiar de ellos: falta de solapamiento entre tratados y no tratados, covariables sin equilibrar, o un efecto que bastaría
un factor oculto moderado para explicar.

  * `estimate_effect`: ponderación por propensión (IPW), emparejamiento, regresión y el estimador doblemente robusto (AIPW), con equilibrio de covariables,
    solapamiento y valor E de sensibilidad.
  * `uplift`: a quién le sirve el tratamiento (efecto por fila, curva de Qini, deciles).
  * `regression_discontinuity`: efecto de pasar un umbral (una nota de corte, una edad).
  * `synthetic_control`: el efecto sobre una unidad tratada frente a una combinación de unidades de control, con prueba de placebos.
  * `iv_2sls`: variables instrumentales cuando el tratamiento está contaminado por algo que no se observa.
Ninguno prueba causalidad por sí solo: dependen de supuestos que se dicen en cada resultado."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from kikolens.analyze.modellab import build_preprocessor
from kikolens.analyze.quality import infer_types

METHODS = ("naive", "regression", "ipw", "matching", "aipw")
MIN_ROWS = 60
OVERLAP_BOUNDS = (0.05, 0.95)


# ── Preparación ──────────────────────────────────────────────────────────────
def _binary(series: pd.Series, treated_value=None) -> tuple:
    values = list(series.dropna().unique())
    if len(values) != 2:
        raise ValueError(f"The treatment column must have exactly two values; it has {len(values)}.")
    if treated_value is None:
        as_text = {str(v).lower(): v for v in values}
        if set(as_text) in ({"0", "1"}, {"false", "true"}, {"0.0", "1.0"}):
            treated_value = as_text.get("1", as_text.get("true", as_text.get("1.0")))
        else:
            treated_value = next((as_text[k] for k in ("yes", "treated", "true", "1") if k in as_text), sorted(values, key=str)[-1])
    elif treated_value not in values:
        raise ValueError(f"'{treated_value}' is not a value of the treatment column.")
    return (series == treated_value).astype(int), treated_value


def _prepare(df: pd.DataFrame, treatment: str, outcome: str, covariates, treated_value=None):
    for c in (treatment, outcome):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    df, _ = infer_types(df)
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError(f"The outcome '{outcome}' must be numeric (use 0/1 for yes/no).")
    covariates = [c for c in (covariates if covariates is not None else df.columns) if c not in (treatment, outcome) and c in df.columns]
    if not covariates:
        raise ValueError("At least one covariate is needed to adjust for confounding.")
    t, treated_value = _binary(df[treatment], treated_value)
    data = pd.concat([t.rename("_t"), df[outcome].astype(float).rename("_y"), df[covariates]], axis=1).dropna(subset=["_t", "_y"])
    if len(data) < MIN_ROWS or data["_t"].sum() < 15 or (1 - data["_t"]).sum() < 15:
        raise ValueError(f"At least {MIN_ROWS} rows and 15 treated and 15 untreated rows are needed.")
    X_raw = data[covariates]
    prep = build_preprocessor(X_raw).fit(X_raw)
    X = np.asarray(prep.transform(X_raw), dtype=float)
    names = list(prep.get_feature_names_out())
    return data["_t"].to_numpy(), data["_y"].to_numpy(), X, names, covariates, treated_value


def _folds(t: np.ndarray, k: int, seed: int):
    return StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(t)), t)


def _propensity(X: np.ndarray, t: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Probabilidad de ser tratado dada la covariables, con validación cruzada (cada fila se predice con un modelo que no la vio)."""
    out = np.zeros(len(t))
    Z = StandardScaler().fit_transform(X)
    for train, test in _folds(t, k, seed):
        model = LogisticRegression(C=0.5, max_iter=2000).fit(Z[train], t[train])
        out[test] = model.predict_proba(Z[test])[:, 1]
    return out


def _outcome_models(X: np.ndarray, y: np.ndarray, t: np.ndarray, k: int = 5, seed: int = 0) -> tuple:
    """Predicción de Y si no se hubiera tratado (mu0) y si sí (mu1) para cada fila, con validación cruzada."""
    mu0, mu1 = np.zeros(len(y)), np.zeros(len(y))
    for train, test in _folds(t, k, seed):
        for arm, target in ((0, mu0), (1, mu1)):
            rows = train[t[train] == arm]
            model = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.08, random_state=seed).fit(X[rows], y[rows])
            target[test] = model.predict(X[test])
    return mu0, mu1


# ── Estimadores ──────────────────────────────────────────────────────────────
def _ipw(y, t, e):
    w1, w0 = t / e, (1 - t) / (1 - e)
    return float(np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0))


def _regression(y, t, X):
    A = np.column_stack([np.ones(len(y)), t, StandardScaler().fit_transform(X)])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(beta[1])


def _matching(y, t, e):
    """Emparejamiento 1 a 1 con reemplazo por el logit de la propensión, con calibre de 0,2 desviaciones: efecto sobre los tratados (ATT)."""
    logit = np.log(e / (1 - e))
    caliper = 0.2 * logit.std()
    treated, control = np.where(t == 1)[0], np.where(t == 0)[0]
    nn = NearestNeighbors(n_neighbors=1).fit(logit[control, None])
    distance, index = nn.kneighbors(logit[treated, None])
    keep = distance[:, 0] <= caliper
    if keep.sum() < 10:
        raise ValueError("Almost no treated row has a comparable untreated row. The groups barely overlap.")
    matched = control[index[keep, 0]]
    return float(np.mean(y[treated[keep]] - y[matched])), int(keep.sum()), int(len(treated))


def _aipw(y, t, e, mu0, mu1):
    """Estimador doblemente robusto: vale si el modelo de propensión *o* el de resultado es correcto. Devuelve (efecto, error estándar)."""
    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))


def e_value(effect: float, sd: float, ci_bound: float | None = None) -> dict:
    """Valor E (VanderWeele y Ding): lo fuerte que tendría que ser un factor no medido, asociado a tratamiento y resultado, para anular el efecto.
    El efecto estandarizado d se pasa a razón de riesgos aproximada con exp(0,91 d)."""
    def convert(d):
        rr = np.exp(0.91 * abs(d))
        return float(rr + np.sqrt(rr * (rr - 1)))
    d = effect / sd if sd else 0.0
    out = {"point": convert(d), "standardized_effect": float(d)}
    if ci_bound is not None:
        out["interval"] = 1.0 if (np.sign(ci_bound) != np.sign(effect)) else convert(ci_bound / sd)
    return out


def balance_table(X: np.ndarray, names: list, t: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    """Diferencia media estandarizada de cada covariable entre tratados y no tratados, sin ponderar y ponderada (|valor| < 0,1 es buen equilibrio)."""
    def smd(w):
        out = []
        for j in range(X.shape[1]):
            m1 = np.average(X[t == 1, j], weights=w[t == 1])
            m0 = np.average(X[t == 0, j], weights=w[t == 0])
            sd = np.sqrt((X[t == 1, j].var() + X[t == 0, j].var()) / 2) or 1.0
            out.append((m1 - m0) / sd)
        return np.array(out)
    table = pd.DataFrame({"covariate": names, "before": smd(np.ones(len(t))), "after": smd(weights)})
    table["improved"] = table["after"].abs() < table["before"].abs()
    return table.sort_values("before", key=lambda s: -s.abs()).reset_index(drop=True)


def estimate_effect(df: pd.DataFrame, treatment: str, outcome: str, covariates=None, treated_value=None, method: str = "aipw", trim: float = 0.02,
                    bootstrap: int = 150, seed: int = 0) -> dict:
    """Efecto medio del tratamiento sobre el resultado con varios métodos. `method` es el que se destaca; todos aparecen en `table`."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}.")
    if not 0 <= trim < 0.2:
        raise ValueError("trim must be between 0 and 0.2.")
    t, y, X, names, covariates, treated_value = _prepare(df, treatment, outcome, covariates, treated_value)
    e = np.clip(_propensity(X, t, seed=seed), trim or 1e-3, 1 - (trim or 1e-3))
    mu0, mu1 = _outcome_models(X, y, t, seed=seed)
    outside = float(np.mean((e < OVERLAP_BOUNDS[0]) | (e > OVERLAP_BOUNDS[1])))
    weights = np.where(t == 1, 1 / e, 1 / (1 - e))
    rng = np.random.default_rng(seed)

    def boot(fn):
        values = []
        for _ in range(bootstrap):
            idx = rng.integers(0, len(y), len(y))
            try:
                values.append(fn(idx))
            except (ValueError, np.linalg.LinAlgError):
                continue
        return np.percentile(values, [2.5, 97.5]) if len(values) >= 20 else (np.nan, np.nan)

    naive = float(y[t == 1].mean() - y[t == 0].mean())
    se_naive = float(np.sqrt(y[t == 1].var(ddof=1) / (t == 1).sum() + y[t == 0].var(ddof=1) / (t == 0).sum()))
    rows = [{"method": "naive", "effect": naive, "low": naive - 1.96 * se_naive, "high": naive + 1.96 * se_naive, "adjusts for confounding": False, "estimates": "ATE"}]
    reg = _regression(y, t, X)
    lo, hi = boot(lambda i: _regression(y[i], t[i], X[i]))
    rows.append({"method": "regression", "effect": reg, "low": lo, "high": hi, "adjusts for confounding": True, "estimates": "ATE"})
    ipw = _ipw(y, t, e)
    lo, hi = boot(lambda i: _ipw(y[i], t[i], e[i]))
    rows.append({"method": "ipw", "effect": ipw, "low": lo, "high": hi, "adjusts for confounding": True, "estimates": "ATE"})
    att, matched, treated_n = _matching(y, t, e)
    lo, hi = boot(lambda i: _matching(y[i], t[i], e[i])[0])
    rows.append({"method": "matching", "effect": att, "low": lo, "high": hi, "adjusts for confounding": True, "estimates": "ATT"})
    aipw, se = _aipw(y, t, e, mu0, mu1)
    rows.append({"method": "aipw", "effect": aipw, "low": aipw - 1.96 * se, "high": aipw + 1.96 * se, "adjusts for confounding": True, "estimates": "ATE"})
    table = pd.DataFrame(rows)
    chosen = table.set_index("method").loc[method]
    balance = balance_table(X, names, t, weights)
    worse = int((balance["after"].abs() > 0.1).sum())
    ev = e_value(float(chosen["effect"]), float(y.std(ddof=1)), float(chosen["low"] if chosen["effect"] > 0 else chosen["high"]))
    notes = []
    if outside > 0.1:
        notes.append(f"{outside:.0%} of the rows have a propensity below {OVERLAP_BOUNDS[0]} or above {OVERLAP_BOUNDS[1]}: treated and untreated rows are too different "
                     "in some areas, and the estimate leans on a few rows there. Narrow the question to the area where they overlap.")
    if worse:
        notes.append(f"{worse} covariate(s) are still unbalanced after weighting (|standardised difference| above 0.1): the adjustment may be incomplete.")
    if abs(naive - float(chosen["effect"])) > 2 * se_naive:
        notes.append("Adjusting changes the result a lot compared with the raw difference: confounding matters here, and an unmeasured confounder could matter too.")
    return {"table": table, "method": method, "effect": float(chosen["effect"]), "interval": (float(chosen["low"]), float(chosen["high"])), "naive": naive,
            "balance": balance, "propensity": e, "treated": t, "outside_overlap": outside, "e_value": ev, "matched": (matched, treated_n),
            "treated_value": treated_value, "covariates": covariates, "notes": notes, "figure": effects_figure(table, method),
            "overlap_figure": overlap_figure(e, t), "balance_figure": balance_figure(balance),
            "summary": (f"Treatment '{treatment}' changes '{outcome}' by {chosen['effect']:+.4g} ({method}, 95% range {chosen['low']:+.4g} to {chosen['high']:+.4g}); "
                        f"the raw difference is {naive:+.4g}. To explain it away, an unmeasured factor would need a strength of {ev['point']:.2f} (E-value; "
                        f"{ev.get('interval', float('nan')):.2f} for the interval). This assumes every confounder is among the covariates.")}


def effects_figure(table: pd.DataFrame, chosen: str) -> go.Figure:
    t = table.iloc[::-1]
    fig = go.Figure(go.Scatter(x=t["effect"], y=t["method"], mode="markers", marker=dict(size=[13 if m == chosen else 9 for m in t["method"]]),
                               error_x=dict(type="data", symmetric=False, array=t["high"] - t["effect"], arrayminus=t["effect"] - t["low"])))
    fig.add_vline(x=0, line_dash="dot")
    fig.update_layout(title="Estimated effect by method (95% range)", xaxis_title="Effect", showlegend=False)
    return fig


def overlap_figure(e: np.ndarray, t: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for arm, name in ((1, "Treated"), (0, "Untreated")):
        fig.add_trace(go.Histogram(x=e[t == arm], name=name, opacity=0.65, nbinsx=30, histnorm="probability"))
    fig.update_layout(barmode="overlay", title="Propensity score: do treated and untreated rows overlap?", xaxis_title="Probability of being treated")
    return fig


def balance_figure(balance: pd.DataFrame) -> go.Figure:
    top = balance.head(15).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=top["before"], y=top["covariate"], mode="markers", name="Before"))
    fig.add_trace(go.Scatter(x=top["after"], y=top["covariate"], mode="markers", name="After weighting"))
    for bound in (-0.1, 0.1):
        fig.add_vline(x=bound, line_dash="dot")
    fig.update_layout(title="Covariate balance (standardised difference)", xaxis_title="Treated minus untreated, in standard deviations")
    return fig


# ── Uplift ───────────────────────────────────────────────────────────────────
def uplift(df: pd.DataFrame, treatment: str, outcome: str, covariates=None, treated_value=None, folds: int = 5, seed: int = 0) -> dict:
    """Efecto del tratamiento *por fila* con un aprendiz T (un modelo por grupo, con validación cruzada) y su evaluación con la curva de Qini.
    Supone que el tratamiento se asignó al azar o que las covariables lo explican; avisa si la asignación no parece aleatoria."""
    t, y, X, names, covariates, treated_value = _prepare(df, treatment, outcome, covariates, treated_value)
    mu0, mu1 = _outcome_models(X, y, t, folds, seed)
    cate = mu1 - mu0
    e = _propensity(X, t, folds, seed)
    randomised = float(np.abs(e - t.mean()).mean())
    order = np.argsort(-cate)
    n_t, n_c = np.cumsum(t[order]), np.cumsum(1 - t[order])
    gain = np.cumsum(y[order] * t[order]) - np.cumsum(y[order] * (1 - t[order])) * np.divide(n_t, n_c, out=np.zeros(len(y)), where=n_c > 0)
    fraction = np.arange(1, len(y) + 1) / len(y)
    total = gain[-1]
    random_line = fraction * total
    integrate = getattr(np, "trapezoid", None) or np.trapz        # numpy 2 renombró trapz
    qini = float(integrate(gain - random_line, fraction))
    deciles = pd.DataFrame({"cate": cate, "t": t, "y": y})
    deciles["decile"] = pd.qcut(deciles["cate"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    summary = deciles.groupby("decile", observed=True).apply(lambda g: pd.Series({
        "rows": len(g), "predicted uplift": g["cate"].mean(),
        "observed uplift": (g.loc[g["t"] == 1, "y"].mean() - g.loc[g["t"] == 0, "y"].mean()) if g["t"].nunique() == 2 else np.nan}), include_groups=False).reset_index()
    surrogate = HistGradientBoostingRegressor(max_iter=60, random_state=seed).fit(X, cate)
    imp = permutation_importance(surrogate, X, cate, n_repeats=3, random_state=seed).importances_mean
    drivers = pd.Series(imp, index=names).clip(lower=0).sort_values(ascending=False).head(10)
    top_share = 0.3
    k = int(top_share * len(y))
    top_rows = order[:k]
    top_obs = float(y[top_rows][t[top_rows] == 1].mean() - y[top_rows][t[top_rows] == 0].mean()) if len(set(t[top_rows])) == 2 else np.nan
    all_obs = float(y[t == 1].mean() - y[t == 0].mean())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fraction, y=gain, name="Targeting by predicted uplift"))
    fig.add_trace(go.Scatter(x=fraction, y=random_line, name="Random targeting", line=dict(dash="dash")))
    fig.update_layout(title="Qini curve: extra outcome from treating the best-ranked rows first", xaxis_title="Share of rows treated (best first)", yaxis_title="Incremental outcome")
    notes = []
    if randomised > 0.1:
        notes.append("The treatment does not look randomly assigned (the covariates predict it). Uplift is only reliable if they capture everything that decided who was treated.")
    return {"cate": pd.Series(cate), "deciles": summary, "drivers": drivers, "qini": qini, "top_uplift": top_obs, "average_uplift": all_obs, "randomised_gap": randomised,
            "treated_value": treated_value, "order": order, "figure": fig, "notes": notes,
            "summary": (f"The average effect is {all_obs:+.4g}, but treating only the 30% with the highest predicted uplift gives {top_obs:+.4g} per row"
                        + (" (a clear gain over treating everyone)." if top_obs > all_obs * 1.2 and all_obs else ".") + f" Qini area against random targeting: {qini:+.4g}.")}


# ── Discontinuidad ───────────────────────────────────────────────────────────
def _local_fit(x, y, h, kernel):
    u = x / h
    w = np.clip(1 - np.abs(u), 0, None) if kernel == "triangular" else (np.abs(u) <= 1).astype(float)
    keep = w > 0
    return x[keep], y[keep], w[keep]


def _jump(x, y, cutoff, h, kernel, order):
    xs, ys = x - cutoff, y
    left, right = xs < 0, xs >= 0
    parts = []
    for mask in (left, right):
        xl, yl, wl = _local_fit(xs[mask], ys[mask], h, kernel)
        if len(xl) < order + 3:
            raise ValueError("Too few rows near the cutoff for this bandwidth. Widen it.")
        A = np.vander(xl, order + 1, increasing=True)
        W = wl
        coef = np.linalg.lstsq(A * np.sqrt(W)[:, None], yl * np.sqrt(W), rcond=None)[0]
        resid = yl - A @ coef
        bread = np.linalg.pinv((A * W[:, None]).T @ A)
        meat = (A * (W * resid)[:, None]).T @ (A * (W * resid)[:, None])
        cov = bread @ meat @ bread * len(xl) / max(len(xl) - order - 1, 1)
        parts.append((coef[0], cov[0, 0], len(xl)))
    return parts[1][0] - parts[0][0], float(np.sqrt(parts[0][1] + parts[1][1])), parts[0][2], parts[1][2]


def regression_discontinuity(df: pd.DataFrame, running: str, outcome: str, cutoff: float, bandwidth: float | None = None, kernel: str = "triangular",
                             order: int = 1) -> dict:
    """Efecto de pasar un umbral con regresión polinómica local a cada lado (ponderada por cercanía). Incluye varios anchos de banda y una prueba de manipulación."""
    for c in (running, outcome):
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"'{c}' must be a numeric column.")
    if kernel not in ("triangular", "uniform"):
        raise ValueError("kernel must be triangular or uniform.")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    data = df[[running, outcome]].dropna().astype(float)
    x, y = data[running].to_numpy(), data[outcome].to_numpy()
    if (x < cutoff).sum() < 30 or (x >= cutoff).sum() < 30:
        raise ValueError("At least 30 rows are needed on each side of the cutoff.")
    h = bandwidth or float(1.84 * x.std() * len(x) ** (-0.2))
    if h <= 0:
        raise ValueError("The bandwidth must be positive.")
    jump, se, n_left, n_right = _jump(x, y, cutoff, h, kernel, order)
    table = []
    for factor in (0.5, 0.75, 1.0, 1.5, 2.0):
        try:
            j, s, nl, nr = _jump(x, y, cutoff, h * factor, kernel, order)
            table.append({"bandwidth": h * factor, "effect": j, "low": j - 1.96 * s, "high": j + 1.96 * s, "rows": nl + nr})
        except ValueError:
            continue
    near = np.abs(x - cutoff) <= h
    left_n, right_n = int((near & (x < cutoff)).sum()), int((near & (x >= cutoff)).sum())
    p_density = float(stats.binomtest(right_n, left_n + right_n, 0.5).pvalue) if left_n + right_n else 1.0
    z = stats.norm.ppf(0.975)
    notes = []
    if p_density < 0.05:
        notes.append(f"More rows fall on one side of the cutoff than chance allows ({left_n} against {right_n}, p = {p_density:.3f}): people may be sorting themselves across it, which breaks the design.")
    if len(table) > 1 and (max(r["effect"] for r in table) - min(r["effect"] for r in table)) > 2 * se:
        notes.append("The estimate moves a lot with the bandwidth: be careful with it.")
    return {"effect": float(jump), "se": se, "interval": (float(jump - z * se), float(jump + z * se)), "bandwidth": h, "rows": (n_left, n_right), "table": pd.DataFrame(table),
            "density_p": p_density, "notes": notes, "figure": _rd_figure(x, y, cutoff, h, kernel, order),
            "summary": f"Crossing {cutoff:g} changes '{outcome}' by {jump:+.4g} (95% range {jump - z * se:+.4g} to {jump + z * se:+.4g}), using rows within {h:.4g} of the cutoff."}


def _rd_figure(x, y, cutoff, h, kernel, order) -> go.Figure:
    window = np.abs(x - cutoff) <= 2.5 * h
    bins = np.linspace(x[window].min(), x[window].max(), 31)
    index = np.digitize(x[window], bins)
    centres = [x[window][index == i].mean() for i in range(1, len(bins)) if (index == i).sum() > 0]
    means = [y[window][index == i].mean() for i in range(1, len(bins)) if (index == i).sum() > 0]
    fig = go.Figure(go.Scatter(x=centres, y=means, mode="markers", name="Binned averages"))
    for side in (x < cutoff, x >= cutoff):
        xs, ys, w = _local_fit(x[side] - cutoff, y[side], h, kernel)
        if len(xs) > order + 2:
            coef = np.linalg.lstsq(np.vander(xs, order + 1, increasing=True) * np.sqrt(w)[:, None], ys * np.sqrt(w), rcond=None)[0]
            grid = np.linspace(xs.min(), xs.max(), 50)
            fig.add_trace(go.Scatter(x=grid + cutoff, y=np.vander(grid, order + 1, increasing=True) @ coef, mode="lines", name="Local fit", showlegend=False))
    fig.add_vline(x=cutoff, line_dash="dash")
    fig.update_layout(title="Outcome around the cutoff", xaxis_title="Running variable", yaxis_title="Outcome")
    return fig


# ── Control sintético ────────────────────────────────────────────────────────
def _weights(pre_treated: np.ndarray, pre_donors: np.ndarray) -> np.ndarray:
    n = pre_donors.shape[1]
    result = minimize(lambda w: float(np.sum((pre_treated - pre_donors @ w) ** 2)), np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}], options={"maxiter": 300, "ftol": 1e-10})
    return np.clip(result.x, 0, None) / max(np.clip(result.x, 0, None).sum(), 1e-12)


def synthetic_control(df: pd.DataFrame, unit: str, time: str, outcome: str, treated_unit, intervention, donors=None) -> dict:
    """Efecto sobre una unidad tratada frente a una media ponderada de unidades de control que la imita antes de la intervención. La prueba compara su
    brecha posterior con la que saldría si se «tratara» a cada unidad de control (placebos)."""
    for c in (unit, time, outcome):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    wide = df.pivot_table(index=time, columns=unit, values=outcome, aggfunc="mean").sort_index()
    if treated_unit not in wide.columns:
        raise ValueError(f"'{treated_unit}' is not a value of '{unit}'.")
    wide = wide.dropna(axis=1)
    if treated_unit not in wide.columns:
        raise ValueError("The treated unit has missing periods.")
    donor_names = [d for d in (donors or wide.columns) if d != treated_unit and d in wide.columns]
    if len(donor_names) < 3:
        raise ValueError("At least 3 control units with complete data are needed.")
    cut = pd.Timestamp(intervention) if pd.api.types.is_datetime64_any_dtype(wide.index) else intervention
    pre, post = wide.index < cut, wide.index >= cut
    if pre.sum() < 5 or post.sum() < 2:
        raise ValueError("At least 5 periods before and 2 after the intervention are needed.")
    Y0 = wide[donor_names].to_numpy()
    w = _weights(wide[treated_unit].to_numpy()[pre], Y0[pre])
    synthetic = Y0 @ w
    gap = wide[treated_unit].to_numpy() - synthetic
    rmspe_pre = float(np.sqrt(np.mean(gap[pre] ** 2)))
    ratio = float(np.sqrt(np.mean(gap[post] ** 2)) / max(rmspe_pre, 1e-9))
    placebo_gaps, placebo_ratio = {}, []
    for d in donor_names:
        others = [c for c in donor_names if c != d]
        wd = _weights(wide[d].to_numpy()[pre], wide[others].to_numpy()[pre])
        g = wide[d].to_numpy() - wide[others].to_numpy() @ wd
        placebo_gaps[d] = g
        placebo_ratio.append(float(np.sqrt(np.mean(g[post] ** 2)) / max(np.sqrt(np.mean(g[pre] ** 2)), 1e-9)))
    p = float((1 + sum(r >= ratio for r in placebo_ratio)) / (1 + len(placebo_ratio)))
    average_effect = float(gap[post].mean())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wide.index, y=wide[treated_unit], name=str(treated_unit)))
    fig.add_trace(go.Scatter(x=wide.index, y=synthetic, name="Synthetic control", line=dict(dash="dash")))
    fig.add_vline(x=cut, line_dash="dot")
    fig.update_layout(title="The treated unit against its synthetic control")
    gap_fig = go.Figure()
    for d, g in placebo_gaps.items():
        gap_fig.add_trace(go.Scatter(x=wide.index, y=g, mode="lines", line=dict(color="rgba(150,150,150,0.4)", width=1), showlegend=False, hoverinfo="skip"))
    gap_fig.add_trace(go.Scatter(x=wide.index, y=gap, name=str(treated_unit), line=dict(width=3)))
    gap_fig.add_vline(x=cut, line_dash="dot")
    gap_fig.update_layout(title="Gap for the treated unit and for the placebos", yaxis_title="Gap")
    notes = []
    if rmspe_pre > 0.25 * wide[treated_unit].to_numpy()[pre].std():
        notes.append("The synthetic control does not follow the treated unit well before the intervention, so its gap afterwards is hard to trust.")
    weights = pd.Series(w, index=donor_names).sort_values(ascending=False)
    return {"effect": average_effect, "gap": pd.Series(gap, index=wide.index), "synthetic": pd.Series(synthetic, index=wide.index), "weights": weights[weights > 0.01],
            "p_value": p, "rmspe_ratio": ratio, "pre_fit": rmspe_pre, "placebos": len(placebo_ratio), "figure": fig, "gap_figure": gap_fig, "notes": notes,
            "summary": (f"After the intervention '{treated_unit}' is {average_effect:+.4g} per period against its synthetic control "
                        f"(built mostly from {', '.join(map(str, weights.index[:3]))}). Only {p:.0%} of the placebo units show a gap as large relative to their fit "
                        f"(placebo p = {p:.3f}).")}


# ── Variables instrumentales ─────────────────────────────────────────────────
def iv_2sls(df: pd.DataFrame, outcome: str, endogenous: str, instruments: list, covariates=None) -> dict:
    """Mínimos cuadrados en dos etapas. El instrumento debe mover el tratamiento (primera etapa) y afectar al resultado *solo* a través de él (supuesto que no se puede comprobar)."""
    if not instruments:
        raise ValueError("Give at least one instrument.")
    controls = [c for c in (covariates or []) if c not in (outcome, endogenous, *instruments)]
    cols = [outcome, endogenous, *instruments, *controls]
    for c in cols:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"'{c}' must be a numeric column.")
    data = df[cols].dropna().astype(float)
    if len(data) < 50:
        raise ValueError("At least 50 complete rows are needed.")
    y, d = data[outcome].to_numpy(), data[endogenous].to_numpy()
    Z = np.column_stack([np.ones(len(y)), data[controls].to_numpy() if controls else np.empty((len(y), 0)), data[instruments].to_numpy()])
    W = np.column_stack([np.ones(len(y)), data[controls].to_numpy() if controls else np.empty((len(y), 0))])
    first = np.linalg.lstsq(Z, d, rcond=None)[0]
    d_hat = Z @ first
    restricted = np.linalg.lstsq(W, d, rcond=None)[0]
    rss_full, rss_restricted = np.sum((d - d_hat) ** 2), np.sum((d - W @ restricted) ** 2)
    q = len(instruments)
    f_stat = float(((rss_restricted - rss_full) / q) / (rss_full / (len(y) - Z.shape[1])))
    X2 = np.column_stack([W, d_hat])
    beta = np.linalg.lstsq(X2, y, rcond=None)[0]
    X_actual = np.column_stack([W, d])
    resid = y - X_actual @ beta                                    # los residuos se calculan con la variable real, no con la ajustada
    bread = np.linalg.pinv(X2.T @ X2)
    meat = (X2 * resid[:, None]).T @ (X2 * resid[:, None])
    cov = bread @ meat @ bread * len(y) / (len(y) - X2.shape[1])
    se = float(np.sqrt(cov[-1, -1]))
    effect = float(beta[-1])
    ols = np.linalg.lstsq(X_actual, y, rcond=None)[0][-1]
    z = stats.norm.ppf(0.975)
    notes = []
    if f_stat < 10:
        notes.append(f"Weak instrument(s): the first-stage F is {f_stat:.1f} (below 10). The estimate can be badly biased and its range is unreliable.")
    if q == 1 and abs(effect - ols) < 0.05 * max(abs(ols), 1e-9):
        notes.append("IV and ordinary regression agree closely, so there is little sign of confounding that IV needed to fix.")
    return {"effect": effect, "se": se, "interval": (effect - z * se, effect + z * se), "ols": float(ols), "first_stage_f": f_stat, "p_value": float(2 * stats.norm.sf(abs(effect / se))) if se else 0.0,
            "n": len(y), "notes": notes,
            "summary": (f"Using {', '.join(instruments)} as instrument(s), a one-unit increase in '{endogenous}' changes '{outcome}' by {effect:+.4g} "
                        f"(95% range {effect - z * se:+.4g} to {effect + z * se:+.4g}); ordinary regression says {ols:+.4g}. First-stage F = {f_stat:.1f}. "
                        "This assumes the instrument affects the outcome only through the treatment.")}
