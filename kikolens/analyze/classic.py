"""Series temporales clásicas, sin dependencias nuevas.

  * `stationarity`: contrastes ADF y KPSS juntos (cada uno por sí solo confunde) y cuántas diferencias hacen falta.
  * `acf_pacf`: autocorrelaciones con bandas y la prueba de Ljung-Box.
  * `stl`: descomposición robusta tendencia + estacionalidad + resto (Cleveland et al., 1990), con la fuerza de cada componente.
  * `arima` / `auto_arima`: modelos ARIMA estacionales ajustados por mínimos cuadrados condicionales, con intervalos de predicción, diagnóstico de los
    residuos y una comparación honesta con la previsión ingenua estacional en tramos que el modelo no vio.
Cada resultado dice qué supone y cuándo no fiarse."""
import itertools
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import lfilter

MAX_POINTS = 20_000
MIN_POINTS = 20
_MACKINNON = {
    "c": dict(star=-1.61, top=2.74, bottom=-18.83, small=[2.1659, 1.4412, 0.038269], large=[1.7339, 0.93202, -0.12745, -0.010368]),
    "ct": dict(star=-2.89, top=0.70, bottom=-16.18, small=[3.2512, 1.6047, 0.049588], large=[2.5261, 0.61654, -0.37956, -0.060285]),
}
_KPSS = {"c": ([0.347, 0.463, 0.574, 0.739], [0.10, 0.05, 0.025, 0.01]), "ct": ([0.119, 0.146, 0.176, 0.216], [0.10, 0.05, 0.025, 0.01])}


def _values(series, minimum: int = MIN_POINTS) -> np.ndarray:
    y = np.asarray(pd.Series(series).dropna(), dtype=float)
    if len(y) < minimum:
        raise ValueError(f"At least {minimum} observations are needed (found {len(y)}).")
    if len(y) > MAX_POINTS:
        raise ValueError(f"At most {MAX_POINTS:,} observations: aggregate the series to a coarser period first.")
    if not np.isfinite(y).all():
        raise ValueError("The series has infinite values.")
    if np.ptp(y) == 0:
        raise ValueError("The series is constant.")
    return y


# ── Estacionariedad ──────────────────────────────────────────────────────────
def _adf_p(tau: float, regression: str) -> float:
    t = _MACKINNON[regression]
    if tau > t["top"]:
        return 1.0
    if tau < t["bottom"]:
        return 0.0
    coefficients = t["small"] if tau <= t["star"] else t["large"]
    return float(stats.norm.cdf(np.polyval(coefficients[::-1], tau)))


def adf(series, regression: str = "c", max_lag: int | None = None) -> dict:
    """Dickey-Fuller aumentado. H0: la serie tiene raíz unitaria (no es estacionaria). Número de retardos por AIC; p-valor por la aproximación de MacKinnon."""
    if regression not in _MACKINNON:
        raise ValueError("regression must be 'c' (constant) or 'ct' (constant and trend).")
    y = _values(series)
    n = len(y)
    dy = np.diff(y)
    top = int(max_lag if max_lag is not None else min(int(12 * (n / 100) ** 0.25), n // 3 - 2))
    top = max(top, 0)

    def fit(k, start):
        rows = np.arange(start, len(dy))
        cols = [y[rows], *[dy[rows - i] for i in range(1, k + 1)]]
        X = np.column_stack([*cols, np.ones(len(rows))] + ([np.arange(len(rows), dtype=float)] if regression == "ct" else []))
        target = dy[rows]
        beta, *_ = np.linalg.lstsq(X, target, rcond=None)
        resid = target - X @ beta
        sigma2 = float(resid @ resid) / (len(rows) - X.shape[1])
        cov = np.linalg.pinv(X.T @ X) * sigma2
        aic = len(rows) * np.log(float(resid @ resid) / len(rows)) + 2 * X.shape[1]
        return float(beta[0] / np.sqrt(cov[0, 0])), aic, len(rows)

    best = min(range(top + 1), key=lambda k: fit(k, top)[1])                 # los candidatos se comparan sobre la misma muestra
    tau, _, used = fit(best, best)
    crit = {f"{int(level * 100)}%": _critical(level, regression) for level in (0.01, 0.05, 0.10)}
    p = _adf_p(tau, regression)
    return {"statistic": tau, "p_value": p, "lags": best, "observations": used, "critical_values": crit, "regression": regression, "stationary": p < 0.05}


def _critical(level: float, regression: str) -> float:
    """Valor crítico del ADF para ese nivel (invierte la aproximación de MacKinnon)."""
    lo, hi = _MACKINNON[regression]["bottom"] + 1e-6, _MACKINNON[regression]["top"]
    for _ in range(80):
        mid = (lo + hi) / 2
        if _adf_p(mid, regression) < level:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def kpss(series, regression: str = "c", lags: int | None = None) -> dict:
    """KPSS. H0: la serie es estacionaria (alrededor de un nivel o de una tendencia). Es el complemento del ADF."""
    if regression not in _KPSS:
        raise ValueError("regression must be 'c' (level) or 'ct' (trend).")
    y = _values(series)
    n = len(y)
    t = np.arange(n, dtype=float)
    X = np.column_stack([np.ones(n)] + ([t] if regression == "ct" else []))
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    lags = int(lags if lags is not None else np.ceil(12 * (n / 100) ** 0.25))
    lags = min(lags, n - 1)
    s2 = float(resid @ resid) / n
    for k in range(1, lags + 1):
        s2 += 2 * (1 - k / (lags + 1)) * float(resid[k:] @ resid[:-k]) / n
    stat = float(np.sum(np.cumsum(resid) ** 2) / (n ** 2 * s2))
    values, ps = _KPSS[regression]
    p = float(np.interp(stat, values, ps, left=0.10, right=0.01))
    return {"statistic": stat, "p_value": p, "lags": lags, "regression": regression, "stationary": p > 0.05, "bounded": stat < values[0] or stat > values[-1],
            "critical_values": {f"{int(q * 100)}%": v for v, q in zip(values, ps) if q in (0.10, 0.05, 0.01)}}


def stationarity(series, regression: str = "c", max_diff: int = 2) -> dict:
    """ADF y KPSS sobre la serie y sobre sus diferencias, con una conclusión que reúne los dos (si no coinciden, lo dice) y las diferencias sugeridas."""
    y = _values(series)
    steps, current, suggested = [], y, None
    for d in range(max_diff + 1):
        a, k = adf(current, regression), kpss(current, regression)
        steps.append({"differences": d, "adf statistic": a["statistic"], "adf p": a["p_value"], "kpss statistic": k["statistic"], "kpss p": k["p_value"],
                      "verdict": _verdict(a, k)})
        if suggested is None and a["stationary"] and k["stationary"]:
            suggested = d
        if d < max_diff:
            current = np.diff(current)
            if len(current) < MIN_POINTS or np.ptp(current) == 0:
                break
    if suggested is None:
        suggested = next((s["differences"] for s in steps if s["verdict"].startswith("stationary")), min(len(steps) - 1, max_diff))
    base = steps[0]
    notes = []
    if base["verdict"].startswith("inconclusive"):
        notes.append("ADF and KPSS disagree in a way that points to a mix of trend and structural change: look at the series and consider a trend in the model.")
    if len(y) < 50:
        notes.append("Fewer than 50 observations: these tests have little power and can miss a unit root.")
    table = pd.DataFrame(steps)
    return {"table": table, "differences": int(suggested), "adf": adf(y, regression), "kpss": kpss(y, regression), "verdict": base["verdict"], "notes": notes,
            "summary": f"{base['verdict'].capitalize()} (ADF p = {base['adf p']:.3f}, KPSS p {'= ' if 0.01 < base['kpss p'] < 0.10 else ('> ' if base['kpss p'] >= 0.10 else '< ')}"
                       f"{base['kpss p']:.2f}). Suggested differences: {suggested}."}


def _verdict(a: dict, k: dict) -> str:
    if a["stationary"] and k["stationary"]:
        return "stationary (both tests agree)"
    if not a["stationary"] and not k["stationary"]:
        return "not stationary (both tests agree): it needs differencing"
    if a["stationary"] and not k["stationary"]:
        return "inconclusive: ADF says stationary, KPSS says not (a trend or a break may be present)"
    return "inconclusive: ADF cannot reject a unit root but KPSS cannot reject stationarity (too little data to tell)"


# ── Autocorrelación ──────────────────────────────────────────────────────────
def acf(y, nlags: int) -> np.ndarray:
    y = np.asarray(y, dtype=float) - np.mean(y)
    n = len(y)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, size)
    full = np.fft.irfft(f * np.conj(f), size)[: nlags + 1]
    return full / full[0]


def pacf(y, nlags: int) -> np.ndarray:
    """Autocorrelación parcial por Durbin-Levinson."""
    r = acf(y, nlags)
    out, phi = [1.0], np.zeros(nlags + 1)
    for k in range(1, nlags + 1):
        value = (r[k] - phi[1:k] @ r[1:k][::-1]) / (1 - phi[1:k] @ r[1:k]) if k > 1 else r[1]
        phi_new = phi.copy()
        phi_new[k] = value
        phi_new[1:k] = phi[1:k] - value * phi[1:k][::-1]
        phi = phi_new
        out.append(float(value))
    return np.array(out)


def ljung_box(residuals, lags: int, fitted_params: int = 0) -> dict:
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    lags = max(1, min(lags, n // 2))
    rho = acf(r, lags)[1:]
    q = float(n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1))))
    dof = max(lags - fitted_params, 1)
    return {"statistic": q, "p_value": float(stats.chi2.sf(q, dof)), "lags": lags, "dof": dof}


def acf_pacf(series, nlags: int | None = None) -> dict:
    y = _values(series)
    nlags = int(min(nlags or max(10, min(40, len(y) // 4)), len(y) // 2 - 1))
    a, p = acf(y, nlags), pacf(y, nlags)
    band = float(1.96 / np.sqrt(len(y)))
    lb = ljung_box(y, min(10, nlags))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, nlags + 1)), y=a[1:], name="Autocorrelation"))
    fig.add_trace(go.Scatter(x=list(range(1, nlags + 1)), y=p[1:], mode="markers+lines", name="Partial autocorrelation"))
    fig.add_hrect(y0=-band, y1=band, opacity=0.1, line_width=0)
    fig.update_layout(title="Autocorrelation (bars) and partial autocorrelation (line); the band is what noise would give", xaxis_title="Lag")
    notes = []
    if lb["p_value"] < 0.05:
        notes.append("The series is clearly autocorrelated (Ljung-Box p < 0.05): past values carry information about the future.")
    return {"acf": pd.Series(a), "pacf": pd.Series(p), "band": band, "ljung_box": lb, "figure": fig, "notes": notes,
            "table": pd.DataFrame({"lag": range(nlags + 1), "autocorrelation": a, "partial": p}),
            "summary": f"Largest autocorrelation beyond lag 0 is {a[1:][np.abs(a[1:]).argmax()]:+.2f} at lag {int(np.abs(a[1:]).argmax()) + 1} (noise band ±{band:.2f}); "
                       f"Ljung-Box p = {lb['p_value']:.3f}."}


# ── STL ──────────────────────────────────────────────────────────────────────
def _loess(x, y, xout, span: int, w=None) -> np.ndarray:
    """Regresión local lineal con pesos tricúbicos sobre los `span` puntos más cercanos."""
    n = len(x)
    span = max(2, min(int(span), n)) if n >= 2 else 1
    out = np.empty(len(xout))
    for i, x0 in enumerate(xout):
        d = np.abs(x - x0)
        idx = np.argpartition(d, span - 1)[:span] if span < n else np.arange(n)
        h = d[idx].max()
        if span > n:
            h *= span / n
        h = max(h, 1e-12)
        u = d[idx] / (h * 1.000001)
        wt = np.where(u < 1, (1 - u ** 3) ** 3, 0.0)
        if w is not None:
            wt = wt * w[idx]
        sw = wt.sum()
        if sw <= 0:
            out[i] = y[idx].mean()
            continue
        xs, ys = x[idx], y[idx]
        mx, my = (wt @ xs) / sw, (wt @ ys) / sw
        var = wt @ (xs - mx) ** 2
        slope = (wt @ ((xs - mx) * (ys - my))) / var if var > 1e-12 * max(h, 1) ** 2 else 0.0
        out[i] = my + slope * (x0 - mx)
    return out


def _odd(value: float) -> int:
    value = int(np.ceil(value))
    return value if value % 2 else value + 1


def _moving(y, k: int) -> np.ndarray:
    return np.convolve(y, np.ones(k) / k, mode="valid")


def stl_decompose(y: np.ndarray, period: int, seasonal: int = 7, trend: int | None = None, robust: bool = True, inner: int = 2, outer: int | None = None) -> tuple:
    """STL con los pasos de Cleveland: suavizado de las subseries estacionales, filtro paso bajo y suavizado de la tendencia; con pesos de robustez opcionales."""
    n = len(y)
    seasonal = _odd(max(seasonal, 3))
    low = _odd(period + 1)
    trend = trend or _odd(1.5 * period / (1 - 1.5 / seasonal))
    outer = (15 if robust else 0) if outer is None else outer
    t_comp, s_comp, rho = np.zeros(n), np.zeros(n), np.ones(n)
    index = np.arange(n, dtype=float)
    for round_ in range(outer + 1):
        for _ in range(inner):
            detrended = y - t_comp
            cycle = np.zeros(n + 2 * period)
            for j in range(period):
                positions = np.arange(j, n, period)
                m = len(positions)
                sub = _loess(np.arange(m, dtype=float), detrended[positions], np.arange(-1, m + 1, dtype=float), seasonal, rho[positions])
                cycle[positions + period] = sub[1:-1]
                cycle[j] = sub[0]                                    # una posición más allá de cada extremo de la subserie
                cycle[j + (m + 1) * period] = sub[-1]
            filtered = _moving(_moving(_moving(cycle, period), period), 3)
            lowpass = _loess(index, filtered[:n], index, low)
            s_comp = cycle[period: period + n] - lowpass
            t_comp = _loess(index, y - s_comp, index, trend, rho)
        resid = y - t_comp - s_comp
        if round_ < outer:
            h = 6 * np.median(np.abs(resid))
            u = np.abs(resid) / h if h > 0 else np.zeros(n)
            rho = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
    return t_comp, s_comp, y - t_comp - s_comp


def stl(series, period: int | None = None, robust: bool = True, seasonal: int = 7) -> dict:
    """Descompone en tendencia, estacionalidad y resto. `period` se detecta si no se da. Con `robust` los valores extremos no arrastran a la tendencia."""
    from kikolens.analyze import timeseries as ts

    s = pd.Series(series).dropna()
    y = _values(s, 4)
    if period is None:
        period = ts.guess_period(s.astype(float))
    if not period or period < 2:
        raise ValueError("No seasonal period was found; give one (for example 7 for daily data with a weekly pattern).")
    if len(y) < 2 * period:
        raise ValueError(f"At least two full periods ({2 * period} observations) are needed.")
    trend_c, seasonal_c, resid = stl_decompose(y, int(period), seasonal=seasonal, robust=robust)
    centre = float(np.median(resid))
    sigma = float(np.median(np.abs(resid - centre)) * 1.4826)
    cut = 4 * sigma
    clipped = np.clip(resid, centre - cut, centre + cut) if sigma > 0 else resid            # un valor extremo no debe hundir la fuerza de los componentes
    var = lambda a: float(np.var(a))                                                       # noqa: E731
    strength_t = max(0.0, 1 - var(clipped) / max(var(trend_c + clipped), 1e-12))
    strength_s = max(0.0, 1 - var(clipped) / max(var(seasonal_c + clipped), 1e-12))
    idx = s.index
    parts = pd.DataFrame({"observed": y, "trend": trend_c, "seasonal": seasonal_c, "remainder": resid}, index=idx)
    unusual = parts[np.abs(parts["remainder"] - centre) > max(cut, 1e-12)]
    notes = []
    if strength_s < 0.5:
        notes.append("The seasonal pattern is weak (strength below 0.5; even pure noise scores about 0.4): a seasonal model may add noise rather than accuracy.")
    if len(unusual) > 0.05 * len(y):
        notes.append("More than 5% of the points are far from trend and season: the period may be wrong or the series has several regimes.")
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("Observed", "Trend", "Seasonal", "Remainder"))
    for row, name in enumerate(("observed", "trend", "seasonal", "remainder"), start=1):
        fig.add_trace(go.Scatter(x=idx, y=parts[name], mode="lines", showlegend=False), row=row, col=1)
    fig.update_layout(height=640, title=f"STL decomposition (period {period})")
    return {"table": parts, "components": parts, "period": int(period), "trend_strength": strength_t, "seasonal_strength": strength_s, "unusual": unusual, "robust": robust,
            "notes": notes, "figure": fig,
            "summary": f"Period {period}: trend strength {strength_t:.2f}, seasonal strength {strength_s:.2f} (0 is none, 1 is all of the variation); "
                       f"{len(unusual)} point(s) far from trend and season."}


# ── ARIMA ────────────────────────────────────────────────────────────────────
def _pacf_to_coefficients(raw: np.ndarray) -> np.ndarray:
    """Coeficientes AR que garantizan estacionariedad (o MA invertibles) a partir de parámetros libres."""
    r = np.tanh(raw)
    phi = np.zeros(0)
    for k, rk in enumerate(r, start=1):
        phi = np.concatenate([phi - rk * phi[::-1], [rk]]) if k > 1 else np.array([rk])
    return phi


def _poly(coef: np.ndarray, sign: float, period: int = 1) -> np.ndarray:
    out = np.zeros(len(coef) * period + 1)
    out[0] = 1.0
    out[period::period][: len(coef)] = sign * coef
    return out


def _difference_poly(d: int, D: int, m: int) -> np.ndarray:
    poly = np.array([1.0])
    for _ in range(d):
        poly = np.convolve(poly, [1.0, -1.0])
    for _ in range(D):
        seasonal = np.zeros(m + 1)
        seasonal[0], seasonal[m] = 1.0, -1.0
        poly = np.convolve(poly, seasonal)
    return poly


class _Spec:
    def __init__(self, order, seasonal, m):
        (self.p, self.d, self.q), (self.P, self.D, self.Q) = order, seasonal
        self.m = m
        self.mean = (self.d + self.D) == 0
        self.k = self.p + self.q + self.P + self.Q + (1 if self.mean else 0)

    def unpack(self, theta: np.ndarray) -> dict:
        i = 0
        parts = {}
        for name, size in (("ar", self.p), ("ma", self.q), ("sar", self.P), ("sma", self.Q)):
            parts[name] = _pacf_to_coefficients(theta[i: i + size]) if size else np.zeros(0)
            i += size
        parts["mu"] = float(theta[i]) if self.mean else 0.0
        return parts

    def polynomials(self, parts: dict) -> tuple:
        ar = np.convolve(_poly(parts["ar"], -1.0), _poly(parts["sar"], -1.0, self.m))
        ma = np.convolve(_poly(-parts["ma"], -1.0), _poly(-parts["sma"], -1.0, self.m))      # el MA entra con signo + : 1 + θ1 B + ...
        full_ar = np.convolve(ar, _difference_poly(self.d, self.D, self.m))
        return full_ar, ma


def _residuals(spec: _Spec, parts: dict, y: np.ndarray) -> tuple:
    A, B = spec.polynomials(parts)
    e = lfilter(A, B, y - parts["mu"])
    start = len(A) - 1
    return e, start


def _fit(spec: _Spec, y: np.ndarray, start_values=None, common_start: int = 0) -> dict:
    def sse(theta):
        parts = spec.unpack(theta)
        e, start = _residuals(spec, parts, y)
        value = float(e[start:] @ e[start:])
        return value if np.isfinite(value) else 1e30

    n_eff = len(y) - (len(_difference_poly(spec.d, spec.D, spec.m)) - 1) - max(spec.p + spec.P * spec.m, 0)
    scale = max(float(np.var(np.diff(y, spec.d))) if spec.d else float(np.var(y)), 1e-12)
    x0 = np.zeros(spec.k) if start_values is None else start_values
    if spec.mean:
        x0 = x0.copy()
        x0[-1] = float(np.mean(y)) if start_values is None else x0[-1]
    if spec.k:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(lambda t: sse(t) / scale / max(n_eff, 1), x0, method="BFGS", options={"maxiter": 200, "gtol": 1e-7})
            theta = res.x
    else:
        theta = x0
    parts = spec.unpack(theta)
    e, start = _residuals(spec, parts, y)
    usable = e[max(start, common_start):]                       # los criterios de información se calculan sobre la misma muestra para poder compararlos
    n_used = len(usable)
    sigma2 = float(usable @ usable) / max(n_used, 1)
    loglik = -0.5 * n_used * (np.log(2 * np.pi * max(sigma2, 1e-300)) + 1)
    k = spec.k + 1
    aic = -2 * loglik + 2 * k
    aicc = aic + 2 * k * (k + 1) / max(n_used - k - 1, 1)
    return {"theta": theta, "parts": parts, "residuals": e, "start": start, "sigma2": sigma2, "loglik": float(loglik), "aic": float(aic), "aicc": float(aicc), "n_used": n_used}


def _forecast(spec: _Spec, fit: dict, y: np.ndarray, horizon: int, level: float) -> pd.DataFrame:
    A, B = spec.polynomials(fit["parts"])
    mu = fit["parts"]["mu"]
    la, lb = len(A) - 1, len(B) - 1
    yy = list(y - mu)
    ee = list(fit["residuals"])
    for _ in range(horizon):
        value = -sum(A[k] * yy[-k] for k in range(1, la + 1) if len(yy) >= k) + sum(B[j] * ee[-j] for j in range(1, lb + 1) if len(ee) >= j)
        yy.append(value)
        ee.append(0.0)
    mean = np.array(yy[len(y):]) + mu
    impulse = np.zeros(horizon)
    impulse[0] = 1.0
    psi = lfilter(B, A, impulse)
    se = np.sqrt(fit["sigma2"] * np.cumsum(psi ** 2))
    z = stats.norm.ppf(0.5 + level / 2)
    return pd.DataFrame({"forecast": mean, "low": mean - z * se, "high": mean + z * se})


def arima(series, order=(1, 1, 1), seasonal=(0, 0, 0), period: int | None = None, horizon: int = 12, level: float = 0.95, validate: bool = True) -> dict:
    """Ajusta un ARIMA(p,d,q)(P,D,Q)[m] por mínimos cuadrados condicionales y prevé `horizon` pasos con intervalos."""
    s = pd.Series(series).dropna()
    y = _values(s, 30)
    p, d, q = order
    P, D, Q = seasonal
    if min(p, d, q, P, D, Q) < 0 or max(p, q, P, Q) > 5 or d > 2 or D > 1:
        raise ValueError("Orders must be 0 or more, with p, q, P, Q at most 5, d at most 2 and D at most 1.")
    if (P or D or Q) and (not period or period < 2):
        raise ValueError("A seasonal model needs the seasonal period (for example 12 for monthly data).")
    if not 0 < level < 1 or not 1 <= horizon <= 5000:
        raise ValueError("level must be between 0 and 1, and horizon between 1 and 5000.")
    spec = _Spec((p, d, q), (P, D, Q), period or 1)
    if len(y) < 3 * (p + q + (P + Q) * spec.m + d + D * spec.m) + 10:
        raise ValueError("The series is too short for this order; use a simpler model.")
    fit = _fit(spec, y)
    freq = pd.infer_freq(s.index) if isinstance(s.index, pd.DatetimeIndex) else None
    forecast = _forecast(spec, fit, y, horizon, level)
    if isinstance(s.index, pd.DatetimeIndex):
        step = pd.tseries.frequencies.to_offset(freq) if freq else (s.index[-1] - s.index[-2])
        forecast.index = pd.date_range(s.index[-1] + step, periods=horizon, freq=step) if freq else [s.index[-1] + step * (i + 1) for i in range(horizon)]
    resid = pd.Series(fit["residuals"][fit["start"]:], index=s.index[fit["start"]:])
    lb = ljung_box(resid, min(20, max(5, len(resid) // 5)), spec.p + spec.q + spec.P + spec.Q)
    notes = []
    if lb["p_value"] < 0.05:
        notes.append(f"The residuals are still autocorrelated (Ljung-Box p = {lb['p_value']:.3f}): the model misses some structure; try more terms or a seasonal part.")
    for name in ("ar", "sar"):
        if len(fit["parts"][name]) and np.abs(fit["parts"][name]).sum() > 0.98 * len(fit["parts"][name]):
            notes.append("An autoregressive term is at the edge of stationarity: the series may need one more difference.")
    skill = _backtest(spec, y, horizon) if validate else None
    if skill and skill["model_mae"] > skill["naive_mae"]:
        notes.append(f"On stretches the model did not see, a repeat-the-last-season forecast was more accurate (error {skill['naive_mae']:.4g} against {skill['model_mae']:.4g}): "
                     "do not trust this model over the simple one.")
    label = f"ARIMA({p},{d},{q})" + (f"({P},{D},{Q})[{period}]" if (P or D or Q) else "")
    fig = go.Figure()
    shown = s.iloc[-min(len(s), 4 * horizon + 50):]
    fig.add_trace(go.Scatter(x=shown.index, y=shown.values, name="Observed"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["high"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["low"], mode="lines", line=dict(width=0), fill="tonexty", name=f"{level:.0%} range", opacity=0.3))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["forecast"], name=label, line=dict(dash="dash")))
    fig.update_layout(title=f"{label}: forecast with its range")
    return {"label": label, "order": (p, d, q), "seasonal": (P, D, Q), "period": period, "forecast": forecast, "table": forecast, "residuals": resid, "aicc": fit["aicc"], "aic": fit["aic"],
            "sigma": float(np.sqrt(fit["sigma2"])), "ljung_box": lb, "coefficients": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in fit["parts"].items()}, "backtest": skill,
            "notes": notes, "figure": fig, "_spec": spec, "_fit": fit,
            "summary": (f"{label}, AICc {fit['aicc']:.1f}. Next {horizon}: {forecast['forecast'].iloc[0]:,.4g} rising to {forecast['forecast'].iloc[-1]:,.4g}, "
                        f"with a {level:.0%} range of {forecast['low'].iloc[-1]:,.4g} to {forecast['high'].iloc[-1]:,.4g} at the end."
                        + (f" On unseen stretches its error was {skill['model_mae']:.4g} against {skill['naive_mae']:.4g} for repeating the last season." if skill else ""))}


def _backtest(spec: _Spec, y: np.ndarray, horizon: int, folds: int = 3) -> dict | None:
    """Origen móvil: ajusta con el principio y prevé el tramo siguiente. Compara con repetir la última estación (o el último valor)."""
    h = int(min(horizon, max(len(y) // 8, 1)))
    need = len(y) - folds * h
    if need < 3 * spec.k + 20:
        return None
    model_err, naive_err = [], []
    last = None
    for i in range(folds):
        cut = need + i * h
        train, test = y[:cut], y[cut: cut + h]
        try:
            fit = _fit(spec, train, last)
            last = fit["theta"]
            pred = _forecast(spec, fit, train, len(test), 0.95)["forecast"].to_numpy()
        except (ValueError, np.linalg.LinAlgError):
            return None
        if not np.isfinite(pred).all():
            return None
        m = spec.m if spec.m > 1 and len(train) > spec.m else 1
        naive = np.array([train[-m + (j % m)] for j in range(len(test))]) if m > 1 else np.full(len(test), train[-1])
        model_err.append(np.mean(np.abs(test - pred)))
        naive_err.append(np.mean(np.abs(test - naive)))
    return {"model_mae": float(np.mean(model_err)), "naive_mae": float(np.mean(naive_err)), "folds": folds, "horizon": h}


def _seasonal_strength(y: np.ndarray, period: int) -> float:
    try:
        return stl(pd.Series(y), period, robust=False)["seasonal_strength"]
    except ValueError:
        return 0.0


def auto_arima(series, period: int | None = None, horizon: int = 12, max_p: int = 3, max_q: int = 3, seasonal: bool | None = None, level: float = 0.95) -> dict:
    """Elige las diferencias con KPSS (y la estacional por la fuerza de la estacionalidad), prueba combinaciones pequeñas de p, q, P, Q y se queda con la
    de menor AICc. Con series largas busca sobre los últimos 1500 puntos."""
    from kikolens.analyze import timeseries as ts

    s = pd.Series(series).dropna()
    y = _values(s, 30)
    if period is None and isinstance(s.index, pd.DatetimeIndex):
        guessed = ts.guess_period(s.astype(float))
        period = int(guessed) if guessed and guessed >= 2 else None
    use_seasonal = bool(period and period >= 2 and len(y) >= 3 * period) if seasonal is None else bool(seasonal and period and period >= 2)
    window = y[-1500:]
    D = 1 if use_seasonal and _seasonal_strength(window, period) > 0.64 else 0
    work = window[period:] - window[:-period] if D else window
    d = 0
    current = work
    while d < 2 and len(current) > 25 and np.ptp(current) > 0 and not kpss(current)["stationary"]:
        current = np.diff(current)
        d += 1
    common = max_p + (period if use_seasonal else 0) + d + D * (period or 0)
    rows, best = [], None
    grid = [(p, q) for p, q in itertools.product(range(max_p + 1), range(max_q + 1)) if p + q <= 4]
    seasonal_grid = list(itertools.product((0, 1), (0, 1))) if use_seasonal else [(0, 0)]
    for (p, q), (P, Q) in itertools.product(grid, seasonal_grid):
        spec = _Spec((p, d, q), (P, D, Q), period or 1)
        if len(window) < 3 * (p + q + (P + Q) * spec.m + d + D * spec.m) + 10:
            continue
        try:
            fit = _fit(spec, window, common_start=common)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.isfinite(fit["aicc"]):
            continue
        rows.append({"model": f"ARIMA({p},{d},{q})" + (f"({P},{D},{Q})[{period}]" if use_seasonal else ""), "p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "AICc": fit["aicc"]})
        if best is None or fit["aicc"] < best[0]:
            best = (fit["aicc"], (p, d, q), (P, D, Q))
    if best is None:
        raise ValueError("No model could be fitted: the series may be too short or too irregular.")
    table = pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)
    table["difference to best"] = table["AICc"] - table["AICc"].iloc[0]
    result = arima(s, best[1], best[2], period if use_seasonal else None, horizon, level)
    result["candidates"] = table
    result["table"] = table
    result["forecast_table"] = result["forecast"]
    runner_up = table.iloc[1]["difference to best"] if len(table) > 1 else np.inf
    if runner_up < 2:
        result["notes"].append(f"{(table['difference to best'] < 2).sum() - 1} other model(s) are within 2 AICc points of this one: the data cannot tell them apart.")
    result["summary"] = f"Chosen from {len(table)} candidates: " + result["summary"]
    return result
