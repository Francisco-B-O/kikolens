"""¿Qué distribución describe mejor una columna? Ajusta varias distribuciones habituales por máxima verosimilitud, las compara
con el criterio de Akaike (AIC) y con el test de Kolmogorov-Smirnov, y dibuja el histograma con la curva y el gráfico Q-Q de la
mejor. También dice si la columna se parece a una normal."""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

MIN_ROWS = 30
CANDIDATES = {"normal": stats.norm, "lognormal": stats.lognorm, "exponential": stats.expon, "gamma": stats.gamma,
              "Weibull": stats.weibull_min, "beta": stats.beta, "uniform": stats.uniform, "Laplace": stats.laplace,
              "Student t": stats.t}
POSITIVE = {"lognormal", "gamma", "Weibull"}
NON_NEGATIVE = {"exponential"}
UNIT = {"beta"}


def _usable(name: str, x: np.ndarray) -> bool:
    if name in POSITIVE:
        return x.min() > 0
    if name in NON_NEGATIVE:
        return x.min() >= 0
    if name in UNIT:
        return x.min() >= 0 and x.max() <= 1
    return True


def fit_distributions(series: pd.Series, names=None, top_note: int = 3) -> dict:
    """Tabla de ajustes ordenada por AIC (menor es mejor) y la mejor distribución con sus parámetros."""
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    x = x[np.isfinite(x)]
    if len(x) < MIN_ROWS:
        raise ValueError(f"At least {MIN_ROWS} numbers are needed; there are {len(x)}.")
    if np.ptp(x) == 0:
        raise ValueError("All the values are the same; there is nothing to fit.")
    names = list(names) if names else list(CANDIDATES)
    unknown = [n for n in names if n not in CANDIDATES]
    if unknown:
        raise ValueError(f"Unknown distribution(s): {', '.join(unknown)}. Choose from {', '.join(CANDIDATES)}.")
    rows, params = [], {}
    for name in names:
        if not _usable(name, x):
            continue
        dist = CANDIDATES[name]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = dist.fit(x, floc=0) if name in POSITIVE | NON_NEGATIVE else dist.fit(x)
                ll = float(np.sum(dist.logpdf(x, *fitted)))
                ks = stats.kstest(x, dist.cdf, args=fitted)
        except Exception:
            continue
        if not np.isfinite(ll):
            continue
        k = len(fitted) - (1 if name in POSITIVE | NON_NEGATIVE else 0)    # el parámetro de posición fijado no cuenta
        params[name] = fitted
        rows.append({"distribution": name, "AIC": 2 * k - 2 * ll, "log-likelihood": ll, "KS statistic": float(ks.statistic),
                     "KS p-value": float(ks.pvalue), "parameters": k})
    if not rows:
        raise ValueError("None of the distributions could be fitted to these values.")
    table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    table["AIC difference"] = table["AIC"] - table["AIC"].iloc[0]
    best = table["distribution"].iloc[0]
    sample = x[:5000]
    normal_p = float(stats.shapiro(sample).pvalue) if len(x) >= 8 else np.nan
    skew, kurt = float(stats.skew(x)), float(stats.kurtosis(x))
    close = table[table["AIC difference"] < 2]["distribution"].tolist()
    return {"table": table, "best": best, "parameters": tuple(float(v) for v in params[best]), "fitted": params,
            "normality_p": normal_p, "skewness": skew, "excess_kurtosis": kurt, "n": len(x), "figure": figure(x, best, params[best]),
            "qq": qq_figure(x, best, params[best]),
            "summary": (f"{best} fits best (AIC {table['AIC'].iloc[0]:,.1f})"
                        + (f"; {', '.join(c for c in close if c != best)} fit almost as well" if len(close) > 1 else "")
                        + f". Skewness {skew:+.2f}, excess kurtosis {kurt:+.2f}. "
                        + ("The values look close to normal." if normal_p > 0.05 else
                           f"The values are not normal (Shapiro-Wilk p = {normal_p:.2g})."))}


def figure(x: np.ndarray, name: str, params) -> go.Figure:
    dist = CANDIDATES[name]
    grid = np.linspace(x.min(), x.max(), 200)
    fig = go.Figure(go.Histogram(x=x, histnorm="probability density", nbinsx=40, name="Data", opacity=0.6))
    fig.add_trace(go.Scatter(x=grid, y=dist.pdf(grid, *params), mode="lines", name=name))
    fig.update_layout(title=f"Data and the fitted {name} curve")
    return fig


def qq_figure(x: np.ndarray, name: str, params) -> go.Figure:
    dist = CANDIDATES[name]
    probs = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    theory = dist.ppf(probs, *params)
    ordered = np.sort(x)
    fig = go.Figure(go.Scatter(x=theory, y=ordered, mode="markers", name="Values"))
    lo, hi = float(min(theory.min(), ordered.min())), float(max(theory.max(), ordered.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect fit", line=dict(dash="dash")))
    fig.update_layout(title=f"Q-Q plot against the {name} (points on the line mean a good fit)",
                      xaxis_title="Expected", yaxis_title="Observed")
    return fig
