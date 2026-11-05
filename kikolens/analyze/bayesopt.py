"""Optimización bayesiana de hiperparámetros. En lugar de probar combinaciones al azar, ajusta un proceso gaussiano a los
resultados ya vistos y elige la siguiente combinación con la mayor mejora esperada (EI), equilibrando probar lo prometedor y
explorar lo desconocido. Con pocos intentos suele encontrar mejores ajustes que la búsqueda aleatoria."""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import cross_val_score


class Space:
    """El espacio de búsqueda en el cubo unidad. Entiende las distribuciones de SciPy que usa el laboratorio
    (loguniform, uniform, randint) y las listas de opciones."""

    def __init__(self, definition: dict):
        if not definition:
            raise ValueError("There is nothing to tune: the search space is empty.")
        self.names, self.specs = list(definition), []
        for name, spec in definition.items():
            if isinstance(spec, (list, tuple)):
                self.specs.append(("choice", list(spec)))
                continue
            dist = getattr(spec, "dist", None)
            kind = getattr(dist, "name", "")
            args = getattr(spec, "args", ())
            if kind in ("reciprocal", "loguniform"):
                self.specs.append(("log", float(args[0]), float(args[1])))
            elif kind == "uniform":
                self.specs.append(("lin", float(args[0]), float(args[0]) + float(args[1])))
            elif kind == "randint":
                self.specs.append(("int", int(args[0]), int(args[1]) - 1))
            else:
                raise ValueError(f"Cannot tune '{name}': unsupported search distribution.")

    @property
    def dimension(self) -> int:
        return len(self.names)

    def decode(self, unit) -> dict:
        out = {}
        for name, spec, u in zip(self.names, self.specs, unit):
            u = float(np.clip(u, 0, 1 - 1e-12))
            if spec[0] == "log":
                out[name] = float(np.exp(np.log(spec[1]) + u * (np.log(spec[2]) - np.log(spec[1]))))
            elif spec[0] == "lin":
                out[name] = float(spec[1] + u * (spec[2] - spec[1]))
            elif spec[0] == "int":
                out[name] = int(spec[1] + int(u * (spec[2] - spec[1] + 1)))
            else:
                out[name] = spec[1][int(u * len(spec[1]))]
        return out


def _expected_improvement(gp, candidates: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    mean, std = gp.predict(candidates, return_std=True)
    std = np.maximum(std, 1e-9)
    gain = mean - best - xi
    z = gain / std
    return gain * norm.cdf(z) + std * norm.pdf(z)


def optimize(objective, space: Space, n_iter: int = 20, n_initial: int | None = None, seed: int = 42, candidates: int = 2000) -> dict:
    """Maximiza `objective(params) -> float`. Devuelve el historial, el mejor punto y su puntuación."""
    if n_iter < 3:
        raise ValueError("At least 3 attempts are needed.")
    rng = np.random.default_rng(seed)
    n_initial = n_initial or max(3, n_iter // 4)
    n_initial = min(n_initial, n_iter)
    units, scores, rows = [], [], []

    def run(unit, phase):
        params = space.decode(unit)
        try:
            score = float(objective(params))
        except Exception:                                    # una combinación inválida no detiene la búsqueda
            score = np.nan
        units.append(np.asarray(unit, float))
        scores.append(score)
        rows.append({"attempt": len(rows) + 1, "phase": phase, "score": score,
                     "best so far": np.nanmax(scores) if np.isfinite(scores).any() else np.nan, **params})

    for unit in rng.random((n_initial, space.dimension)):
        run(unit, "explore")
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(space.dimension, 0.3), length_scale_bounds=(0.05, 5.0), nu=2.5) \
        + WhiteKernel(1e-4, (1e-8, 1e-1))
    for _ in range(n_iter - n_initial):
        valid = np.isfinite(scores)
        if valid.sum() < 2:
            run(rng.random(space.dimension), "explore")
            continue
        X, y = np.array(units)[valid], np.array(scores)[valid]
        spread = y.std() or 1.0
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=int(rng.integers(1 << 30)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X, (y - y.mean()) / spread)
        pool = rng.random((candidates, space.dimension))
        ei = _expected_improvement(gp, pool, float(((y - y.mean()) / spread).max()))
        run(pool[int(np.argmax(ei))], "exploit")
    history = pd.DataFrame(rows)
    if not np.isfinite(history["score"]).any():
        raise ValueError("No combination could be evaluated.")
    best = history.loc[history["score"].idxmax()]
    return {"history": history, "best_params": space.decode(units[int(best["attempt"]) - 1]), "best_score": float(best["score"]),
            "figure": history_figure(history)}


def history_figure(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for phase, part in history.groupby("phase"):
        fig.add_trace(go.Scatter(x=part["attempt"], y=part["score"], mode="markers", name=phase.title(), marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=history["attempt"], y=history["best so far"], mode="lines", name="Best so far", line=dict(shape="hv")))
    fig.update_layout(title="Each attempt and the best score so far", xaxis_title="Attempt", yaxis_title="Cross-validated score")
    return fig


def tune_bayes(pool_entry, make_pipeline, X, y, scoring: str, cv, n_iter: int = 20, seed: int = 42) -> dict:
    """Ajusta un modelo del laboratorio. `pool_entry` es (estimador, espacio con prefijo `model__`)."""
    estimator, definition = pool_entry
    space = Space(definition)
    base = make_pipeline(X, estimator)

    def objective(params):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.mean(cross_val_score(clone(base).set_params(**params), X, y, cv=cv, scoring=scoring, error_score="raise")))

    result = optimize(objective, space, n_iter=n_iter, seed=seed)
    result["pipeline"] = clone(base).set_params(**result["best_params"])
    return result
