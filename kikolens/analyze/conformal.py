"""Predicción conforme: intervalos (regresión) y conjuntos de clases (clasificación) con una cobertura garantizada
en muestras finitas, sin suponer que los errores sean normales. Se calibra con datos que el modelo no vio al entrenar."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _quantile(scores: np.ndarray, coverage: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


class ConformalModel:
    """Envuelve un pipeline ya ajustado. `coverage` es la parte de casos nuevos que debe quedar dentro."""

    def __init__(self, pipe, task: str, coverage: float = 0.9):
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'.")
        if not 0.5 <= coverage < 1:
            raise ValueError("coverage must be between 0.5 and 1.")
        if task == "classification" and not hasattr(pipe, "predict_proba"):
            raise ValueError("This model cannot give class probabilities, so it cannot build prediction sets.")
        self.pipe, self.task, self.coverage = pipe, task, coverage
        self.threshold = None
        self.n_calibration = 0

    def calibrate(self, X: pd.DataFrame, y: pd.Series) -> "ConformalModel":
        if len(X) < 20:
            raise ValueError("At least 20 rows are needed to calibrate (the more, the tighter the result).")
        if self.task == "regression":
            scores = np.abs(np.asarray(y, float) - self.pipe.predict(X))
        else:
            index = {c: i for i, c in enumerate(self.pipe.classes_)}
            proba = self.pipe.predict_proba(X)
            true = np.array([index.get(v, -1) for v in y])
            if (true < 0).any():
                raise ValueError("The calibration data has classes the model never saw.")
            scores = 1 - proba[np.arange(len(true)), true]
        self.threshold = _quantile(scores, self.coverage)
        self.n_calibration = len(scores)
        return self

    def _need(self):
        if self.threshold is None:
            raise ValueError("Calibrate the model first.")

    def predict_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        self._need()
        if self.task != "regression":
            raise ValueError("Intervals are for regression; use predict_set for classification.")
        pred = self.pipe.predict(X)
        return pd.DataFrame({"prediction": pred, "lower": pred - self.threshold, "upper": pred + self.threshold},
                            index=X.index)

    def predict_set(self, X: pd.DataFrame) -> pd.DataFrame:
        """Las clases plausibles de cada fila. Un conjunto grande significa que el modelo duda."""
        self._need()
        if self.task != "classification":
            raise ValueError("Sets are for classification; use predict_interval for regression.")
        proba = self.pipe.predict_proba(X)
        classes = np.asarray(self.pipe.classes_)
        sets = [list(classes[row >= 1 - self.threshold]) or [classes[row.argmax()]] for row in proba]
        return pd.DataFrame({"prediction": classes[proba.argmax(axis=1)], "set": [", ".join(map(str, s)) for s in sets],
                             "size": [len(s) for s in sets]}, index=X.index)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Cobertura real sobre datos nuevos: debería rondar `coverage`."""
        self._need()
        if self.task == "regression":
            out = self.predict_interval(X)
            inside = (np.asarray(y, float) >= out["lower"]) & (np.asarray(y, float) <= out["upper"])
            return {"coverage": float(inside.mean()), "target": self.coverage, "width": float(2 * self.threshold)}
        out = self.predict_set(X)
        inside = [str(t) in {s.strip() for s in text.split(",")} for t, text in zip(y, out["set"])]
        return {"coverage": float(np.mean(inside)), "target": self.coverage, "average_set_size": float(out["size"].mean()),
                "single_answer": float((out["size"] == 1).mean())}


def interval_figure(intervals: pd.DataFrame, actual=None, limit: int = 80) -> go.Figure:
    part = intervals.head(limit).reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=part.index, y=part["upper"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=part.index, y=part["lower"], mode="lines", line=dict(width=0), fill="tonexty",
                             name="Range", opacity=0.3))
    fig.add_trace(go.Scatter(x=part.index, y=part["prediction"], mode="lines+markers", name="Prediction"))
    if actual is not None:
        fig.add_trace(go.Scatter(x=part.index, y=np.asarray(actual)[:limit], mode="markers", name="Actual"))
    fig.update_layout(title="Predictions with their range", xaxis_title="Row")
    return fig
