"""Anomalías en una serie temporal: puntos que se salen de lo esperado una vez descontadas la tendencia y la estacionalidad.
La tendencia es una mediana móvil, la estacionalidad la mediana de cada posición del ciclo, y el umbral usa la desviación
absoluta mediana (MAD), que no se deja arrastrar por los propios picos que busca."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from kikolens.analyze import timeseries as ts

MIN_POINTS = 20


def expected_series(series: pd.Series, period=None) -> pd.DataFrame:
    """Tendencia, estacionalidad, valor esperado y residuo de cada punto."""
    if len(series) < MIN_POINTS:
        raise ValueError(f"At least {MIN_POINTS} periods are needed.")
    y = series.astype(float)
    period = ts.guess_period(y) if period == "auto" or period is None else period
    seasonal_ok = bool(period and period >= 2 and len(y) >= 2 * period)
    window = int(max(5, (period or 0) * 2 + 1, len(y) // 20)) | 1       # impar
    trend = y.rolling(window, center=True, min_periods=max(3, window // 3)).median()
    trend = trend.interpolate(limit_direction="both")
    detrended = y - trend
    if seasonal_ok:
        phase = np.arange(len(y)) % int(period)
        cycle = detrended.groupby(phase).median()
        seasonal = pd.Series(cycle.reindex(phase).to_numpy(), index=y.index)
    else:
        seasonal = pd.Series(0.0, index=y.index)
    expected = trend + seasonal
    return pd.DataFrame({"value": y, "trend": trend, "seasonal": seasonal, "expected": expected, "residual": y - expected})


def detect(series: pd.Series, period="auto", threshold: float = 3.5, min_gap: int = 0) -> dict:
    """Puntos con una puntuación robusta (z de MAD) por encima de `threshold`. `min_gap` separa avisos consecutivos."""
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    frame = expected_series(series, period)
    resid = frame["residual"]
    mad = float(np.median(np.abs(resid - resid.median())))
    scale = 1.4826 * mad if mad > 0 else float(resid.std() or 1.0)
    frame["score"] = (resid - resid.median()) / scale
    flagged = frame[frame["score"].abs() >= threshold].copy()
    if min_gap and len(flagged) > 1:
        keep, last = [], None
        for pos in range(len(flagged)):
            location = frame.index.get_loc(flagged.index[pos])
            if last is None or location - last > min_gap:
                keep.append(pos)
                last = location
        flagged = flagged.iloc[keep]
    flagged["direction"] = np.where(flagged["residual"] > 0, "above", "below")
    table = flagged[["value", "expected", "residual", "score", "direction"]].rename_axis("date").reset_index()
    share = len(flagged) / len(frame)
    if len(table):
        worst = table.loc[table["score"].abs().idxmax()]
        detail = (f"; the largest is {pd.Timestamp(worst['date']):%Y-%m-%d}, {abs(worst['residual']):,.4g} "
                  f"{worst['direction']} what was expected.")
    else:
        detail = ". Nothing stands out."
    return {"frame": frame, "anomalies": table, "threshold": threshold, "share": share, "figure": figure(frame, flagged),
            "summary": f"{len(flagged)} of {len(frame)} periods ({share:.1%}) are unusual for their trend and season{detail}"}


def figure(frame: pd.DataFrame, flagged: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["value"], mode="lines", name="Observed"))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["expected"], mode="lines", name="Expected", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=flagged.index, y=flagged["value"], mode="markers", name="Unusual", marker=dict(size=10, symbol="x")))
    fig.update_layout(title="Observed against what trend and season predict")
    return fig


def from_frame(df: pd.DataFrame, date: str, value: str, agg: str = "mean", **kwargs) -> dict:
    """Atajo desde una tabla: prepara la serie regular y detecta."""
    return detect(ts.prepare_series(df, date, value, agg=agg), **kwargs)
