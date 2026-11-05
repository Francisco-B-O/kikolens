"""Series temporales: detección de fechas, remuestreo, descomposición y previsión."""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from kikolens.analyze.quality import looks_datetime_text

_PERIOD_BY_FREQ = {"h": 24, "D": 7, "W": 52, "MS": 12, "QS": 4, "YS": None}
FREQUENCIES = {"Hourly": "h", "Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}
AGGREGATIONS = ("mean", "sum", "median", "min", "max", "count")


def detect_datetime_columns(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cols += [c for c in df.columns if c not in cols and looks_datetime_text(df[c])]
    return cols


def infer_frequency(index: pd.DatetimeIndex) -> str:
    """Frecuencia dominante según la mediana del salto entre fechas."""
    if len(index) < 2:
        return "D"
    step = pd.Series(index).sort_values().diff().dropna().median()
    if step <= pd.Timedelta(hours=1):
        return "h"
    if step <= pd.Timedelta(days=1):
        return "D"
    if step <= pd.Timedelta(days=7):
        return "W"
    if step <= pd.Timedelta(days=31):
        return "MS"
    if step <= pd.Timedelta(days=92):
        return "QS"
    return "YS"


def prepare_series(df, date_col, value_col, freq=None, agg="mean") -> pd.Series:
    """Serie regular y ordenada: agrega por periodo e interpola huecos."""
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError("Date or value column not found.")
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"Value column '{value_col}' must be numeric.")
    if agg not in AGGREGATIONS:
        raise ValueError(f"Aggregation must be one of {AGGREGATIONS}.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dates = pd.to_datetime(df[date_col], errors="coerce")
    frame = pd.DataFrame({"d": dates, "v": df[value_col]}).dropna(subset=["d"])
    if frame.empty:
        raise ValueError(f"No valid dates found in '{date_col}'.")

    frame = frame.sort_values("d").set_index("d")
    freq = freq or infer_frequency(frame.index)
    series = frame["v"].resample(freq).agg(agg)
    if agg in ("sum", "count"):
        series = series.fillna(0)
    else:
        series = series.interpolate(limit_direction="both")
    series = series.dropna()
    if len(series) < 8:
        raise ValueError("At least 8 time periods are required; try a finer frequency.")
    series.name = value_col
    return series


def guess_period(series: pd.Series):
    freq = series.index.freq.freqstr if series.index.freq else infer_frequency(series.index)
    key = {"H": "h"}.get(freq, freq)
    period = _PERIOD_BY_FREQ.get(key)
    if period and len(series) >= 2 * period:
        return period
    return _autocorrelation_period(series)


def _autocorrelation_period(series: pd.Series, max_lag: int = 60):
    values = series.to_numpy(dtype=float)
    n = len(values)
    values = values - values.mean()
    denom = float(np.dot(values, values))
    if denom == 0 or n < 12:
        return None
    best_lag, best = None, 0.3  # umbral mínimo de autocorrelación
    for lag in range(2, min(max_lag, n // 2)):
        ac = float(np.dot(values[:-lag], values[lag:]) / denom)
        if ac > best:
            best_lag, best = lag, ac
    return best_lag


def decompose(series: pd.Series, period=None) -> dict:
    """Descomposición aditiva: tendencia (media móvil centrada), estacionalidad y residuo."""
    period = period or guess_period(series)
    window = period if period and period >= 2 else max(3, len(series) // 10)
    trend = series.rolling(window, center=True, min_periods=max(1, window // 2)).mean()
    detrended = series - trend
    if period and period >= 2:
        positions = np.arange(len(series)) % period
        pattern = detrended.groupby(positions).mean()
        pattern = pattern - pattern.mean()
        seasonal = pd.Series(pattern.reindex(positions).to_numpy(), index=series.index)
    else:
        seasonal = pd.Series(0.0, index=series.index)
    resid = series - trend - seasonal
    strength = None
    if period and period >= 2 and float(np.var(detrended.dropna() - 0)) > 0:
        strength = float(max(0.0, 1 - np.var(resid.dropna()) / np.var((seasonal + resid).dropna())))
    return {"trend": trend, "seasonal": seasonal, "resid": resid, "period": period,
            "seasonal_strength": strength}


def _design(n_start, n_end, period, scale):
    t = np.arange(n_start, n_end, dtype=float)
    cols = [t / max(scale, 1)]
    if period and period >= 2:
        pos = (np.arange(n_start, n_end) % period)
        for k in range(1, min(period // 2, 6) + 1):  # armónicos de Fourier
            cols += [np.sin(2 * np.pi * k * pos / period), np.cos(2 * np.pi * k * pos / period)]
    return np.column_stack(cols)


def _fit_predict(values, horizon, period):
    n = len(values)
    model = Ridge(alpha=1e-3).fit(_design(0, n, period, n), values)
    fitted = model.predict(_design(0, n, period, n))
    return model.predict(_design(n, n + horizon, period, n)), fitted


def forecast(series: pd.Series, horizon: int = 12, period=None) -> dict:
    """Previsión por tendencia + estacionalidad de Fourier, validada con un backtest del final de la serie."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    period = period if period is not None else guess_period(series)
    values = series.to_numpy(dtype=float)
    n = len(values)

    holdout = min(horizon, max(2, n // 5))
    train, test = values[:-holdout], values[-holdout:]
    pred_test, _ = _fit_predict(train, holdout, period)
    naive = np.repeat(train[-1], holdout)
    mae = float(np.mean(np.abs(test - pred_test)))
    naive_mae = float(np.mean(np.abs(test - naive)))
    denom = np.abs(test)
    mape = float(np.mean(np.abs(test - pred_test)[denom > 0] / denom[denom > 0]) * 100) if (denom > 0).any() else None

    future, fitted = _fit_predict(values, horizon, period)
    sigma = float(np.std(values - fitted))
    steps = np.arange(1, horizon + 1)
    width = 1.96 * sigma * np.sqrt(1 + steps / max(n, 1))
    freq = series.index.freq or pd.tseries.frequencies.to_offset(infer_frequency(series.index))
    future_index = pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]
    result = pd.DataFrame({"forecast": future, "lower": future - width, "upper": future + width}, index=future_index)

    return {
        "forecast": result, "period": period,
        "metrics": {"MAE": mae, "MAPE %": mape, "Naive MAE": naive_mae,
                    "Beats naive": bool(mae < naive_mae)},
        "backtest": pd.DataFrame({"actual": test, "predicted": pred_test}, index=series.index[-holdout:]),
    }


def trend_summary(series: pd.Series) -> dict:
    """Pendiente por periodo, variación total y significado en lenguaje llano."""
    x = np.arange(len(series), dtype=float)
    slope, _ = np.polyfit(x, series.to_numpy(dtype=float), 1)
    first, last = float(series.iloc[: max(1, len(series) // 10)].mean()), float(series.iloc[-max(1, len(series) // 10):].mean())
    change_pct = (last - first) / abs(first) * 100 if first else None
    r = float(np.corrcoef(x, series.to_numpy(dtype=float))[0, 1]) if series.std() > 0 else 0.0
    significant = abs(r) > 2 / np.sqrt(len(series))  # ~95%: la correlación con el tiempo supera al ruido
    direction = ("rising" if slope > 0 else "falling") if significant else "flat"
    return {"slope": float(slope), "direction": direction, "change_pct": change_pct,
            "start": first, "end": last}
