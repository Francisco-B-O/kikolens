import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import timeseries as ts


@pytest.fixture
def daily():
    idx = pd.date_range("2024-01-01", periods=210, freq="D")
    t = np.arange(len(idx))
    rng = np.random.default_rng(1)
    value = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, len(idx))
    return pd.DataFrame({"date": idx.astype(str), "sales": value, "label": "x"})


def test_detect_datetime_columns(daily):
    assert ts.detect_datetime_columns(daily) == ["date"]
    daily["date"] = pd.to_datetime(daily["date"])
    assert "date" in ts.detect_datetime_columns(daily)


@pytest.mark.parametrize("freq,expected", [("h", "h"), ("D", "D"), ("W", "W"), ("MS", "MS"), ("QS", "QS"), ("YS", "YS")])
def test_infer_frequency(freq, expected):
    assert ts.infer_frequency(pd.date_range("2000-01-01", periods=10, freq=freq)) == expected


def test_prepare_series_regularizes_gaps(daily):
    gapped = daily.drop(index=range(50, 60)).sample(frac=1, random_state=0)
    s = ts.prepare_series(gapped, "date", "sales")
    assert s.index.is_monotonic_increasing and len(s) == 210
    assert not s.isna().any()


def test_prepare_series_aggregates_to_coarser_freq(daily):
    s = ts.prepare_series(daily, "date", "sales", freq="W", agg="sum")
    assert len(s) < 40


def test_prepare_series_validation(daily):
    with pytest.raises(ValueError, match="numeric"):
        ts.prepare_series(daily, "date", "label")
    with pytest.raises(ValueError, match="not found"):
        ts.prepare_series(daily, "nope", "sales")
    with pytest.raises(ValueError, match="Aggregation"):
        ts.prepare_series(daily, "date", "sales", agg="median2")
    with pytest.raises(ValueError, match="8 time periods"):
        ts.prepare_series(daily.head(5), "date", "sales")


def test_period_and_decomposition(daily):
    s = ts.prepare_series(daily, "date", "sales")
    assert ts.guess_period(s) == 7
    parts = ts.decompose(s)
    assert parts["period"] == 7
    assert parts["seasonal_strength"] > 0.8
    assert abs(parts["seasonal"].iloc[:7].sum()) < 1e-6
    assert parts["trend"].dropna().iloc[-1] > parts["trend"].dropna().iloc[0]


def test_forecast_beats_naive_on_seasonal_trend(daily):
    s = ts.prepare_series(daily, "date", "sales")
    res = ts.forecast(s, horizon=14)
    f = res["forecast"]
    assert len(f) == 14
    assert f.index[0] == s.index[-1] + pd.Timedelta(days=1)
    assert (f["lower"] < f["forecast"]).all() and (f["forecast"] < f["upper"]).all()
    assert res["metrics"]["Beats naive"]
    assert res["metrics"]["MAE"] < 5


def test_forecast_without_seasonality():
    s = pd.Series(np.linspace(1, 50, 40), index=pd.date_range("2020-01-01", periods=40, freq="MS"))
    res = ts.forecast(s, horizon=3, period=None)
    assert res["forecast"]["forecast"].iloc[0] == pytest.approx(51, abs=2)


def test_forecast_rejects_bad_horizon(daily):
    with pytest.raises(ValueError):
        ts.forecast(ts.prepare_series(daily, "date", "sales"), horizon=0)


def test_trend_summary(daily):
    s = ts.prepare_series(daily, "date", "sales")
    summary = ts.trend_summary(s)
    assert summary["direction"] == "rising" and summary["change_pct"] > 0
    flat = pd.Series(np.random.default_rng(2).normal(0, 1, 50), index=pd.date_range("2024-01-01", periods=50))
    assert ts.trend_summary(flat)["direction"] == "flat"
