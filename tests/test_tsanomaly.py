import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import tsanomaly as ta


def weekly(n=300, seed=0, noise=2.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 100 + 0.1 * t + 15 * np.sin(2 * np.pi * t / 7) + rng.normal(0, noise, n)
    return pd.Series(y, index=pd.date_range("2024-01-01", periods=n), name="sales")


def test_planted_spikes_are_found_and_normal_weeks_are_not():
    s = weekly()
    s.iloc[100] += 60
    s.iloc[200] -= 55
    res = ta.detect(s)
    dates = set(res["anomalies"]["date"])
    assert {s.index[100], s.index[200]} <= dates and len(dates) <= 6
    row = res["anomalies"].set_index("date")
    assert row.loc[s.index[100], "direction"] == "above" and row.loc[s.index[200], "direction"] == "below"
    assert "largest" in res["summary"]


def test_seasonal_peaks_are_not_flagged_as_anomalies():
    res = ta.detect(weekly(noise=1.0))
    assert res["share"] < 0.03                      # el ciclo semanal (±15) es normal, no anómalo


def test_threshold_controls_sensitivity_and_min_gap_merges_neighbours():
    s = weekly()
    s.iloc[150:153] += 70
    loose, strict = ta.detect(s, threshold=2.5), ta.detect(s, threshold=6)
    assert len(loose["anomalies"]) >= len(strict["anomalies"]) >= 1
    merged = ta.detect(s, min_gap=5)["anomalies"]
    assert len(merged) < len(ta.detect(s)["anomalies"])


def test_expected_series_adds_up_and_works_without_seasonality():
    s = weekly()
    frame = ta.expected_series(s, period=7)
    assert np.allclose(frame["expected"], frame["trend"] + frame["seasonal"]) and np.allclose(frame["value"] - frame["expected"], frame["residual"])
    flat = ta.expected_series(pd.Series(np.random.default_rng(0).normal(10, 1, 40), index=pd.date_range("2024-01-01", periods=40)), period=None)
    assert (flat["seasonal"] == 0).all() or flat["seasonal"].abs().max() < 5


def test_from_frame_and_validation():
    s = weekly()
    s.iloc[50] += 80
    df = pd.DataFrame({"day": s.index, "v": s.values})
    assert pd.Timestamp(s.index[50]) in set(ta.from_frame(df, "day", "v")["anomalies"]["date"])
    with pytest.raises(ValueError, match="At least"):
        ta.detect(s.head(10))
    with pytest.raises(ValueError, match="positive"):
        ta.detect(s, threshold=0)
