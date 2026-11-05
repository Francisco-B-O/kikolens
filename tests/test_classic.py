import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import classic as cl


def ar1(n=300, phi=0.7, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + 50)
    y = np.zeros(n + 50)
    for t in range(1, n + 50):
        y[t] = phi * y[t - 1] + e[t]
    return y[50:]


def monthly(n=144, seed=3, outlier=False):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = 10 * np.sin(2 * np.pi * t / 12) + 4 * np.cos(4 * np.pi * t / 12)
    y = pd.Series(100 + 0.5 * t + seasonal + rng.normal(0, 1.5, n), index=pd.date_range("2012-01-01", periods=n, freq="MS"))
    if outlier:
        y.iloc[50] += 60
    return y, seasonal


# ── Estacionariedad ──────────────────────────────────────────────────────────
def test_the_adf_p_value_reproduces_the_published_critical_values():
    for regression, table in (("c", {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}), ("ct", {0.01: -3.96, 0.05: -3.41, 0.10: -3.13})):
        for level, value in table.items():
            assert cl._adf_p(value, regression) == pytest.approx(level, abs=0.003)
            assert cl._critical(level, regression) == pytest.approx(value, abs=0.02)
    assert cl._adf_p(5.0, "c") == 1.0 and cl._adf_p(-30.0, "c") == 0.0


def test_the_two_tests_are_calibrated_by_simulation():
    rng = np.random.default_rng(1)
    adf_size = kpss_power = kpss_size = adf_power = 0
    runs = 120
    for _ in range(runs):
        walk = np.cumsum(rng.normal(size=250))
        adf_size += cl.adf(walk)["p_value"] < 0.05
        kpss_power += cl.kpss(walk)["p_value"] < 0.05
        stationary = ar1(250, 0.5, seed=int(rng.integers(1_000_000)))
        kpss_size += cl.kpss(stationary)["p_value"] < 0.05
        adf_power += cl.adf(stationary)["p_value"] < 0.05
    assert adf_size / runs < 0.12 and kpss_size / runs < 0.14        # no rechazan de más cuando no deben
    assert kpss_power / runs > 0.6 and adf_power / runs > 0.95       # y sí cuando deben


def test_stationarity_combines_both_tests_and_suggests_the_differences():
    stationary = cl.stationarity(ar1())
    assert stationary["differences"] == 0 and stationary["verdict"].startswith("stationary")
    rng = np.random.default_rng(5)
    trending = np.cumsum(rng.normal(0.3, 1, 400))
    result = cl.stationarity(trending)
    assert result["differences"] == 1 and result["verdict"].startswith("not stationary") and result["table"].shape[0] == 3
    assert "Suggested differences: 1" in result["summary"]
    assert any("50" in n for n in cl.stationarity(ar1(40))["notes"])


def test_stationarity_inputs_are_checked():
    for bad in ([1.0] * 30, list(range(5)), [1.0, np.inf] * 20):
        with pytest.raises(ValueError):
            cl.adf(bad)
    with pytest.raises(ValueError, match="regression"):
        cl.adf(ar1(), "x")
    with pytest.raises(ValueError, match="regression"):
        cl.kpss(ar1(), "x")
    with pytest.raises(ValueError, match="At most"):
        cl.adf(np.random.default_rng(0).normal(size=30_000))


# ── Autocorrelación ──────────────────────────────────────────────────────────
def test_acf_pacf_and_ljung_box_identify_an_autoregressive_series():
    y = ar1(2000, 0.7)
    a, p = cl.acf(y, 5), cl.pacf(y, 5)
    assert a[1] == pytest.approx(0.7, abs=0.06) and a[2] == pytest.approx(0.49, abs=0.08)
    assert p[1] == pytest.approx(0.7, abs=0.06) and abs(p[2]) < 0.08 and abs(p[3]) < 0.08           # la parcial se corta en el retardo 1
    res = cl.acf_pacf(y)
    assert res["ljung_box"]["p_value"] < 0.001 and res["notes"] and len(res["figure"].data) == 2 and "Largest autocorrelation" in res["summary"]
    noise = np.random.default_rng(2).normal(size=500)
    assert cl.ljung_box(noise, 10)["p_value"] > 0.01


# ── STL ──────────────────────────────────────────────────────────────────────
def test_stl_recovers_the_season_and_the_trend_and_the_parts_add_up():
    y, seasonal = monthly()
    res = cl.stl(y, 12)
    parts = res["components"]
    assert (parts["observed"] - parts["trend"] - parts["seasonal"] - parts["remainder"]).abs().max() < 1e-9
    assert np.corrcoef(parts["seasonal"], seasonal)[0, 1] > 0.97
    assert np.polyfit(np.arange(len(y)), parts["trend"], 1)[0] == pytest.approx(0.5, abs=0.03)
    assert res["seasonal_strength"] > 0.9 and res["trend_strength"] > 0.95 and res["period"] == 12 and len(res["figure"].data) == 4


def test_stl_is_robust_to_an_outlier_and_flags_it():
    y, _ = monthly(outlier=True)
    res = cl.stl(y, 12)
    assert res["components"]["remainder"].iloc[50] > 50 and y.index[50] in res["unusual"].index
    assert res["seasonal_strength"] > 0.9                                             # el valor extremo no hunde la fuerza estacional
    clean, _ = monthly()
    assert abs(res["components"]["trend"].iloc[60] - cl.stl(clean, 12)["components"]["trend"].iloc[60]) < 2.0


def test_stl_detects_the_period_and_refuses_what_it_cannot_do():
    y, _ = monthly()
    assert cl.stl(y)["period"] == 12
    with pytest.raises(ValueError, match="two full periods"):
        cl.stl(y.iloc[:20], 12)
    noise = pd.Series(np.random.default_rng(0).normal(size=60), index=pd.date_range("2020-01-01", periods=60, freq="D"))
    weak = cl.stl(noise, 7)
    assert weak["seasonal_strength"] < 0.5 and any("weak" in n for n in weak["notes"])


# ── ARIMA ────────────────────────────────────────────────────────────────────
def test_arima_recovers_known_coefficients():
    res = cl.arima(pd.Series(ar1(1500, 0.7)), (1, 0, 0), horizon=5, validate=False)
    assert res["coefficients"]["ar"][0] == pytest.approx(0.7, abs=0.05) and res["sigma"] == pytest.approx(1.0, abs=0.06)
    rng = np.random.default_rng(4)
    e = rng.normal(size=1501)
    ma = pd.Series(e[1:] + 0.6 * e[:-1] + 5)
    fit = cl.arima(ma, (0, 0, 1), horizon=3, validate=False)
    assert fit["coefficients"]["ma"][0] == pytest.approx(0.6, abs=0.06) and fit["coefficients"]["mu"] == pytest.approx(5, abs=0.15)
    mixed = np.zeros(2050)
    e = rng.normal(size=2050)
    for t in range(1, 2050):
        mixed[t] = 0.6 * mixed[t - 1] + e[t] + 0.3 * e[t - 1]
    arma = cl.arima(pd.Series(mixed[50:]), (1, 0, 1), horizon=3, validate=False)
    assert arma["coefficients"]["ar"][0] == pytest.approx(0.6, abs=0.08) and arma["coefficients"]["ma"][0] == pytest.approx(0.3, abs=0.09)


def test_forecast_intervals_are_calibrated_and_widen_with_the_horizon():
    covered = total = 0
    widths = []
    for seed in range(40):
        y = ar1(260, 0.6, seed=seed)
        res = cl.arima(pd.Series(y[:240]), (1, 0, 0), horizon=10, level=0.9, validate=False)
        f = res["forecast"]
        covered += int(((y[240:250] >= f["low"].to_numpy()) & (y[240:250] <= f["high"].to_numpy())).sum())
        total += 10
        widths.append((f["high"] - f["low"]).to_numpy())
    assert 0.82 < covered / total < 0.97
    assert (np.diff(np.mean(widths, axis=0)) >= -1e-9).all()


def test_integrated_models_follow_a_random_walk_with_flat_forecasts():
    walk = pd.Series(np.cumsum(np.random.default_rng(6).normal(size=300)) + 50)
    res = cl.arima(walk, (0, 1, 0), horizon=8, validate=False)
    assert np.allclose(res["forecast"]["forecast"], walk.iloc[-1], atol=1e-8)                    # el mejor pronóstico de un paseo es el último valor
    assert res["forecast"]["high"].iloc[-1] - res["forecast"]["low"].iloc[-1] > res["forecast"]["high"].iloc[0] - res["forecast"]["low"].iloc[0]
    assert res["label"] == "ARIMA(0,1,0)"


def test_auto_arima_finds_the_season_and_beats_repeating_the_last_one():
    y, _ = monthly()
    res = cl.auto_arima(y, 12, horizon=12)
    assert res["seasonal"][1] == 1 and res["period"] == 12 and len(res["forecast"]) == 12
    assert res["forecast"].index[0] == pd.Timestamp("2024-01-01") and res["candidates"]["AICc"].is_monotonic_increasing and len(res["candidates"]) > 10
    assert res["backtest"]["model_mae"] < res["backtest"]["naive_mae"] and res["ljung_box"]["p_value"] > 0.01
    t = np.arange(144, 156)
    expected = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + 4 * np.cos(4 * np.pi * t / 12)
    assert np.abs(res["forecast"]["forecast"].to_numpy() - expected).mean() < 4
    assert "Chosen from" in res["summary"]


def test_auto_arima_differences_a_trending_series_and_keeps_noise_simple():
    walk = pd.Series(np.cumsum(np.random.default_rng(8).normal(0.2, 1, 300)))
    assert cl.auto_arima(walk, horizon=5)["order"][1] >= 1
    noise = pd.Series(np.random.default_rng(9).normal(size=200))
    res = cl.auto_arima(noise, horizon=5)
    table = res["candidates"].set_index("model")
    assert res["order"][1] == 0 and table.loc["ARIMA(0,0,0)", "difference to best"] < 4          # no hay casi nada que modelar


def test_a_model_that_misses_structure_is_called_out():
    y, _ = monthly()
    res = cl.arima(y, (0, 0, 0), horizon=6)
    assert any("autocorrelated" in n for n in res["notes"]) and res["ljung_box"]["p_value"] < 0.05


def test_arima_rejects_impossible_requests():
    y = pd.Series(ar1())
    for kwargs, fragment in (({"order": (9, 0, 0)}, "Orders"), ({"order": (1, 3, 0)}, "Orders"), ({"seasonal": (1, 0, 0)}, "period"), ({"horizon": 0}, "horizon"),
                             ({"level": 1.2}, "level")):
        with pytest.raises(ValueError, match=fragment):
            cl.arima(y, **kwargs)
    with pytest.raises(ValueError, match="too short"):
        cl.arima(y.iloc[:40], (5, 2, 5))


def test_python_api_and_terminal(tmp_path):
    import kikolens as kl
    from typer.testing import CliRunner

    from kikolens.cli.main import app
    y, _ = monthly()
    df = pd.DataFrame({"month": y.index, "sales": y.values})
    assert kl.stationarity(df, "month", "sales")["kind"] == "stationarity"
    assert kl.decompose_stl(df, "month", "sales", period=12)["period"] == 12
    assert kl.arima(df, "month", "sales", period=12, horizon=6)["kind"] == "arima" and len(kl.arima(df, "month", "sales", order=(1, 1, 1), horizon=4)["forecast"]) == 4
    assert kl.autocorrelation(df, "month", "sales")["kind"] == "autocorrelation"
    csv = tmp_path / "s.csv"
    df.to_csv(csv, index=False)
    runner = CliRunner()
    for args, text in ((["stationarity", str(csv), "month", "sales"], "Suggested differences"), (["stl", str(csv), "month", "sales", "--period", "12"], "seasonal strength"),
                       (["arima", str(csv), "month", "sales", "--period", "12", "--horizon", "6"], "Chosen from"),
                       (["arima", str(csv), "month", "sales", "--order", "1,1,1", "--horizon", "3"], "ARIMA(1,1,1)"), (["acf", str(csv), "month", "sales"], "Ljung-Box")):
        out = runner.invoke(app, args)
        assert out.exit_code == 0 and text in out.output, (args, out.output)
    assert runner.invoke(app, ["arima", str(csv), "month", "sales", "--order", "x"]).exit_code == 1
