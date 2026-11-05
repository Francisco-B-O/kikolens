import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import forecasting as fc
from kikolens.cli.main import app

runner = CliRunner()


def make(n=240, seed=0, jump=0.0, jump_at=150, noise=3.0, freq="D"):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 100 + 0.2 * t + 15 * np.sin(2 * np.pi * t / 7) + rng.normal(0, noise, n)
    y[jump_at:] += jump
    return pd.Series(y, index=pd.date_range("2024-01-01", periods=n, freq=freq), name="sales")


def test_every_model_returns_the_requested_length():
    s = make()
    for name, model in fc.MODELS.items():
        out = model(s.to_numpy(), s.index, 10, 7)
        assert len(out) == 10 and np.isfinite(out).all(), name


def test_seasonal_models_beat_naive_on_a_seasonal_series():
    result = fc.forecast_models(make(), 14)
    board = result["leaderboard"].set_index("model")
    assert result["best"] != "Naive" and result["beats_naive"]
    assert board.loc["Naive", "MAE"] > 3 * board.loc[result["best"], "MAE"]
    assert board.loc["Seasonal naive", "MAE"] < board.loc["Naive", "MAE"]
    assert (board["status"] == "ok").all() and result["period"] == 7


def test_forecast_interval_covers_the_backtest_errors_and_is_ordered():
    result = fc.forecast_models(make(), 14, coverage=0.9)
    f = result["forecast"]
    assert (f["lower"] < f["forecast"]).all() and (f["forecast"] < f["upper"]).all()
    width = (f["upper"] - f["lower"]).to_numpy()
    assert (np.diff(width) >= -1e-9).all()                        # el intervalo no se estrecha con el horizonte
    # con datos nuevos del mismo proceso, ~90% de los reales cae dentro
    s = make(n=300, seed=1)
    train, future = s.iloc[:286], s.iloc[286:]
    res = fc.forecast_models(train, 14)
    inside = ((future.to_numpy() >= res["forecast"]["lower"]) & (future.to_numpy() <= res["forecast"]["upper"])).mean()
    assert inside >= 0.7


def test_forecast_dates_continue_the_series():
    s = make()
    f = fc.forecast_models(s, 5)["forecast"]
    assert f.index[0] == s.index[-1] + pd.Timedelta(days=1) and len(f) == 5


def test_holt_winters_handles_non_seasonal_and_short_series():
    s = make()
    out = fc.holt_winters(s.to_numpy(), s.index, 5, None)
    assert len(out) == 5 and abs(out[0] - s.iloc[-1]) < 40
    short = fc.holt_winters(s.to_numpy()[:10], s.index[:10], 3, 7)   # menos de dos ciclos: sin estacionalidad
    assert len(short) == 3


def test_events_improve_the_boosting_model():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=300, freq="D")
    events = idx[::30]
    y = 50 + rng.normal(0, 1, 300) + np.isin(idx, events) * 40
    s = pd.Series(y, index=idx)
    base = fc.backtest(s, 15, ["Lag boosting"], folds=2, events=events)["leaderboard"].iloc[0]["MAE"]
    none = fc.backtest(s, 15, ["Lag boosting"], folds=2)["leaderboard"].iloc[0]["MAE"]
    assert base < none * 0.8


def test_backtest_validation(monkeypatch):
    s = make()
    with pytest.raises(ValueError, match="Unknown model"):
        fc.backtest(s, 7, ["Magic"])
    with pytest.raises(ValueError, match="At least"):
        fc.backtest(s.head(10), 3)
    with pytest.raises(ValueError, match="horizon"):
        fc.backtest(s, 0)
    with pytest.raises(ValueError, match="coverage"):
        fc.forecast_models(s, 7, coverage=0.2)
    monkeypatch.setitem(fc.MODELS, "Broken", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    board = fc.backtest(s, 7, ["Naive", "Broken"])["leaderboard"].set_index("model")
    assert board.loc["Broken", "status"] == "failed" and board.loc["Naive", "status"] == "ok"


def test_changepoints_find_a_level_shift_and_ignore_noise():
    found = fc.detect_changepoints(make(jump=60, jump_at=150, noise=2))
    assert len(found) == 1 and found[0]["kind"] == "level shift"
    assert abs((found[0]["date"] - pd.Timestamp("2024-01-01")).days - 150) <= 6 and found[0]["level change"] > 40
    assert fc.detect_changepoints(pd.Series(100 + np.random.default_rng(2).normal(0, 3, 240),
                                            index=pd.date_range("2024-01-01", periods=240))) == []
    with pytest.raises(ValueError, match="At least 20"):
        fc.detect_changepoints(make().head(10))
    with pytest.raises(ValueError, match="max_points"):
        fc.detect_changepoints(make(), max_points=0)


def test_changepoints_find_a_trend_change():
    n = 240
    t = np.arange(n)
    y = np.where(t < 120, 0.1 * t, 12 + 1.5 * (t - 120)) + np.random.default_rng(4).normal(0, 1, n)
    found = fc.detect_changepoints(pd.Series(y, index=pd.date_range("2024-01-01", periods=n)))
    assert found and found[0]["slope after"] > 10 * found[0]["slope before"] and found[0]["kind"] == "trend change"


def test_forecast_by_group():
    frames = []
    for g, level in (("A", 100), ("B", 300), ("C", 50)):
        s = make(seed=ord(g)).rename("v") + level
        frames.append(pd.DataFrame({"date": s.index, "v": s.to_numpy(), "shop": g}))
    short = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5), "v": range(5), "shop": "tiny"})
    out = fc.forecast_by_group(pd.concat(frames + [short]), "date", "v", "shop", horizon=7, agg="sum")
    assert set(out["summary"]["group"]) == {"A", "B", "C"} and list(out["skipped"]["group"]) == ["tiny"]
    f = out["forecast"]
    assert len(f) == 21 and f[f["group"] == "B"]["forecast"].mean() > f[f["group"] == "C"]["forecast"].mean()
    with pytest.raises(ValueError, match="not found"):
        fc.forecast_by_group(pd.concat(frames), "date", "v", "ghost")
    with pytest.raises(ValueError, match="at least two"):
        fc.forecast_by_group(frames[0], "date", "v", "shop")


def test_figures_render():
    s = make()
    res = fc.forecast_models(s, 7)
    assert len(fc.forecast_figure(s, res).data) >= 4
    fig = fc.changepoint_figure(s, fc.detect_changepoints(make(jump=60)))
    assert len(fig.layout.shapes) == 1 and len(fig.layout.annotations) == 1


def test_cli_forecast_compare_and_changes(tmp_path):
    s = make(jump=60)
    path = tmp_path / "s.csv"
    pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "sales": s.to_numpy()}).to_csv(path, index=False)
    out = runner.invoke(app, ["forecast", str(path), "date", "sales", "--horizon", "7", "--compare", "--changes"])
    assert out.exit_code == 0, out.output
    assert "Model comparison" in out.output and "level shift" in out.output
    grouped = tmp_path / "g.csv"
    both = pd.concat([pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "sales": s.to_numpy(), "shop": g}) for g in "AB"])
    both.to_csv(grouped, index=False)
    assert runner.invoke(app, ["forecast", str(grouped), "date", "sales", "--horizon", "5", "--by", "shop"]).exit_code == 0
    assert runner.invoke(app, ["forecast", str(grouped), "date", "sales", "--by", "ghost"]).exit_code == 1
