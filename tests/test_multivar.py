import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import multivar as mv
from kikolens.cli.main import app


def world(n=400, seed=0):
    rng = np.random.default_rng(seed)
    price = 10 + np.cumsum(rng.normal(0, 0.3, n))                      # el precio evoluciona solo
    promo = (rng.random(n) < 0.2).astype(float)
    sales = np.zeros(n)
    for t in range(2, n):
        sales[t] = 5 + 0.5 * sales[t - 1] - 3 * (price[t - 2] - 10) + 8 * promo[t - 1] + rng.normal(0, 1)
    noise = rng.normal(size=n)
    return pd.DataFrame({"date": pd.date_range("2023-01-01", periods=n), "sales": sales, "price": price, "promo": promo, "noise": noise})


def test_granger_finds_the_driver_and_not_the_reverse_or_noise():
    df = world()
    fwd = mv.granger(df, "date", "price", "sales", max_lag=4)
    assert fwd["significant"] and fwd["best_lag"] >= 2 and "predictive, not proof" in fwd["summary"]
    # sin relación real, con alfa 5 % puede salir alguna falsa alarma suelta: se mira en varios mundos
    noise = [mv.granger(world(seed=s), "date", "noise", "sales", max_lag=4)["significant"] for s in range(1, 7)]
    reverse = [mv.granger(world(seed=s), "date", "sales", "price", max_lag=4)["significant"] for s in range(1, 7)]
    assert sum(noise) <= 1 and sum(reverse) <= 1
    assert (fwd["table"]["adjusted p"] >= fwd["table"]["p-value"]).all()


def test_granger_validation():
    df = world()
    with pytest.raises(ValueError, match="different"):
        mv.granger(df, "date", "sales", "sales")
    with pytest.raises(ValueError, match="max_lag"):
        mv.granger(df, "date", "price", "sales", max_lag=0)
    with pytest.raises(ValueError, match="At least"):
        mv.granger(df.head(20), "date", "price", "sales")
    with pytest.raises(ValueError, match="numeric"):
        mv.granger(df.assign(price="x"), "date", "price", "sales")
    with pytest.raises(ValueError, match="not found"):
        mv.granger(df, "date", "nope", "sales")


def test_drivers_improve_the_forecast_when_they_matter_and_not_when_they_do_not():
    df = world()
    good = mv.forecast_with_drivers(df, "date", "sales", ["price", "promo"], horizon=10)
    assert good["drivers_help"] and good["mae_with_drivers"] < good["mae_alone"] and len(good["forecast"]) == 10
    assert (good["forecast"]["lower"] < good["forecast"]["forecast"]).all() and (good["forecast"]["forecast"] < good["forecast"]["upper"]).all()
    useless = mv.forecast_with_drivers(df, "date", "sales", ["noise"], horizon=5)
    assert not useless["drivers_help"] and "do not clearly improve" in useless["summary"]


def test_scenario_moves_the_forecast_in_the_expected_direction():
    df = world()
    base = mv.forecast_with_drivers(df, "date", "sales", ["price"], horizon=8)["forecast"]["forecast"]
    dear = mv.forecast_with_drivers(df, "date", "sales", ["price"], horizon=8, scenario={"price": 1.5})["forecast"]["forecast"]
    assert dear.mean() < base.mean() - 3                                # más precio, menos ventas
    assert base.index[0] == df["date"].max() + pd.Timedelta(days=1)


def test_forecast_validation():
    df = world()
    with pytest.raises(ValueError, match="at least one"):
        mv.forecast_with_drivers(df, "date", "sales", [])
    with pytest.raises(ValueError, match="own drivers"):
        mv.forecast_with_drivers(df, "date", "sales", ["sales"])
    with pytest.raises(ValueError, match="horizon"):
        mv.forecast_with_drivers(df, "date", "sales", ["price"], horizon=0)
    with pytest.raises(ValueError, match="not drivers"):
        mv.forecast_with_drivers(df, "date", "sales", ["price"], scenario={"promo": 2})


def test_cli(tmp_path):
    csv = tmp_path / "w.csv"
    world().to_csv(csv, index=False)
    out = CliRunner().invoke(app, ["granger", str(csv), "date", "price", "sales", "--max-lag", "3"])
    assert out.exit_code == 0 and "predict" in out.output
    out = CliRunner().invoke(app, ["forecast-drivers", str(csv), "date", "sales", "--drivers", "price,promo", "--horizon", "5", "--scenario", "price=1.2"])
    assert out.exit_code == 0 and "error" in out.output
    assert CliRunner().invoke(app, ["granger", str(csv), "date", "price", "price"]).exit_code == 1
    assert CliRunner().invoke(app, ["forecast-drivers", str(csv), "date", "sales", "--drivers", "price", "--scenario", "price=abc"]).exit_code == 1
