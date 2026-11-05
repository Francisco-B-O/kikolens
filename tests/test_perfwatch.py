import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import perfwatch as pw
from kikolens.analyze.modellab import run_model_lab
from kikolens.analyze.predict import build_bundle, save_bundle
from kikolens.cli.main import app


def world(n=600, seed=0, broken_from=None):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["y"] = 4 * df["x"] - 2 * df["z"] + rng.normal(0, 0.5, n)
    if broken_from is not None:                       # el mundo cambia: la relación con x se invierte
        late = df["date"] >= broken_from
        df.loc[late, "y"] = -4 * df.loc[late, "x"] - 2 * df.loc[late, "z"] + rng.normal(0, 0.5, late.sum())
    return df


@pytest.fixture(scope="module")
def bundle():
    train = world(seed=1)
    res = run_model_lab(train, "y", models=["Ridge"], folds=3, tune=False, drop=["date"])
    return build_bundle(res, train, name="w")


def test_stable_world_raises_no_alert(bundle):
    res = pw.performance_over_time(bundle, world(seed=2), date="date", period="Month")
    assert res["status"] == "ok" and not res["alerts"] and res["overall"]["R2"] > 0.9
    assert (res["table"]["enough rows"]).all() and "holding up" in res["summary"]


def test_a_changed_world_is_caught_with_the_period_it_started(bundle):
    res = pw.performance_over_time(bundle, world(seed=2, broken_from="2024-05-01"), date="date", period="Month")
    assert res["status"] == "alert"
    text = " ".join(res["alerts"])
    assert "2024-05" in text and "2024-02" not in text and res["overall"]["MAE"] > res["baseline"]


def test_without_a_date_the_rows_are_split_in_order(bundle):
    data = world(seed=2, broken_from="2024-07-01")
    res = pw.performance_over_time(bundle, data.drop(columns="date"), chunks=4)
    assert list(res["table"]["period"]) == ["part 1", "part 2", "part 3", "part 4"] and res["status"] in ("alert", "ok")
    assert res["table"]["rows"].sum() == len(data)


def test_classification_accuracy_drop(tmp_path):
    rng = np.random.default_rng(0)

    def make(n, flip_after=None):
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "x": rng.normal(size=n), "z": rng.normal(size=n)})
        df["k"] = np.where(df["x"] + df["z"] > 0, "up", "down")
        if flip_after:
            late = df["date"] >= flip_after
            df.loc[late, "k"] = np.where(df.loc[late, "x"] - df.loc[late, "z"] > 0, "up", "down")
        return df
    train = make(600)
    bundle = build_bundle(run_model_lab(train, "k", models=["Logistic Regression"], folds=3, tune=False, drop=["date"]), train)
    ok = pw.performance_over_time(bundle, make(600), date="date")
    bad = pw.performance_over_time(bundle, make(600, "2024-05-01"), date="date")
    assert ok["metric"] == "accuracy" and ok["status"] == "ok" and bad["status"] == "alert"


def test_validation_errors(bundle):
    data = world()
    with pytest.raises(ValueError, match="named 'y'"):
        pw.performance_over_time(bundle, data.drop(columns="y"))
    with pytest.raises(ValueError, match="Period"):
        pw.performance_over_time(bundle, data, date="date", period="Decade")
    with pytest.raises(ValueError, match="not found"):
        pw.performance_over_time(bundle, data, date="nope")
    with pytest.raises(ValueError, match="At least"):
        pw.performance_over_time(bundle, data.head(10))


def test_cli_performance(bundle, tmp_path):
    model = save_bundle(bundle, path=tmp_path / "m.klmodel")
    csv = tmp_path / "later.csv"
    world(seed=3, broken_from="2024-05-01").to_csv(csv, index=False)
    out = CliRunner().invoke(app, ["performance", str(csv), str(model), "--date", "date"])
    assert out.exit_code == 2 and "may need retraining" in out.output           # 2 = aviso, como en `drift`
    good = tmp_path / "good.csv"
    world(seed=4).to_csv(good, index=False)
    assert CliRunner().invoke(app, ["performance", str(good), str(model), "--date", "date"]).exit_code == 0
    assert CliRunner().invoke(app, ["performance", str(csv), str(model), "--period", "Decade", "--date", "date"]).exit_code == 1
