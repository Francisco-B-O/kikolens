import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import featureselect as fs
from kikolens.cli.main import app


def data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"signal1": rng.normal(size=n), "signal2": rng.normal(size=n), "cat": rng.choice(["a", "b", "c"], n)})
    for i in range(5):
        df[f"noise{i}"] = rng.normal(size=n)
    df["y"] = 3 * df["signal1"] + 2 * df["signal2"] + 2 * (df["cat"] == "a") + rng.normal(0, 0.3, n)
    return df


def test_consensus_ranks_the_real_drivers_first_and_recommends_a_small_set():
    res = fs.select_features(data(), "y")
    top = res["table"]["column"].head(3).tolist()
    assert set(top) == {"signal1", "signal2", "cat"}
    assert set(res["recommended"]) >= {"signal1", "signal2"} and len(res["recommended"]) <= 5
    assert set(res["removable"]) | set(res["recommended"]) == set(data().drop(columns="y").columns)
    assert res["score_recommended"] > 0.9 and res["metric"] == "R2" and "enough" in res["summary"]
    assert list(res["table"].columns[:5]) == ["column", *fs.METHODS] and (res["table"]["consensus"] == range(1, 9)).all()


def test_every_method_agrees_on_the_strongest_column_and_categories_are_summed():
    X, y = data().drop(columns="y"), data()["y"]
    scores = fs.score_columns(X, y, "regression")
    assert set(scores.columns) == set(fs.METHODS) and (scores >= 0).all().all()
    assert scores["permutation"].idxmax() == "signal1" and scores["random forest"].idxmax() == "signal1"
    assert scores.loc["cat", "L1 model"] > scores.loc["noise0", "L1 model"]


def test_classification_and_validation():
    df = data()
    df["label"] = np.where(df["signal1"] + df["signal2"] > 0, "yes", "no")
    res = fs.select_features(df.drop(columns="y"), "label")
    assert res["task"] == "classification" and res["metric"] == "F1" and {"signal1", "signal2"} <= set(res["recommended"])
    with pytest.raises(ValueError, match="At least two"):
        fs.select_features(df[["signal1", "y"]], "y")
    with pytest.raises(ValueError, match="at most"):
        fs.select_features(df, "y", max_columns=3)
    with pytest.raises(ValueError, match="not found"):
        fs.select_features(df, "nope")


def test_cli(tmp_path):
    csv = tmp_path / "d.csv"
    data(300).to_csv(csv, index=False)
    out = CliRunner().invoke(app, ["select-features", str(csv), "y", "--output", str(tmp_path / "slim.csv")])
    assert out.exit_code == 0 and "enough" in out.output
    slim = pd.read_csv(tmp_path / "slim.csv")
    assert "y" in slim.columns and slim.shape[1] < 9
    assert CliRunner().invoke(app, ["select-features", str(csv), "nope"]).exit_code == 1
