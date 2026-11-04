import numpy as np
import pandas as pd
import pytest
from scipy.stats import loguniform, randint, uniform
from typer.testing import CliRunner

from kikolens.analyze import bayesopt as bo
from kikolens.analyze import modellab as lab
from kikolens.cli.main import app


def test_space_decodes_every_kind_inside_its_bounds():
    space = bo.Space({"a": loguniform(1e-3, 1e3), "b": uniform(0.5, 0.4), "c": randint(2, 10), "d": [None, 6, (3, 4)]})
    for unit in np.random.default_rng(0).random((200, 4)):
        p = space.decode(unit)
        assert 1e-3 <= p["a"] <= 1e3 and 0.5 <= p["b"] <= 0.9 and 2 <= p["c"] <= 9 and isinstance(p["c"], int) and p["d"] in (None, 6, (3, 4))
    assert space.decode([0, 0, 0, 0]) == {"a": pytest.approx(1e-3), "b": 0.5, "c": 2, "d": None}
    assert space.decode([1, 1, 1, 1])["c"] == 9 and space.decode([1, 1, 1, 1])["d"] == (3, 4)
    with pytest.raises(ValueError, match="empty"):
        bo.Space({})
    from scipy.stats import norm
    with pytest.raises(ValueError, match="unsupported"):
        bo.Space({"x": norm()})


def test_optimizer_beats_random_search_on_a_smooth_function():
    def objective(p):                         # máximo 0 en (a=1, b=0.7); log-escala en a
        return -(np.log10(p["a"]) ** 2) - 4 * (p["b"] - 0.7) ** 2
    space = bo.Space({"a": loguniform(1e-3, 1e3), "b": uniform(0, 1)})
    bayes = [bo.optimize(objective, space, n_iter=24, seed=s)["best_score"] for s in range(5)]
    rng = np.random.default_rng(0)
    random = [max(objective(space.decode(u)) for u in rng.random((24, 2))) for _ in range(5)]
    assert np.mean(bayes) > np.mean(random) and np.mean(bayes) > -0.05


def test_history_phases_and_failed_evaluations():
    calls = {"n": 0}

    def flaky(p):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return -(p["x"] - 0.3) ** 2
    res = bo.optimize(flaky, bo.Space({"x": uniform(0, 1)}), n_iter=12)
    h = res["history"]
    assert len(h) == 12 and h["phase"].iloc[0] == "explore" and (h["phase"] == "exploit").sum() == 12 - max(3, 12 // 4)
    assert np.isnan(h["score"].iloc[1]) and res["best_score"] == h["score"].max() and abs(res["best_params"]["x"] - 0.3) < 0.2
    assert (np.diff(h["best so far"]) >= 0).all() and len(res["figure"].data) == 3
    with pytest.raises(ValueError, match="3 attempts"):
        bo.optimize(flaky, bo.Space({"x": uniform(0, 1)}), n_iter=2)
    with pytest.raises(ValueError, match="No combination"):
        bo.optimize(lambda p: 1 / 0, bo.Space({"x": uniform(0, 1)}), n_iter=5)


def regression_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    df["y"] = np.sin(3 * df["a"]) + df["b"] ** 2 + rng.normal(0, 0.1, n)
    return df


def test_lab_tunes_with_bayes_and_keeps_the_history():
    res = lab.run_model_lab(regression_df(), "y", models=["Ridge", "Random Forest"], folds=3, tune=True, n_iter=8, tuner="bayes")
    t = res["tuning"]
    assert t["method"] == "bayes" and len(t["history"]) == 8 and t["tuned_cv"] >= t["default_cv"] - 0.1
    X, y, _ = lab.prepare_xy(regression_df(), "y")
    pipe, params, score, base = lab.tune_model(X, y, "regression", "Random Forest", n_iter=6, folds=3, method="bayes")
    assert score >= base - 0.2 and set(params) >= {"max_depth"} and len(lab.tune_model.last_history) == 6
    with pytest.raises(ValueError, match="method"):
        lab.tune_model(X, y, "regression", "Ridge", method="grid")


def test_lab_cli_bayes_flag(tmp_path):
    csv = tmp_path / "r.csv"
    regression_df(200).to_csv(csv, index=False)
    out = CliRunner().invoke(app, ["lab", str(csv), "y", "--models", "Ridge,Random Forest", "--folds", "3", "--bayes"])
    assert out.exit_code == 0, out.output
