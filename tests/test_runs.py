import json
import os

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import runs as rn
from kikolens.analyze.modellab import run_model_lab
from kikolens.cli.main import app

runner = CliRunner()


def data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = 2 * df["a"] + rng.normal(0, 0.5, n)
    return df


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("KIKOLENS_HOME", str(tmp_path))
    return tmp_path


def test_tests_never_write_to_the_real_user_directory():
    assert "kikolens_tests_" in os.environ["KIKOLENS_HOME"] or "pytest" in os.environ["KIKOLENS_HOME"]


def test_log_list_get_and_delete(home):
    a = rn.log_run("lab", "y", {"R2": 0.8, "weird": float("nan"), "label": "x"}, {"folds": 5, "tuning": {"alpha": 1.0}}, name="Ridge", dataset="d.csv")
    b = rn.log_run("lab", "y", {"R2": 0.9, "RMSE": 3.0}, {"folds": 5}, name="Forest", dataset="d.csv")
    c = rn.log_run("lab", "z", {"R2": 0.1}, name="Tree")
    assert (home / "runs").exists() and len(list((home / "runs").glob("*.json"))) == 3
    frame = rn.list_runs()
    assert len(frame) == 3 and set(frame["model"]) == {"Ridge", "Forest", "Tree"}
    assert "weird" not in rn.get_run(a["id"])["metrics"] and "label" not in rn.get_run(a["id"])["metrics"]
    assert set(rn.list_runs("y")["id"]) == {a["id"], b["id"]} and rn.list_runs("nothing").empty
    rn.delete_run(c["id"])
    assert len(rn.list_runs()) == 2
    with pytest.raises(ValueError, match="No run"):
        rn.get_run("zzzzzzzz")
    with pytest.raises(ValueError, match="No run"):
        rn.delete_run("zzzzzzzz")
    with pytest.raises(ValueError, match="kind and a target"):
        rn.log_run("", "y", {})


def test_numpy_values_and_broken_files_are_handled(home):
    rec = rn.log_run("lab", "y", {"R2": np.float64(0.7), "n": np.int64(3)}, {"shape": (1, 2), "arr": np.float32(1.5), "fn": len})
    stored = json.loads(next((home / "runs").glob("*.json")).read_text())
    assert stored["metrics"] == {"R2": 0.7, "n": 3} and stored["params"]["shape"] == [1, 2] and isinstance(stored["params"]["fn"], str)
    (home / "runs" / "junk.json").write_text("not json")
    (home / "runs" / "other.json").write_text('{"no": "id"}')
    assert list(rn.list_runs()["id"]) == [rec["id"]]


def test_compare_marks_only_what_differs_and_best_picks_by_metric(home):
    a = rn.log_run("lab", "y", {"R2": 0.8, "RMSE": 4.0}, {"folds": 5, "tuner": "random"}, name="Ridge", dataset="d.csv")
    b = rn.log_run("lab", "y", {"R2": 0.9, "RMSE": 3.0}, {"folds": 5, "tuner": "bayes"}, name="Forest", dataset="d.csv")
    table = rn.compare([a["id"], b["id"]])
    differs = set(table.index[table["differs"]])
    assert {"metric: R2", "param: tuner", "model"} <= differs and "param: folds" not in differs and "dataset" not in differs
    assert rn.best_run("y")["id"] == b["id"] and rn.best_run("y", "RMSE")["id"] == b["id"]
    with pytest.raises(ValueError, match="at least two"):
        rn.compare([a["id"]])
    with pytest.raises(ValueError, match="No runs"):
        rn.best_run("missing")


def test_log_lab_records_a_real_lab_result(home):
    res = run_model_lab(data(), "y", models=["Ridge", "Random Forest"], folds=3, tune=True, n_iter=4)
    rec = rn.log_lab(res, "d.csv", {"folds": 3})
    full = rn.get_run(rec["id"])
    assert full["target"] == "y" and full["name"] == res["best_model"] and full["metrics"]["R2"] == pytest.approx(res["metrics"]["R2"])
    assert full["rows"] == 200 and full["params"]["folds"] == 3 and "CV R2" in full["metrics"] and full["dataset"] == "d.csv"


def test_cli_lab_logs_and_runs_command_reads(home, tmp_path):
    csv = tmp_path / "d.csv"
    data().to_csv(csv, index=False)
    assert runner.invoke(app, ["lab", str(csv), "y", "--models", "Ridge,Random Forest", "--folds", "3", "--no-tune"]).exit_code == 0
    out = runner.invoke(app, ["runs"])
    assert out.exit_code == 0 and "Runs, newest first" in out.output
    run_id = rn.list_runs()["id"].iloc[0]
    assert runner.invoke(app, ["runs", "--target", "y", "--best"]).exit_code == 0
    assert runner.invoke(app, ["runs", "--best"]).exit_code == 1
    assert runner.invoke(app, ["runs", "--compare", run_id]).exit_code == 1
    assert runner.invoke(app, ["runs", "--delete", run_id]).exit_code == 0
    assert "No runs yet" in " ".join(runner.invoke(app, ["runs"]).output.split())
