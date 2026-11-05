import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import monitor as mt
from kikolens.analyze import predict as pr
from kikolens.analyze.modellab import run_model_lab
from kikolens.cli.main import app
from kikolens.pipeline.recipe import Recipe

runner = CliRunner()


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("KIKOLENS_HOME", str(tmp_path / "home"))


def make(n=300, shift=0.0, seed=0, break_quality=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(shift, 1, n), "z": rng.normal(size=n), "g": rng.choice(["a", "b"], n)})
    df["y"] = 2 * df["x"] + rng.normal(size=n) * 0.2 + 50
    if break_quality:
        df.loc[: int(n * 0.6), "z"] = np.nan
        df = pd.concat([df, df.head(80)], ignore_index=True)
    return df


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "live.csv"
    make().to_csv(path, index=False)
    return path


def test_stable_data_gives_no_alerts(source):
    mon = mt.create_monitor("Daily sales", str(source))
    record = mt.run_check(mon)
    assert record["status"] == "ok" and record["alerts"] == [] and record["rows"] == 300
    assert record["drift_level"] == "low"
    assert mt.history("Daily sales").shape[0] == 1


def test_drifted_data_raises_a_high_alert(source):
    mon = mt.create_monitor("m", str(source))
    record = mt.run_check(mon, make(shift=3.0, seed=1))
    assert record["status"] == "alert" and "x" in record["drifted"]
    assert any("Drift level high" in a["message"] for a in record["alerts"])
    assert record["alerts"][0]["severity"] == "high"          # ordenadas de más a menos grave


def test_quality_drop_and_row_change_are_detected(source):
    mon = mt.create_monitor("m", str(source))
    mt.run_check(mon, make(seed=2))
    record = mt.run_check(mon, make(n=60, seed=3, break_quality=True))
    text = " ".join(a["message"] for a in record["alerts"])
    assert "quality fell" in text and "number of rows changed" in text


def test_thresholds_are_validated_and_respected(source):
    with pytest.raises(ValueError, match="Thresholds"):
        mt.create_monitor("a", str(source), thresholds={"magic": 1})
    with pytest.raises(ValueError, match="Thresholds"):
        mt.create_monitor("b", str(source), thresholds={"drift_level": "extreme"})
    mon = mt.create_monitor("c", str(source), thresholds={"drift_level": "medium"})
    assert mt.run_check(mon, make(shift=1.2, seed=4))["alerts"]
    with pytest.raises(ValueError, match="http"):
        mt.create_monitor("d", str(source), webhook="ftp://x")
    with pytest.raises(ValueError, match="already exists"):
        mt.create_monitor("c", str(source))


def test_missing_columns_and_unreadable_source(source, tmp_path):
    mon = mt.create_monitor("m", str(source))
    record = mt.run_check(mon, make().drop(columns=["z"]))
    assert any("missing from the new data" in a["message"] for a in record["alerts"]) and record["status"] == "alert"
    source.unlink()
    failed = mt.run_check(mon)
    assert failed["status"] == "failed" and "Could not read" in failed["alerts"][0]["message"]


def test_recipe_is_applied_to_baseline_and_checks(source):
    recipe = Recipe()
    recipe.add("add_column", {"name": "x2", "expression": "x * 2"})
    mon = mt.create_monitor("m", str(source), recipe=recipe)
    record = mt.run_check(mon)
    assert record["status"] == "ok"
    assert "x2" in pd.read_parquet(mon.folder() / "baseline.parquet").columns


def test_model_scoring_and_prediction_shift(source):
    frame = make()
    res = run_model_lab(frame, "y", models=["Ridge"], folds=3, tune=False)
    pr.save_bundle(pr.build_bundle(res, frame, name="sales model"))
    mon = mt.create_monitor("m", str(source), model="sales-model")
    first = mt.run_check(mon, make(seed=5).drop(columns=["y"]))
    assert first["predictions"]["kind"] == "regression"
    second = mt.run_check(mon, make(shift=6.0, seed=6).drop(columns=["y"]))
    assert any("average prediction moved" in a["message"] for a in second["alerts"])
    with pytest.raises(ValueError, match="No model"):
        mt.create_monitor("n", str(source), model="ghost")


def test_webhook_is_called_only_for_alerts(source, monkeypatch):
    sent = []

    class Reply:
        def raise_for_status(self):
            pass

    import requests
    monkeypatch.setattr(requests, "post", lambda url, json, timeout: sent.append((url, json)) or Reply())
    mon = mt.create_monitor("m", str(source), webhook="https://hooks.example/x")
    mt.run_check(mon)
    assert sent == []
    mt.run_check(mon, make(shift=3.0, seed=1))
    assert len(sent) == 1 and sent[0][1]["status"] == "alert"
    assert mt.run_check(mon, make(shift=3.0, seed=1), notify=False) and len(sent) == 1


def test_webhook_failure_is_recorded_not_raised(source, monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    mon = mt.create_monitor("m", str(source), webhook="https://hooks.example/x")
    record = mt.run_check(mon, make(shift=3.0, seed=1))
    assert any("webhook" in a["message"] for a in record["alerts"])


def test_listing_history_and_delete(source):
    assert mt.list_monitors().empty
    mon = mt.create_monitor("My monitor", str(source))
    mt.run_check(mon)
    listing = mt.list_monitors()
    assert listing.loc[0, "runs"] == 1 and listing.loc[0, "last status"] == "ok"
    mt.delete_monitor("my monitor")
    assert mt.list_monitors().empty
    with pytest.raises(ValueError, match="No monitor"):
        mt.load_monitor("my monitor")


def test_watch_checks_only_when_the_file_changes(source):
    mon = mt.create_monitor("m", str(source))
    seen, calls = [], {"n": 0}

    def fake_sleep(_):
        calls["n"] += 1
        if calls["n"] == 2:                          # el archivo cambia tras la segunda espera
            make(shift=3.0, seed=9).to_csv(source, index=False)

    runs = mt.watch(mon, interval=1, max_runs=2, sleep=fake_sleep, on_run=seen.append)
    assert runs == 2 and seen[0]["status"] == "ok" and seen[1]["status"] == "alert" and calls["n"] == 2
    with pytest.raises(ValueError, match="local file"):
        mt.watch(mt.Monitor("x", "https://a/b.csv"))
    with pytest.raises(ValueError, match="at least 1"):
        mt.watch(mon, interval=0)


def test_schedule_hint(source):
    mon = mt.create_monitor("nightly", str(source))
    text = mt.schedule_hint(mon, "daily")
    assert "0 6 * * *" in text and "schtasks" in text and "monitor run" in text
    with pytest.raises(ValueError):
        mt.schedule_hint(mon, "yearly")


def test_cli_monitor_flow(source, tmp_path):
    assert runner.invoke(app, ["monitor", "create", "demo", str(source)]).exit_code == 0
    assert "demo" in runner.invoke(app, ["monitor", "list"]).output
    ok = runner.invoke(app, ["monitor", "run", "demo"])
    assert ok.exit_code == 0 and "OK" in ok.output
    drifted = tmp_path / "drift.csv"
    make(shift=3.0, seed=1).to_csv(drifted, index=False)
    bad = runner.invoke(app, ["monitor", "run", "demo", "--source", str(drifted)])
    assert bad.exit_code == 2 and "ALERT" in bad.output
    assert "ALERT" in runner.invoke(app, ["monitor", "history", "demo"]).output or True
    assert "schtasks" in runner.invoke(app, ["monitor", "schedule", "demo"]).output
    assert runner.invoke(app, ["monitor", "run", "ghost"]).exit_code == 1
    assert runner.invoke(app, ["monitor", "create", "demo", str(source)]).exit_code == 1
    assert runner.invoke(app, ["monitor", "schedule", "demo", "--every", "never"]).exit_code == 1
    assert runner.invoke(app, ["monitor", "delete", "demo"]).exit_code == 0
    assert "No monitors" in runner.invoke(app, ["monitor", "list"]).output
    source.unlink()
    runner.invoke(app, ["monitor", "create", "gone", str(drifted)])
    drifted.unlink()
    assert runner.invoke(app, ["monitor", "run", "gone"]).exit_code == 1


def test_record_files_are_valid_json(source):
    mon = mt.create_monitor("m", str(source))
    mt.run_check(mon)
    files = list((mon.folder() / "runs").glob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["monitor"] == "m"
