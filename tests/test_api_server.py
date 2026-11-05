import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from kikolens import api_server
from kikolens.analyze import predict as pr
from kikolens.analyze.modellab import run_model_lab


def call(server, method, path, body=None, token=None, raw=None):
    url = f"http://127.0.0.1:{server.server_address[1]}{path}"
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read()
            return response.status, (json.loads(text) if text else None)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


@pytest.fixture
def started(tmp_path):
    made = []

    def start(**kwargs):
        server = api_server.make_server("127.0.0.1", 0, **kwargs)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        made.append(server)
        return server

    yield start
    for server in made:
        server.shutdown()
        server.server_close()
        server.store.pool.shutdown(wait=False, cancel_futures=True)


@pytest.fixture
def rows():
    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(size=n)
    return [{"x": float(a), "z": float(b), "y": float(3 * a + b + rng.normal(scale=0.3)), "g": "ab"[i % 2],
             "day": f"2024-{1 + i // 30:02d}-{1 + i % 28:02d}"} for i, (a, b) in enumerate(zip(x, rng.normal(size=n)))]


def test_health_needs_no_token_and_everything_else_does(started):
    server = started(token="s3cret")
    assert call(server, "GET", "/health") == (200, {"status": "ok"})
    assert call(server, "GET", "/functions")[0] == 401
    assert call(server, "GET", "/functions", token="wrong")[0] == 401
    assert call(server, "POST", "/datasets", {"rows": [{"a": 1}]})[0] == 401
    assert call(server, "GET", "/functions", token="s3cret")[0] == 200


def test_a_network_address_requires_a_token():
    with pytest.raises(ValueError, match="needs a token"):
        api_server.make_server("0.0.0.0", 0)
    with pytest.raises(ValueError, match="does not exist"):
        api_server.make_server("127.0.0.1", 0, data_dir="/definitely/not/here")


def test_datasets_can_be_created_inspected_and_deleted(started, rows):
    server = started()
    status, made = call(server, "POST", "/datasets", {"rows": rows, "name": "demo"})
    assert status == 200 and made["rows"] == 120 and made["columns"] == ["x", "z", "y", "g", "day"]
    ident = made["id"]
    status, info = call(server, "GET", f"/datasets/{ident}")
    assert status == 200 and info["name"] == "demo" and len(info["head"]) == 5 and {c["name"] for c in info["columns"]} == {"x", "z", "y", "g", "day"}
    csv_id = call(server, "POST", "/datasets", {"csv": "a,b\n1,2\n3,4\n"})[1]["id"]
    assert call(server, "GET", "/datasets")[1]["datasets"][1] == {"id": csv_id, "name": "dataset", "rows": 2, "columns": 2}
    assert call(server, "DELETE", f"/datasets/{csv_id}")[1] == {"deleted": csv_id}
    assert call(server, "GET", f"/datasets/{csv_id}")[0] == 404
    for body, fragment in (({}, "Send 'rows'"), ({"rows": []}, "non-empty"), ({"rows": [1]}, "non-empty"), ({"csv": ""}, "could not be read"), ({"file": "x.csv"}, "data folder")):
        status, error = call(server, "POST", "/datasets", body)
        assert status == 400 and fragment in error["error"], (body, error)
    assert call(server, "POST", "/datasets", raw=b"{nope")[0] == 400
    assert call(server, "POST", "/datasets", raw=b"[1, 2]")[0] == 400


def test_files_are_read_only_from_the_data_folder(started, tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(folder / "t.csv", index=False)
    (tmp_path / "secret.csv").write_text("a\n9\n", encoding="utf-8")
    server = started(data_dir=folder)
    assert call(server, "POST", "/datasets", {"file": "t.csv"})[1]["rows"] == 3
    for name in ("../secret.csv", str(tmp_path / "secret.csv"), "missing.csv", "..\\secret.csv"):
        assert call(server, "POST", "/datasets", {"file": name})[0] == 404, name


def test_limits_on_datasets_and_body_size(started, monkeypatch):
    server = started()
    monkeypatch.setattr(api_server, "MAX_DATASETS", 2)
    for _ in range(2):
        assert call(server, "POST", "/datasets", {"rows": [{"a": 1}]})[0] == 200
    assert call(server, "POST", "/datasets", {"rows": [{"a": 1}]})[0] == 409
    monkeypatch.setattr(api_server, "MAX_BODY", 50)
    status, error = call(server, "POST", "/datasets", {"rows": [{"a": "x" * 100}]})
    assert status == 413 and "bytes" in error["error"]


def test_functions_run_with_json_results_and_clear_errors(started, rows):
    server = started()
    ident = call(server, "POST", "/datasets", {"rows": rows})[1]["id"]
    listing = {f["name"]: f for f in call(server, "GET", "/functions")[1]["functions"]}
    assert {"profile", "regress", "effect", "forecast"} <= set(listing) and "target" in listing["regress"]["parameters"]
    assert not {"load", "save_model", "analyst", "score"} & set(listing)
    status, out = call(server, "POST", "/run/regress", {"dataset": ident, "args": {"target": "y", "features": ["x", "z"]}})
    assert status == 200 and out["function"] == "regress" and "summary" in out["result"] and out["seconds"] >= 0
    table = out["result"]["table"]
    assert set(table) == {"columns", "rows", "total_rows"} and table["total_rows"] >= 3
    assert "figure" in out["omitted"] or "figure" not in out["result"]
    with_figures = call(server, "POST", "/run/regress?figures=1", {"dataset": ident, "args": {"target": "y"}})[1]["result"]
    assert any(isinstance(v, dict) and "data" in v for v in with_figures.values())
    assert call(server, "POST", "/run/profile", {"dataset": ident})[1]["result"]["summary"]
    assert call(server, "POST", "/run/ghost", {"dataset": ident})[0] == 404
    assert call(server, "POST", "/run/regress", {"dataset": "nothing", "args": {}})[0] == 404
    status, error = call(server, "POST", "/run/regress", {"dataset": ident, "args": {"target": "missing"}})
    assert status == 400 and "missing" in error["error"]
    status, error = call(server, "POST", "/run/regress", {"dataset": ident, "args": {"nope": 1}})
    assert status == 400 and "Invalid arguments" in error["error"]
    assert call(server, "POST", "/run/regress", {"dataset": ident, "args": "x"})[0] == 400
    assert call(server, "POST", "/run/query", {"dataset": ident, "args": {"sql": "select * from read_csv('/etc/passwd')"}})[0] == 400


def test_jobs_run_in_the_background(started, rows):
    server = started()
    ident = call(server, "POST", "/datasets", {"rows": rows})[1]["id"]
    status, job = call(server, "POST", "/jobs", {"function": "regress", "dataset": ident, "args": {"target": "y"}})
    assert status == 200 and job["status"] == "queued"
    for _ in range(100):
        state = call(server, "GET", f"/jobs/{job['id']}")[1]
        if state["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert state["status"] == "done" and state["result"]["result"]["summary"]
    bad = call(server, "POST", "/jobs", {"function": "regress", "dataset": ident, "args": {"target": "nope"}})[1]
    for _ in range(100):
        failed = call(server, "GET", f"/jobs/{bad['id']}")[1]
        if failed["status"] != "queued" and failed["status"] != "running":
            break
        time.sleep(0.1)
    assert failed["status"] == "failed" and "nope" in failed["error"]
    assert call(server, "GET", "/jobs/ffffffffffff")[0] == 404
    assert call(server, "POST", "/jobs", {"function": "ghost", "dataset": ident})[0] == 404


def test_models_are_served_only_when_named_at_startup(started, rows, tmp_path, monkeypatch):
    monkeypatch.setenv("KIKOLENS_HOME", str(tmp_path / "home"))
    frame = pd.DataFrame(rows).drop(columns=["day"])
    bundle = pr.build_bundle(run_model_lab(frame, "y", models=["Ridge"], folds=3, tune=False), frame, name="demo")
    server = started(models={"demo": bundle})
    status, listing = call(server, "GET", "/models")
    assert status == 200 and listing["models"][0]["name"] == "demo" and listing["models"][0]["target"] == "y"
    status, out = call(server, "POST", "/models/demo/predict", {"rows": [{"x": 1.0, "z": 0.0, "g": "a"}, {"x": -1.0, "z": 0.0, "g": "b"}]})
    assert status == 200 and len(out["predictions"]) == 2 and out["predictions"][0]["predicted_y"] > out["predictions"][1]["predicted_y"]
    assert call(server, "POST", "/models/other/predict", {"rows": [{"x": 1}]})[0] == 404
    assert call(server, "POST", "/models/demo/predict", {"rows": []})[0] == 400
    assert call(server, "POST", "/models/demo/predict", {"rows": [{"x": 1, "z": 0, "g": "a"}] * 10_001})[0] == 400


def test_openapi_cors_and_unknown_routes(started):
    server = started(cors="https://example.test")
    spec = call(server, "GET", "/openapi.json")[1]
    assert spec["openapi"].startswith("3") and {"/run/{function}", "/jobs", "/datasets", "/models/{name}/predict"} <= set(spec["paths"])
    request = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}/functions", method="OPTIONS")
    with urllib.request.urlopen(request, timeout=10) as response:
        assert response.status == 204 and response.headers["Access-Control-Allow-Origin"] == "https://example.test"
    assert call(server, "GET", "/nothing/here")[0] == 404
    assert call(server, "DELETE", "/datasets/UPPERCASE123")[0] == 404


def test_results_are_made_json_safe():
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "t": pd.date_range("2024-01-01", periods=3), "s": ["x", None, "z"]})
    out, omitted = api_server.to_json({"table": frame, "n": np.int64(3), "f": np.float64("nan"), "arr": np.arange(3), "obj": object(), "_hidden": 1,
                                       "nested": {"k": np.float32(1.5), "when": pd.Timestamp("2024-05-01")}, "series": pd.Series([1, 2], name="v")})
    assert json.dumps(out) and omitted == ["obj"] and out["n"] == 3 and out["f"] is None and out["arr"] == [0, 1, 2] and "_hidden" not in out
    assert out["table"]["rows"][1]["a"] is None and out["table"]["rows"][0]["t"].startswith("2024-01-01") and out["nested"]["when"].startswith("2024-05-01")
    big, _ = api_server.to_json({"t": pd.DataFrame({"a": range(1000)})}, limit=10)
    assert len(big["t"]["rows"]) == 10 and big["t"]["total_rows"] == 1000


def test_the_api_command_refuses_a_network_address_without_a_token_and_a_missing_model():
    from typer.testing import CliRunner
    from kikolens.cli.main import app
    runner = CliRunner()
    out = runner.invoke(app, ["api", "--host", "0.0.0.0", "--port", "0"])
    assert out.exit_code == 1 and "needs a token" in out.output
    out = runner.invoke(app, ["api", "--model", "no-such-model", "--port", "0"])
    assert out.exit_code == 1 and "No model called" in out.output
