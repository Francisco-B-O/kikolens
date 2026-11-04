import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze.modellab import run_model_lab
from kikolens.analyze.predict import build_bundle, save_bundle
from kikolens.cli.main import app
from kikolens.serve import MAX_ROWS, make_server


@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200), "g": rng.choice(["a", "b"], 200)})
    df["y"] = 3 * df["x"] + (df["g"] == "a") * 2 + rng.normal(0, 0.1, 200)
    res = run_model_lab(df, "y", models=["Ridge"], folds=3, tune=False)
    return build_bundle(res, df, name="served")


def start(bundle, token=None):
    server = make_server(bundle, "127.0.0.1", 0, token)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def call(url, body=None, token=None, raw=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_model_and_predict(bundle):
    server, url = start(bundle)
    try:
        assert call(url + "/health") == (200, {"status": "ok"})
        status, info = call(url + "/model")
        assert status == 200 and info["target"] == "y" and set(info["inputs"]) == {"x", "g"}
        status, out = call(url + "/predict", {"rows": [{"x": 1.0, "g": "a"}, {"x": -1.0, "g": "b"}]})
        assert status == 200 and len(out["predictions"]) == 2
        first, second = out["predictions"]
        assert abs(first["predicted_y"] - 5.0) < 0.5 and abs(second["predicted_y"] + 3.0) < 0.5
        assert "y_low" in first and out["warnings"] == []
        status, out = call(url + "/predict", {"rows": [{"x": 1.0, "g": "zzz"}]})
        assert status == 200 and out["warnings"]
    finally:
        server.shutdown()


def test_bad_requests_get_clear_errors(bundle):
    server, url = start(bundle)
    try:
        assert call(url + "/predict", raw=b"not json")[0] == 400
        assert call(url + "/predict", {"nope": 1})[0] == 400
        assert call(url + "/predict", {"rows": []})[0] == 400
        assert call(url + "/predict", {"rows": [1, 2]})[0] == 400
        status, out = call(url + "/predict", {"rows": [{"x": 1.0}] * (MAX_ROWS + 1)})
        assert status == 400 and "At most" in out["error"]
        assert call(url + "/missing")[0] == 404
        assert call(url + "/predict", raw=b"")[0] == 400
    finally:
        server.shutdown()


def test_token_is_required_when_set(bundle):
    server, url = start(bundle, token="s3cret")
    try:
        assert call(url + "/health")[0] == 200
        assert call(url + "/model")[0] == 401
        assert call(url + "/predict", {"rows": [{"x": 1, "g": "a"}]})[0] == 401
        assert call(url + "/predict", {"rows": [{"x": 1, "g": "a"}]}, token="wrong")[0] == 401
        assert call(url + "/predict", {"rows": [{"x": 1, "g": "a"}]}, token="s3cret")[0] == 200
    finally:
        server.shutdown()


def test_network_address_without_token_is_refused(bundle):
    with pytest.raises(ValueError, match="token"):
        make_server(bundle, "0.0.0.0", 0)
    make_server(bundle, "0.0.0.0", 0, token="t").server_close()


def test_cli_serve_refuses_an_open_network_address(bundle, tmp_path):
    path = save_bundle(bundle, path=tmp_path / "m.klmodel")
    out = CliRunner().invoke(app, ["serve", str(path), "--host", "0.0.0.0"])
    assert out.exit_code == 1 and "token" in out.output


def test_openapi_describes_the_inputs_and_needs_the_token(bundle):
    server, url = start(bundle, token="t")
    try:
        assert call(url + "/openapi.json")[0] == 401
        status, spec = call(url + "/openapi.json", token="t")
        assert status == 200 and spec["openapi"].startswith("3.")
        props = spec["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]["schema"]["properties"]["rows"]["items"]["properties"]
        assert props["x"] == {"type": "number"} and props["g"] == {"type": "string"}
    finally:
        server.shutdown()
