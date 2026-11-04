import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from typer.testing import CliRunner

from kikolens import benchmark, config, doctor
from kikolens.cli.main import app

runner = CliRunner()


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for name in ("OLLAMA_HOST", "OLLAMA_MODEL", "OLLAMA_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KIKOLENS_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ── Configuración ────────────────────────────────────────────────────────────
def test_defaults_without_any_file(clean_env):
    assert config.ollama_host() == config.DEFAULT_HOST and config.ollama_model() == config.DEFAULT_MODEL
    assert config.dashboard_port() == 8501 and config.default_language() == "en" and config.problems() == []


def test_file_in_the_current_folder_overrides_the_user_file_and_env_beats_both(clean_env, monkeypatch):
    home = clean_env / "home"
    home.mkdir()
    (home / "config.toml").write_text('[ollama]\nmodel = "from-home"\nhost = "http://home:1"\n[dashboard]\nport = 9000\nlanguage = "es"\n')
    (clean_env / "kikolens.toml").write_text('[ollama]\nmodel = "from-cwd"\n')
    assert config.ollama_model() == "from-cwd" and config.ollama_host() == "http://home:1"
    assert config.dashboard_port() == 9000 and config.default_language() == "es"
    monkeypatch.setenv("OLLAMA_MODEL", "from-env")
    assert config.ollama_model() == "from-env"


def test_bad_values_unknown_keys_and_broken_files_never_stop_startup(clean_env):
    (clean_env / "kikolens.toml").write_text('[ollama]\ntimeout = "slow"\nmodle = "x"\n[dashboard]\nport = "abc"\nlanguage = "fr"\n[other]\na = 1\n')
    assert config.ollama_timeout(60) == 60 and config.dashboard_port() == 8501 and config.default_language() == "en"
    found = " ".join(config.problems())
    assert "unknown key 'modle'" in found and "unknown section [other]" in found
    (clean_env / "kikolens.toml").write_text("this is [not toml")
    assert config.settings() == {} and "not valid TOML" in " ".join(config.problems())
    assert config.ollama_model() == config.DEFAULT_MODEL


def test_start_uses_the_configured_port(clean_env, monkeypatch):
    (clean_env / "kikolens.toml").write_text("[dashboard]\nport = 9123\n")
    calls = []
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    assert runner.invoke(app, ["start"]).exit_code == 0 and "9123" in calls[-1]
    assert runner.invoke(app, ["start", "--port", "9555"]).exit_code == 0 and "9555" in calls[-1]


# ── doctor ───────────────────────────────────────────────────────────────────
def test_this_environment_passes_the_required_checks(clean_env):
    checks = doctor.run_checks(network=False)
    assert doctor.check_python()["status"] == "ok"
    assert all(c["status"] != "error" for c in checks) and doctor.summary(checks).startswith("Everything required")
    assert {"pandas", "streamlit", "torch"} <= {c["name"] for c in checks}


def test_missing_and_old_packages_are_errors_and_missing_extras_only_warnings(clean_env, monkeypatch):
    real = doctor._installed
    monkeypatch.setattr(doctor, "_installed", lambda name: None if name == "duckdb" else ("0.1" if name == "pandas" else real(name)))
    by_name = {c["name"]: c for c in doctor.check_required()}
    assert by_name["duckdb"]["status"] == "error" and "not installed" in by_name["duckdb"]["detail"]
    assert by_name["pandas"]["status"] == "error" and "older" in by_name["pandas"]["detail"]
    monkeypatch.setattr(doctor, "_installed", lambda name: None)
    assert all(c["status"] == "warning" for c in doctor.check_optional())
    assert "must be fixed" in doctor.summary([{"status": "error"}])


def test_version_tuple_handles_odd_versions():
    assert doctor._version_tuple("2.15.0+cpu") == (2, 15, 0) and doctor._version_tuple("1.4.1.post1") == (1, 4, 1)
    assert doctor._version_tuple("3.1.0rc1") > doctor._version_tuple("3.0.9") and doctor._version_tuple("junk") == ()


def test_unwritable_user_folder_is_an_error(clean_env, monkeypatch):
    blocker = clean_env / "file"
    blocker.write_text("x")
    monkeypatch.setenv("KIKOLENS_HOME", str(blocker / "inside"))
    assert doctor.check_home()["status"] == "error"


class FakeOllama(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"models": [{"name": "llama3.2:latest"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_ollama_reachable_missing_model_and_unreachable(clean_env, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{server.server_port}")
        assert doctor.check_ollama()["status"] == "ok"
        monkeypatch.setenv("OLLAMA_MODEL", "mistral")
        missing = doctor.check_ollama()
        assert missing["status"] == "warning" and "ollama pull mistral" in missing["detail"]
    finally:
        server.shutdown()
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    assert doctor.check_ollama(timeout=0.5)["status"] == "warning"


def test_doctor_command_table_json_and_exit_code(clean_env, monkeypatch):
    out = runner.invoke(app, ["doctor", "--offline"])
    assert out.exit_code == 0 and "KikoLens doctor" in out.output and "pandas" in out.output
    as_json = runner.invoke(app, ["doctor", "--offline", "--json"])
    assert as_json.exit_code == 0 and {c["status"] for c in json.loads(as_json.output)} <= {"ok", "warning", "error"}
    monkeypatch.setattr(doctor, "_installed", lambda name: None)
    assert runner.invoke(app, ["doctor", "--offline"]).exit_code == 1


# ── benchmark ────────────────────────────────────────────────────────────────
def test_benchmark_times_operations_and_filters(clean_env):
    res = benchmark.run_benchmark(600, only=["profile", "sql"])
    assert list(res["table"]["operation"]) == ["profile quality", "SQL aggregation"] and (res["table"]["status"] == "ok").all()
    assert (res["table"]["seconds"] >= 0).all() and res["total"] >= 0 and "2 operations" in res["summary"]
    with pytest.raises(ValueError, match="200 rows"):
        benchmark.run_benchmark(50)
    with pytest.raises(ValueError, match="repeat"):
        benchmark.run_benchmark(500, repeat=0)
    with pytest.raises(ValueError, match="matches"):
        benchmark.run_benchmark(500, only=["nothing"])


def test_benchmark_survives_a_failing_operation(clean_env, monkeypatch):
    monkeypatch.setattr(benchmark, "_operations", lambda df: {"broken": lambda: 1 / 0, "fine": lambda: None})
    table = benchmark.run_benchmark(300)["table"]
    assert table.loc[table["operation"] == "broken", "status"].iloc[0].startswith("failed")
    assert table.loc[table["operation"] == "fine", "status"].iloc[0] == "ok"


def test_benchmark_command_writes_json(clean_env):
    out = clean_env / "bench.json"
    res = runner.invoke(app, ["benchmark", "--rows", "400", "--only", "profile", "--output", str(out)])
    assert res.exit_code == 0 and json.loads(out.read_text())["results"][0]["operation"] == "profile quality"
    assert runner.invoke(app, ["benchmark", "--rows", "10"]).exit_code == 1
