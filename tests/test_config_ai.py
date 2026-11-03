import pandas as pd
import pytest
import requests

from kikolens import config
from kikolens.ai.insights import get_ai_insights
from kikolens.sql.generator import _extract_sql, generate_sql_from_text


class _Resp:
    def __init__(self, text):
        self._text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self._text}


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "5")
    assert config.ollama_generate_url() == "http://ollama:11434/api/generate"
    assert config.ollama_model() == "mistral"
    assert config.ollama_timeout() == 5


def test_config_defaults(monkeypatch):
    for var in ("OLLAMA_HOST", "OLLAMA_MODEL", "OLLAMA_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)
    assert config.ollama_model() == "llama3.2"
    assert config.ollama_host() == "http://localhost:11434"
    assert config.ollama_timeout() == 120


def test_config_invalid_timeout(monkeypatch):
    monkeypatch.setenv("OLLAMA_TIMEOUT", "abc")
    assert config.ollama_timeout() == 120


@pytest.mark.parametrize("raw,expected", [
    ("```sql\nSELECT 1\n```", "SELECT 1"),
    ("```\nSELECT 2\n```", "SELECT 2"),
    ("  SELECT 3  ", "SELECT 3"),
])
def test_extract_sql(raw, expected):
    assert _extract_sql(raw) == expected


def test_generate_sql_uses_default_model(monkeypatch, df):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(json=json, timeout=timeout)
        return _Resp("```sql\nSELECT * FROM df\n```")

    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr(requests, "post", fake_post)
    assert generate_sql_from_text(df, "all rows") == "SELECT * FROM df"
    assert sent["json"]["model"] == "llama3.2"
    assert sent["timeout"] == 120


def test_generate_sql_reports_errors(monkeypatch, df):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    assert generate_sql_from_text(df, "q").startswith("-- Error generating SQL")


def test_insights_success_and_timeout(monkeypatch, df):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp("- insight"))
    assert get_ai_insights(df) == "- insight"

    def timeout(*a, **k):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(requests, "post", timeout)
    assert "Timeout" in get_ai_insights(df)


def test_streamlit_config_file_matches_options():
    import tomllib
    from pathlib import Path

    def flatten(d, prefix=""):
        for k, v in d.items():
            yield from flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else [(f"{prefix}{k}", v)]

    toml = tomllib.loads((Path(__file__).parent.parent / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert dict(flatten(toml)) == config.STREAMLIT_OPTIONS


def test_streamlit_flags_are_pairs():
    flags = config.streamlit_flags()
    assert len(flags) == 2 * len(config.STREAMLIT_OPTIONS)
    assert flags[flags.index("--browser.gatherUsageStats") + 1] == "false"
