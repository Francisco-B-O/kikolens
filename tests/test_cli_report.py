import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.cli.main import app
from kikolens.report.builder import build_report, save_report
from kikolens.analyze.neural import run_neural_network

runner = CliRunner()


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n).astype(str),
        "x": rng.normal(size=n),
        "z": rng.normal(size=n),
        "group": rng.choice(["a", "b"], n),
    })
    df["y"] = 4 * df["x"] + 50 + np.arange(n) * 0.1
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_report_contains_sections_and_escapes_html():
    df = pd.DataFrame({"<script>alert(1)</script>": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.5]})
    page = build_report(df, name="<b>evil</b>", offline=False)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;b&gt;evil&lt;/b&gt;" in page
    for heading in ("Data quality", "Descriptive statistics", "Correlations"):
        assert heading in page
    assert "Produced locally" in page


def test_report_with_model_and_ai_text(classification_df):
    res = run_neural_network(classification_df, "label", hidden_layers=(8,), max_iter=100)
    page = build_report(classification_df, models={"Neural network": res}, ai_text="Looks fine <i>x</i>", offline=False)
    assert "Neural network" in page and "Accuracy" in page
    assert "&lt;i&gt;x&lt;/i&gt;" in page


def test_offline_report_embeds_plotly_once(tmp_path):
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) ** 2, "c": np.arange(10.0) * 3})
    path = save_report(df, tmp_path / "out" / "r.html")
    text = path.read_text(encoding="utf-8")
    assert text.count("plotly.js v") == 1 or text.count("window.PlotlyConfig") >= 1
    assert path.stat().st_size > 1_000_000


def test_empty_dataframe_report_does_not_crash():
    assert "Data quality" in build_report(pd.DataFrame({"a": []}), offline=False)


def test_cli_analyze_and_sql(csv_path):
    out = runner.invoke(app, ["analyze", csv_path])
    assert out.exit_code == 0 and "Quality score" in out.output
    out = runner.invoke(app, ["sql", csv_path, "SELECT COUNT(*) AS n FROM df"])
    assert out.exit_code == 0 and "120" in out.output


def test_cli_sql_rejects_writes(csv_path):
    out = runner.invoke(app, ["sql", csv_path, "DROP TABLE df"])
    assert out.exit_code == 1 and "Security" in out.output


def test_cli_missing_file_and_column(csv_path):
    assert runner.invoke(app, ["analyze", "nope.csv"]).exit_code == 1
    out = runner.invoke(app, ["predict", csv_path, "missing"])
    assert out.exit_code == 1 and "not found" in out.output


def test_cli_predict_and_save(csv_path, tmp_path):
    model_path = tmp_path / "m.pkl"
    out = runner.invoke(app, ["predict", csv_path, "y", "--save", str(model_path)])
    assert out.exit_code == 0 and model_path.exists()
    out = runner.invoke(app, ["predict", csv_path, "y", "--nn", "--layers", "8"])
    assert out.exit_code == 0 and "Neural network" in out.output


def test_cli_forecast(csv_path):
    out = runner.invoke(app, ["forecast", csv_path, "date", "y", "--horizon", "5"])
    assert out.exit_code == 0 and "Forecast" in out.output
    assert runner.invoke(app, ["forecast", csv_path, "date", "group"]).exit_code == 1


def test_cli_clean(csv_path, tmp_path):
    target = tmp_path / "clean.parquet"
    out = runner.invoke(app, ["clean", csv_path, "--output", str(target)])
    assert out.exit_code == 0 and target.exists()
    assert runner.invoke(app, ["clean", csv_path, "--output", str(tmp_path / "x.txt")]).exit_code == 1


def test_cli_report(csv_path, tmp_path):
    target = tmp_path / "r.html"
    out = runner.invoke(app, ["report", csv_path, "--output", str(target), "--target", "y"])
    assert out.exit_code == 0 and target.exists()
    assert "AutoML" in target.read_text(encoding="utf-8")


def test_cli_lab(csv_path, tmp_path):
    saved = tmp_path / "lab.pkl"
    out = runner.invoke(app, ["lab", csv_path, "y", "--folds", "3", "--no-tune", "--models", "Baseline (mean),Ridge",
                              "--save", str(saved)])
    assert out.exit_code == 0 and "Leaderboard" in out.output and saved.exists()
    assert runner.invoke(app, ["lab", csv_path, "nope"]).exit_code == 1
    assert runner.invoke(app, ["lab", csv_path, "y", "--folds", "2"]).exit_code == 1
