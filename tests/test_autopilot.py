import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import autopilot as ap
from kikolens.analyze import modellab as lab
from kikolens.cli.main import app
from kikolens.report.pdf import build_pdf

runner = CliRunner()


def sales(n=240, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    units = 50 + 0.1 * t + 8 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 2, n)
    return pd.DataFrame({
        "order_id": np.arange(1000, 1000 + n),
        "date": pd.date_range("2024-01-01", periods=n),
        "channel": rng.choice(["web", "store"], n),
        "price": [f"{v:.2f}" for v in rng.uniform(5, 20, n)],
        "units": units,
        "revenue": units * rng.uniform(9, 11, n),
    })


def test_column_roles_tell_identifiers_dates_and_text_apart():
    df = sales()
    df["note"] = [f"a long free text comment number {i} written by the customer" for i in range(len(df))]
    df["flat"] = 1
    df["date"] = pd.to_datetime(df["date"])
    roles = ap.column_roles(df)
    assert roles["order_id"] == "identifier" and roles["date"] == "date" and roles["flat"] == "constant"
    assert roles["units"] == "number" and roles["channel"] == "category" and roles["note"] in ("identifier", "text")


def test_suggest_target_prefers_typical_names_and_skips_identifiers():
    df = sales()
    df["date"] = pd.to_datetime(df["date"])
    assert ap.suggest_target(df) == "revenue"
    assert ap.suggest_target(df[["order_id", "date"]]) is None


def test_autopilot_runs_every_phase_and_records_a_replayable_recipe():
    df = sales()
    res = ap.run_autopilot(df, forecast_horizon=7)
    assert res["target"] == "revenue" and res["model"]["metrics"]["R2"] > 0.8
    assert res["forecast"]["result"]["forecast"].shape[0] == 7 and not res["problems"]
    replay, _ = res["recipe"].apply(df)
    assert pd.api.types.is_numeric_dtype(replay["price"])          # el texto con números se convirtió
    assert "order_id" in res["model"]["dropped"][0]["column"] or "order_id" not in res["model"]["features"]
    assert "revenue" in res["narrative"]
    assert ap.next_steps(res)


def test_autopilot_survives_a_dataset_it_cannot_model():
    df = pd.DataFrame({"a": ["x"] * 25 + ["y"] * 25, "b": range(50)})
    res = ap.run_autopilot(df)
    assert res["narrative"]
    with pytest.raises(ValueError):
        ap.run_autopilot(df[["a"]])
    with pytest.raises(ValueError):
        ap.run_autopilot(sales(), target="nope")


def test_advanced_and_ensemble_models_are_opt_in_and_work():
    assert not set(lab.ADVANCED) & set(lab.candidates("regression"))
    df = sales()
    X, y, info = lab.prepare_xy(df.drop(columns=["order_id", "date"]), "revenue")
    board = lab.compare_models(X, y, "regression", ["Ridge", "Stacking Ensemble", "Voting Ensemble", "AdaBoost"], folds=3)
    assert (board["status"] == "ok").all() and set(board["model"]) == {"Ridge", "Stacking Ensemble", "Voting Ensemble", "AdaBoost"}
    Xc, yc, _ = lab.prepare_xy(df.assign(big=(df["revenue"] > df["revenue"].median()).map({True: "hi", False: "lo"}))
                               .drop(columns=["order_id", "date", "revenue"]), "big")
    board = lab.compare_models(Xc, yc, "classification", ["Logistic Regression", "SVM", "Bagging", "Stacking Ensemble"], folds=3)
    assert (board["status"] == "ok").all()
    best, params, tuned, base = lab.tune_model(X, y, "regression", "SVM", n_iter=4, folds=3)
    assert "regressor__C" in params


def test_pdf_has_the_right_header_and_pages():
    df = sales()
    pdf = build_pdf(df, "sales")
    assert pdf.startswith(b"%PDF-") and pdf.count(b"/Type /Page\n") + pdf.count(b"/Type /Page ") >= 3
    with pytest.raises(ValueError):
        build_pdf(df.iloc[0:0])
    full = build_pdf(df, "sales", autopilot=ap.run_autopilot(df, forecast_horizon=5))
    assert len(full) > len(pdf)


def test_cli_autopilot_and_pdf_report(tmp_path):
    csv = tmp_path / "s.csv"
    sales().to_csv(csv, index=False)
    out = runner.invoke(app, ["autopilot", str(csv), "--clean", str(tmp_path / "c.csv"), "--recipe", str(tmp_path / "r.json"),
                              "--pdf", str(tmp_path / "a.pdf"), "--horizon", "5"])
    assert out.exit_code == 0, out.output
    assert (tmp_path / "c.csv").exists() and (tmp_path / "r.json").exists()
    assert (tmp_path / "a.pdf").read_bytes().startswith(b"%PDF-")
    out = runner.invoke(app, ["report", str(csv), "--output", str(tmp_path / "r.html"), "--pdf", str(tmp_path / "b.pdf")])
    assert out.exit_code == 0 and (tmp_path / "b.pdf").exists()
    assert runner.invoke(app, ["autopilot", str(csv), "--target", "nope"]).exit_code == 1
