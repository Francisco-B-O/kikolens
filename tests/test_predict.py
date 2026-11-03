import joblib
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import predict as pr
from kikolens.analyze.automl import run_automl_analysis
from kikolens.analyze.modellab import run_model_lab
from kikolens.analyze.neural import run_neural_network
from kikolens.cli.main import app
from kikolens.pipeline.recipe import Recipe
from kikolens.utils.dtypes import is_text

runner = CliRunner()


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 220
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "x": rng.normal(size=n),
        "z": rng.normal(size=n),
        "g": rng.choice(["a", "b", "c"], n),
    })
    df["y"] = 3 * df["x"] + (df["g"] == "a") * 5 + rng.normal(size=n) * 0.3
    df["label"] = np.where(df["y"] > df["y"].median(), "high", "low")
    df["flag"] = (df["y"] > df["y"].median()).astype(int)
    return df


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("KIKOLENS_HOME", str(tmp_path / "home"))
    return tmp_path / "home" / "models"


@pytest.fixture
def lab_bundle(data):
    frame = data.drop(columns=["label", "flag"])
    res = run_model_lab(frame, "y", models=["Ridge", "Random Forest"], folds=3, tune=False)
    return pr.build_bundle(res, frame, name="My lab"), frame


def new_rows(data, n=6):
    return data.head(n).drop(columns=["y", "label", "flag"])


def test_lab_bundle_predicts_new_rows_with_interval(lab_bundle, data):
    bundle, _ = lab_bundle
    out, report = pr.predict_frame(bundle, new_rows(data))
    assert {"predicted_y", "y_low", "y_high"} <= set(out.columns) and len(out) == 6
    assert (out["y_low"] < out["predicted_y"]).all() and (out["y_high"] > out["predicted_y"]).all()
    assert report["missing_columns"] == [] and report["unknown_categories"] == {}
    assert pr.input_warnings(report) == []
    assert set(bundle["schema"]) == {"date", "x", "z", "g"}


def test_predictions_match_the_training_pipeline(lab_bundle, data):
    bundle, frame = lab_bundle
    out, _ = pr.predict_frame(bundle, new_rows(data, 20))
    res_pred = bundle["estimator"].predict(pr._model_matrix(bundle, pr.prepare_inputs(bundle, new_rows(data, 20))[0]))
    assert np.allclose(out["predicted_y"], res_pred)


def test_input_problems_are_reported_not_fatal(lab_bundle, data):
    bundle, _ = lab_bundle
    messy = new_rows(data, 5).copy()
    messy["x"] = ["0.5", "abc", "0.2", "0.1", "-0.3"]          # un valor ilegible
    messy.loc[0, "g"] = "never-seen"                    # categoría nueva
    messy.loc[1, "z"] = 1e6                             # fuera de rango
    messy = messy.drop(columns=["date"])                # columna ausente
    out, report = pr.predict_frame(bundle, messy)
    assert len(out) == 5 and out["predicted_y"].notna().all()
    assert report["missing_columns"] == ["date"]
    assert report["coerced"] == {"x": 1}
    assert report["unknown_categories"]["g"]["values"] == ["never-seen"]
    assert report["out_of_range"] == {"z": 1}
    severities = [s for s, _ in pr.input_warnings(report)]
    assert "high" in severities and severities.count("medium") >= 3


def test_wrong_data_is_rejected_clearly(lab_bundle):
    bundle, _ = lab_bundle
    with pytest.raises(ValueError, match="None of the columns"):
        pr.predict_frame(bundle, pd.DataFrame({"nope": [1, 2]}))
    with pytest.raises(ValueError, match="no recipe"):
        pr.predict_frame(bundle, pd.DataFrame({"x": [1.0]}), apply_recipe=True)


def test_recipe_is_applied_before_predicting(data):
    frame = data.drop(columns=["label", "flag"])
    recipe = Recipe()
    recipe.add("add_column", {"name": "x2", "expression": "x * 2"})
    prepared, _ = recipe.apply(frame)
    res = run_model_lab(prepared, "y", models=["Ridge"], folds=3, tune=False)
    bundle = pr.build_bundle(res, prepared, recipe=recipe)
    out, _ = pr.predict_frame(bundle, new_rows(data), apply_recipe=True)   # llega sin la columna x2
    assert out["predicted_y"].notna().all()
    _, report = pr.predict_frame(bundle, new_rows(data), apply_recipe=False)
    assert "x2" in report["missing_columns"]


def test_classic_classifier_returns_original_labels(data):
    frame = data.drop(columns=["y", "flag"])
    bundle = pr.build_bundle(run_automl_analysis(frame, "label"), frame)
    out, _ = pr.predict_frame(bundle, frame.drop(columns="label").head(30))
    assert set(out["predicted_label"]) <= {"high", "low"}
    assert {"confidence", "probability_high", "probability_low"} <= set(out.columns)
    assert np.allclose(out["probability_high"] + out["probability_low"], 1)
    accuracy = (out["predicted_label"] == frame["label"].head(30)).mean()
    assert accuracy > 0.8


def test_classic_classifier_with_numeric_classes_is_not_decoded(data):
    frame = data.drop(columns=["y", "label"])
    bundle = pr.build_bundle(run_automl_analysis(frame, "flag"), frame)
    out, _ = pr.predict_frame(bundle, frame.drop(columns="flag").head(20))
    assert set(out["predicted_flag"]) <= {0, 1}


def test_classic_regression_and_neural_network(data):
    frame = data.drop(columns=["label", "flag"])
    for res in (run_automl_analysis(frame, "y"), run_neural_network(frame, "y", hidden_layers=(8,), max_iter=80)):
        bundle = pr.build_bundle(res, frame)
        out, report = pr.predict_frame(bundle, new_rows(data))
        assert out["predicted_y"].notna().all() and report["missing_columns"] == []
        assert "y_low" not in out  # solo el laboratorio guarda el margen de error


def test_input_data_is_not_modified(lab_bundle, data):
    bundle, _ = lab_bundle
    rows = new_rows(data)
    snapshot = rows.copy(deep=True)
    pr.predict_frame(bundle, rows)
    pd.testing.assert_frame_equal(rows, snapshot)


def test_registry_versions_lists_and_deletes(lab_bundle, registry):
    bundle, _ = lab_bundle
    first = pr.save_bundle(bundle)
    second = pr.save_bundle(bundle)
    assert first.name == "my-lab.klmodel" and second.name == "my-lab-v2.klmodel"
    frame = pr.list_models()
    assert sorted(frame["file"]) == ["my-lab", "my-lab-v2"] and set(frame["target"]) == {"y"}
    assert pr.load_bundle(pr.resolve_model("My Lab"))["name"] == "My lab"
    assert pr.load_bundle(pr.resolve_model("my-lab-v2.klmodel"))["name"] == "My lab (v2)"
    pr.delete_model("my-lab")
    assert list(pr.list_models()["file"]) == ["my-lab-v2"] and not first.exists()
    with pytest.raises(ValueError, match="No model called"):
        pr.resolve_model("ghost")


def test_cannot_delete_files_outside_the_registry(lab_bundle, tmp_path):
    bundle, _ = lab_bundle
    outside = pr.save_bundle(bundle, path=tmp_path / "elsewhere" / "m.klmodel")
    with pytest.raises(ValueError, match="inside the KikoLens registry"):
        pr.delete_model(str(outside))
    assert outside.exists()


def test_load_rejects_bare_or_missing_models(tmp_path):
    bare = tmp_path / "bare.klmodel"
    joblib.dump({"just": "a dict"}, bare)
    with pytest.raises(ValueError, match="bare model"):
        pr.load_bundle(bare)
    junk = tmp_path / "junk.klmodel"
    junk.write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="Could not read"):
        pr.load_bundle(junk)
    with pytest.raises(ValueError, match="not found"):
        pr.load_bundle(tmp_path / "missing.klmodel")


def test_empty_registry_and_model_card(lab_bundle):
    assert pr.list_models().empty
    card = pr.model_card(lab_bundle[0])
    assert "# My lab" in card and "Expected inputs" in card and "| g | categorical |" in card


def test_cli_register_score_and_manage(tmp_path, data):
    source = tmp_path / "data.csv"
    data.drop(columns=["label", "flag"]).to_csv(source, index=False)
    out = runner.invoke(app, ["lab", str(source), "y", "--folds", "3", "--no-tune", "--models", "Ridge",
                              "--register", "demo model", "--save", str(tmp_path / "copy.klmodel")])
    assert out.exit_code == 0 and (tmp_path / "copy.klmodel").exists()
    assert "demo-model" in runner.invoke(app, ["models"]).output

    result = tmp_path / "preds.csv"
    out = runner.invoke(app, ["score", str(source), "demo model", "--output", str(result)])
    assert out.exit_code == 0, out.output
    preds = pd.read_csv(result)
    assert "predicted_y" in preds.columns and len(preds) == len(data)

    info = runner.invoke(app, ["models", "info", "demo-model", "--card", str(tmp_path / "card.md")])
    assert info.exit_code == 0 and (tmp_path / "card.md").exists()
    assert runner.invoke(app, ["models", "delete", "demo-model"]).exit_code == 0
    assert runner.invoke(app, ["score", str(source), "demo-model"]).exit_code == 1
    assert runner.invoke(app, ["models", "info", "ghost"]).exit_code == 1
    assert "No saved models" in runner.invoke(app, ["models"]).output


def test_cli_predict_saves_a_scoreable_package(tmp_path, data):
    source = tmp_path / "d.csv"
    data.drop(columns=["label", "flag"]).to_csv(source, index=False)
    package = tmp_path / "m.klmodel"
    assert runner.invoke(app, ["predict", str(source), "y", "--save", str(package)]).exit_code == 0
    out = runner.invoke(app, ["score", str(source), str(package), "--output", str(tmp_path / "p.csv")])
    assert out.exit_code == 0 and "predictions of 'y'" in out.output
    bad = tmp_path / "other.csv"
    pd.DataFrame({"q": [1, 2]}).to_csv(bad, index=False)
    assert runner.invoke(app, ["score", str(bad), str(package)]).exit_code == 1


def test_dates_after_the_training_period_are_flagged(lab_bundle, data):
    bundle, _ = lab_bundle
    later = new_rows(data, 10).assign(date=pd.date_range("2030-01-01", periods=10))
    _, report = pr.predict_frame(bundle, later)
    assert report["out_of_range"].get("date") == 10
    inside = pr.predict_frame(bundle, new_rows(data, 10))[1]
    assert "date" not in inside["out_of_range"]


def test_models_use_dates_and_numbers_stored_as_text(data):
    from kikolens.analyze.quality import infer_types
    frame = data.drop(columns=["label", "flag", "g"]).assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"),
                                                             price=lambda d: [f"{v:,.0f}" for v in (d["z"] * 400 + 3000)])
    typed, converted = infer_types(frame)
    assert dict(converted) == {"date": "datetime", "price": "numeric"}
    assert pd.api.types.is_datetime64_any_dtype(typed["date"]) and pd.api.types.is_numeric_dtype(typed["price"])
    assert is_text(frame["date"])   # el original no cambia (object en pandas 2, str en pandas 3)
    res = run_model_lab(frame, "y", models=["Ridge"], folds=3, tune=False)
    assert res["converted"] == converted and "Read as dates: date" in res["narrative"]
    assert "date_days" in res["features"] and "price" in res["features"]
    bundle = pr.build_bundle(res, frame)
    assert bundle["schema"]["date"]["kind"] == "datetime" and bundle["schema"]["price"]["kind"] == "numeric"
    out, report = pr.predict_frame(bundle, frame.head(5).drop(columns="y"))   # el mismo texto de entrada
    assert out["predicted_y"].notna().all() and report["coerced"] == {}
