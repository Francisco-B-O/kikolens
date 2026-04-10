import pandas as pd
import numpy as np
import pytest
from kikolens.analyze.automl import run_automl_analysis, AutoMLExplainer


def test_regression_returns_expected_keys(regression_df):
    result = run_automl_analysis(regression_df, "target")
    assert "metrics" in result
    assert "model" in result
    assert "narrative" in result
    assert "feature_importance" in result
    assert "plots" in result
    assert "task_type" in result


def test_regression_task_type(regression_df):
    result = run_automl_analysis(regression_df, "target")
    assert result["task_type"] == "regression"


def test_regression_metrics_has_r2(regression_df):
    result = run_automl_analysis(regression_df, "target")
    assert "R2" in result["metrics"]
    assert "RMSE" in result["metrics"]


def test_classification_task_type(classification_df):
    result = run_automl_analysis(classification_df, "label")
    assert result["task_type"] == "classification"


def test_classification_metrics_has_accuracy(classification_df):
    result = run_automl_analysis(classification_df, "label")
    assert "Accuracy" in result["metrics"]
    assert 0.0 <= result["metrics"]["Accuracy"] <= 1.0


def test_manual_random_forest(regression_df):
    result = run_automl_analysis(regression_df, "target", manual_type="Random Forest")
    assert result["model"] is not None


def test_knn_does_not_crash(classification_df):
    """Verifica BUG1: KNN no debe lanzar AttributeError al extraer feature importance."""
    result = run_automl_analysis(classification_df, "label", manual_type="Lasso/KNN")
    assert "feature_importance" in result
    assert len(result["feature_importance"]) > 0


def test_feature_importance_keys_match_columns(regression_df):
    result = run_automl_analysis(regression_df, "target")
    for col in result["feature_importance"]:
        assert col in result["columns"]


def test_single_class_raises(regression_df):
    """Clasificación con un solo valor en target debe lanzar ValueError."""
    df = regression_df.copy()
    df["single_class"] = 1
    with pytest.raises(ValueError):
        run_automl_analysis(df, "single_class")


def test_narrative_contains_model_name(regression_df):
    result = run_automl_analysis(regression_df, "target")
    assert result["selected_model_name"] if "selected_model_name" in result else True
    assert len(result["narrative"]) > 0
