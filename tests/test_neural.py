import numpy as np
import pandas as pd
import pytest

from kikolens.analyze.neural import (architecture_figure, count_parameters,
                                     parse_hidden_layers, run_neural_network)


@pytest.fixture
def learnable_regression():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    return pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "c": x[:, 2],
                         "y": 3 * x[:, 0] - 2 * x[:, 1] + rng.normal(scale=0.1, size=300) + 100})


@pytest.mark.parametrize("raw,expected", [("64, 32", (64, 32)), ("16;8", (16, 8)), ([10, 5], (10, 5)), ("8", (8,))])
def test_parse_hidden_layers(raw, expected):
    assert parse_hidden_layers(raw) == expected


@pytest.mark.parametrize("raw", ["", "0", "-4, 8", "a,b"])
def test_parse_hidden_layers_invalid(raw):
    with pytest.raises(ValueError):
        parse_hidden_layers(raw)


def test_regression_learns_signal(learnable_regression):
    res = run_neural_network(learnable_regression, "y", hidden_layers=(16,), max_iter=500)
    assert res["task_type"] == "regression"
    assert res["metrics"]["R2"] > 0.8
    assert {"performance", "importance", "training", "architecture"} <= set(res["plots"])
    assert res["n_parameters"] == count_parameters(res["model"].regressor_)
    assert list(res["feature_importance"])[0] in ("a", "b")


def test_classification(classification_df):
    res = run_neural_network(classification_df, "label", hidden_layers="8", max_iter=200)
    assert res["task_type"] == "classification"
    assert 0 <= res["metrics"]["Accuracy"] <= 1
    assert res["architecture"][0] == 3


def test_result_is_compatible_with_what_if(learnable_regression):
    res = run_neural_network(learnable_regression, "y", hidden_layers=(8,), max_iter=100)
    row = pd.DataFrame([[0.0] * len(res["columns"])], columns=res["columns"])
    pred = res["model"].predict(res["scaler"].transform(res["imputer"].transform(row)))
    assert pred.shape == (1,)


def test_lbfgs_without_loss_curve(learnable_regression):
    res = run_neural_network(learnable_regression, "y", hidden_layers=(8,), solver="lbfgs", max_iter=100)
    assert res["loss_curve"] == []
    assert "R2" in res["metrics"]


def test_validation_errors(learnable_regression):
    with pytest.raises(ValueError, match="activation"):
        run_neural_network(learnable_regression, "y", activation="swish")
    with pytest.raises(ValueError, match="not found"):
        run_neural_network(learnable_regression, "missing")
    with pytest.raises(ValueError, match="20 rows"):
        run_neural_network(learnable_regression.head(10), "y")


def test_single_class_rejected():
    df = pd.DataFrame({"a": np.arange(50.0), "t": ["x"] * 50})
    with pytest.raises(ValueError, match="single class|at least 2"):
        run_neural_network(df, "t")


def test_architecture_figure_caps_nodes():
    fig = architecture_figure([100, 64, 1], max_nodes=10)
    assert len(fig.data) == 4
    assert len(fig.data[1].x) == 10
