import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import modellab as lab
from kikolens.analyze import torchnet as tn


def test_module_imports_and_reports_availability_without_torch():
    assert isinstance(tn.torch_available(), bool)
    names = lab.advanced_names()
    assert set(lab.ADVANCED) <= set(names) and (tn.NAME in names) == tn.torch_available()


@pytest.mark.skipif(tn.torch_available(), reason="only meaningful when PyTorch is missing")
def test_training_without_torch_explains_how_to_install():
    with pytest.raises(ImportError, match=r"kikolens\[torch\]"):
        tn.TorchMLPRegressor().fit(np.zeros((10, 2)), np.zeros(10))


torch = pytest.importorskip("torch")


def regression_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 2 * X[:, 0] - X[:, 1] ** 2 + rng.normal(0, 0.05, n)
    return X, y


def test_regressor_learns_a_nonlinear_function_and_is_reproducible():
    X, y = regression_data()
    model = tn.TorchMLPRegressor(hidden=(32, 16), epochs=150, seed=1).fit(X[:300], y[:300])
    pred = model.predict(X[300:])
    r2 = 1 - ((y[300:] - pred) ** 2).sum() / ((y[300:] - y[300:].mean()) ** 2).sum()
    assert r2 > 0.8 and len(model.history_) >= 1
    again = tn.TorchMLPRegressor(hidden=(32, 16), epochs=150, seed=1).fit(X[:300], y[:300]).predict(X[300:])
    assert np.allclose(pred, again, atol=1e-5)


def test_classifier_probabilities_and_classes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    y = np.where(X[:, 0] + X[:, 1] > 0, "up", "down")
    model = tn.TorchMLPClassifier(hidden=(16,), epochs=100).fit(X[:400], y[:400])
    proba = model.predict_proba(X[400:])
    assert proba.shape == (100, 2) and np.allclose(proba.sum(axis=1), 1) and list(model.classes_) == ["down", "up"]
    assert (model.predict(X[400:]) == y[400:]).mean() > 0.9


def test_validation_errors():
    X, y = regression_data(60)
    with pytest.raises(ValueError):
        tn.TorchMLPRegressor(epochs=0).fit(X, y)
    with pytest.raises(ValueError):
        tn.TorchMLPRegressor().fit(X, np.where(np.arange(60) == 3, np.nan, y))
    with pytest.raises(ValueError):
        tn.TorchMLPClassifier().fit(X, np.zeros(60))
    model = tn.TorchMLPRegressor(epochs=5).fit(X, y)
    with pytest.raises(ValueError, match="columns"):
        model.predict(X[:, :2])


def test_model_joins_the_lab_and_survives_a_model_package():
    from kikolens.analyze.predict import build_bundle, predict_frame
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300), "g": rng.choice(["u", "v"], 300)})
    df["y"] = 3 * df["a"] - df["b"] + (df["g"] == "u")
    res = lab.run_model_lab(df, "y", models=["Ridge", tn.NAME], folds=3, tune=False)
    assert set(res["leaderboard"]["model"]) == {"Ridge", tn.NAME} and (res["leaderboard"]["status"] == "ok").all()
    out, _ = predict_frame(build_bundle(res, df, name="t"), df.drop(columns="y").head(5))
    assert len(out) == 5
    best, params, tuned, base = lab.tune_model(*lab.prepare_xy(df, "y")[:2], "regression", tn.NAME, n_iter=2, folds=3)
    assert "regressor__lr" in params
