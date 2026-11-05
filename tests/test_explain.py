import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import explain as ex
from kikolens.analyze.modellab import run_model_lab


@pytest.fixture(scope="module")
def reg():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n),
                       "g": rng.choice(["p", "q", "r"], n)})
    noise = np.where(df["g"] == "r", 3.0, 0.2)          # el grupo r es mucho más ruidoso: ahí el modelo fallará más
    df["y"] = 3 * df["a"] - 2 * df["b"] + rng.normal(size=n) * noise
    return run_model_lab(df, "y", models=["Linear Regression", "Random Forest", "Ridge"], folds=3, tune=False)


@pytest.fixture(scope="module")
def clf():
    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    df["label"] = np.where(df["a"] + 0.5 * df["b"] + rng.normal(size=n) * 0.7 > 0, "yes", "no")
    return run_model_lab(df, "label", models=["Logistic Regression", "Random Forest", "Naive Bayes"], folds=3, tune=False)


def test_shapley_adds_up_and_ranks_the_real_drivers(reg):
    row = reg["X_test"].iloc[[3]]
    out = ex.shapley_row(reg["pipeline"], reg["X_train"], row, "regression", n_permutations=60)
    t = out["table"].set_index("feature")["shapley"]
    assert abs(t.sum() - (out["prediction"] - out["base"])) < 1e-6          # propiedad de eficiencia
    assert out["prediction"] == pytest.approx(float(reg["pipeline"].predict(row)[0]), abs=1e-6)
    assert set(out["table"]["feature"].head(2)) == {"a", "b"} and abs(t["c"]) < 0.3
    assert abs(t["a"]) > abs(t["c"]) * 4


def test_shapley_for_classification_uses_probabilities(clf):
    row = clf["X_test"].iloc[[0]]
    out = ex.shapley_row(clf["pipeline"], clf["X_train"], row, "classification", n_permutations=30, target_class="yes")
    assert 0 <= out["base"] <= 1 and 0 <= out["prediction"] <= 1
    assert out["table"].iloc[0]["feature"] in ("a", "b")
    assert len(ex.shapley_figure(out).data) == 1


def test_shapley_validation(reg):
    with pytest.raises(ValueError, match="exactly one"):
        ex.shapley_row(reg["pipeline"], reg["X_train"], reg["X_test"].head(2), "regression")
    wide = pd.DataFrame(np.zeros((5, 41)), columns=[f"f{i}" for i in range(41)])
    with pytest.raises(ValueError, match="limited"):
        ex.shapley_row(None, wide, wide.head(1), "regression")


def test_global_shapley_matches_known_importance(reg):
    out = ex.shapley_global(reg["pipeline"], reg["X_train"], reg["X_test"], "regression", n_rows=40, n_permutations=20)
    assert out.iloc[0]["feature"] == "a" and set(out["feature"].head(2)) == {"a", "b"}


def test_error_analysis_finds_the_noisy_group(reg):
    out = ex.error_analysis(reg["pipeline"], reg["X_test"], reg["y_test"], "regression", min_rows=5)
    assert not out["segments"].empty
    top = out["segments"].iloc[0]
    assert top["feature"] == "g" and top["segment"] == "r" and top["times worse"] > 2
    assert out["worst"]["error"].is_monotonic_decreasing and "times the average" in out["summary"]
    with pytest.raises(ValueError, match="At least"):
        ex.error_analysis(reg["pipeline"], reg["X_test"].head(5), reg["y_test"].head(5), "regression")


def test_error_analysis_on_classification(clf):
    out = ex.error_analysis(clf["pipeline"], clf["X_test"], clf["y_test"], "classification", min_rows=5)
    assert out["metric"] == "error rate" and 0 <= out["overall"] <= 1


def test_threshold_analysis_moves_with_costs(clf):
    cheap_fn = ex.threshold_analysis(clf["pipeline"], clf["X_test"], clf["y_test"], cost_fp=1, cost_fn=1, positive="yes")
    costly_fn = ex.threshold_analysis(clf["pipeline"], clf["X_test"], clf["y_test"], cost_fp=1, cost_fn=10, positive="yes")
    assert costly_fn["best_cost"]["threshold"] <= cheap_fn["best_cost"]["threshold"]   # fallar un positivo cuesta más: umbral más bajo
    t = cheap_fn["table"]
    assert t["recall"].dropna().is_monotonic_decreasing and (t["false positives"].diff().dropna() <= 0).all()
    assert len(ex.threshold_figure(cheap_fn).data) == 4 and "cheapest" in cheap_fn["summary"]


def test_threshold_validation(reg, clf):
    with pytest.raises(ValueError, match="binary"):
        ex.threshold_analysis(reg["pipeline"], reg["X_test"], reg["y_test"])
    with pytest.raises(ValueError, match="not one of"):
        ex.threshold_analysis(clf["pipeline"], clf["X_test"], clf["y_test"], positive="maybe")
    with pytest.raises(ValueError, match="negative"):
        ex.threshold_analysis(clf["pipeline"], clf["X_test"], clf["y_test"], cost_fp=-1)


def test_compare_pair_runs(reg, clf):
    X_train, X_test = reg["X_train"], reg["X_test"]
    y_train = pd.Series(reg["pipeline"].predict(X_train), index=X_train.index)   # objetivo sintético coherente
    out = ex.compare_pair(X_train, y_train, X_test, reg["y_test"], "regression", "Linear Regression", "Random Forest")
    assert set(out["scores"]) == {"Linear Regression", "Random Forest"} and out["metric"] == "R2"
    assert sum(out["wins"].values()) <= len(X_test) and "gap" in out["disagreements"].columns
    assert -1 <= out["agreement"] <= 1 and out["verdict"]
    yc = pd.Series(clf["pipeline"].predict(clf["X_train"]), index=clf["X_train"].index)
    cls = ex.compare_pair(clf["X_train"], yc, clf["X_test"], clf["y_test"], "classification", "Logistic Regression", "Naive Bayes")
    assert cls["metric"].startswith("F1") and 0 <= cls["agreement"] <= 1
    with pytest.raises(ValueError, match="different"):
        ex.compare_pair(X_train, y_train, X_test, reg["y_test"], "regression", "Ridge", "Ridge")
    with pytest.raises(ValueError, match="Unknown model"):
        ex.compare_pair(X_train, y_train, X_test, reg["y_test"], "regression", "Ridge", "Magic")
