import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import modellab as lab

FAST = ["Baseline (mean)", "Linear Regression", "Ridge", "Decision Tree"]
FAST_CLF = ["Baseline (majority)", "Logistic Regression", "Decision Tree"]


@pytest.fixture(scope="module")
def reg_df():
    rng = np.random.default_rng(0)
    n = 240
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["x", "y", "z"], n),
                       "when": pd.date_range("2024-01-01", periods=n)})
    df["y"] = 3 * df["a"] - 2 * df["b"] + (df["c"] == "x") * 4 + rng.normal(0, 0.5, n)
    df.loc[::13, "b"] = np.nan
    return df


@pytest.fixture(scope="module")
def clf_df():
    rng = np.random.default_rng(1)
    n = 240
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["x", "y"], n)})
    df["label"] = np.where(df["a"] + df["b"] > 0, "up", "down")
    return df


# ── preparación ──────────────────────────────────────────────────────────────
def test_detect_task():
    assert lab.detect_task(pd.Series(np.arange(100.0))) == "regression"
    assert lab.detect_task(pd.Series(["a", "b"] * 50)) == "classification"
    assert lab.detect_task(pd.Series([True, False] * 50)) == "classification"
    assert lab.detect_task(pd.Series(np.arange(100) % 3)) == "classification"


def test_prepare_xy_drops_useless_columns(reg_df):
    df = reg_df.assign(const=1, ident=np.arange(len(reg_df)), text=[f"id{i}" for i in range(len(reg_df))])
    X, y, info = lab.prepare_xy(df, "y", drop=["c"])
    reasons = {d["column"]: d["reason"] for d in info["dropped"]}
    assert reasons["const"] == "constant"
    assert reasons["ident"] == "unique identifier"
    assert reasons["c"] == "excluded by you"
    assert "too many" in reasons["text"] or reasons["text"] == "unique identifier"
    assert {"when_days", "when_month", "when_weekday"} <= set(X.columns) and "when" not in X.columns
    assert info["task"] == "regression" and len(X) == len(y)


@pytest.mark.parametrize("make,match", [
    (lambda d: (d, "missing"), "not found"),
    (lambda d: (d.assign(t=d["when"]), "t"), "date column"),
    (lambda d: (d.assign(t="same"), "t"), "single class"),
    (lambda d: (d.assign(t=[f"k{i}" for i in range(len(d))]), "t"), "too many"),
    (lambda d: (d.head(10), "y"), "30 rows"),
])
def test_prepare_xy_errors(reg_df, make, match):
    df, target = make(reg_df)
    with pytest.raises(ValueError, match=match):
        lab.prepare_xy(df, target)


def test_rows_without_target_are_ignored(reg_df):
    df = reg_df.copy()
    df.loc[:9, "y"] = np.nan
    X, y, _ = lab.prepare_xy(df, "y")
    assert len(y) == len(df) - 10 and y.notna().all()


def test_preprocessor_handles_missing_and_unseen_categories(reg_df):
    X, y, _ = lab.prepare_xy(reg_df, "y")
    pipe = lab.make_pipeline(X, lab.candidates("regression")["Ridge"][0]).fit(X, y)
    new = X.head(3).copy()
    new["c"] = "never seen"
    new.loc[new.index[0], "a"] = np.nan
    assert np.isfinite(pipe.predict(new)).all()


# ── comparación ──────────────────────────────────────────────────────────────
def test_compare_models_ranks_and_signs(reg_df):
    X, y, _ = lab.prepare_xy(reg_df, "y")
    board = lab.compare_models(X, y, "regression", FAST, folds=3)
    assert board["R2"].is_monotonic_decreasing
    assert board.iloc[-1]["model"] == "Baseline (mean)" and board.iloc[-1]["R2"] < 0.1
    assert board.iloc[0]["R2"] > 0.9 and (board["RMSE"] > 0).all()
    assert (board["status"] == "ok").all() and {"R2 std", "fit time (s)"} <= set(board.columns)


def test_progress_callback_and_unknown_models(reg_df):
    X, y, _ = lab.prepare_xy(reg_df, "y")
    seen = []
    lab.compare_models(X, y, "regression", ["Ridge", "nonsense"], folds=3, progress=lambda f, n: seen.append((f, n)))
    assert seen[0][1] == "Ridge" and seen[-1] == (1.0, "done")
    with pytest.raises(ValueError, match="No valid models"):
        lab.compare_models(X, y, "regression", ["nonsense"], folds=3)


def test_failing_model_does_not_stop_the_others(reg_df, monkeypatch):
    from sklearn.base import BaseEstimator, RegressorMixin

    class Broken(BaseEstimator, RegressorMixin):
        def fit(self, X, y):
            raise RuntimeError("boom")

    real = lab.candidates
    monkeypatch.setattr(lab, "candidates", lambda *a, **k: {**real(*a, **k), "Broken": (Broken(), {})})
    X, y, _ = lab.prepare_xy(reg_df, "y")
    board = lab.compare_models(X, y, "regression", ["Ridge", "Broken"], folds=3)
    assert board.set_index("model").loc["Ridge", "status"] == "ok"
    assert board.set_index("model").loc["Broken", "status"].startswith("failed")
    assert board.iloc[0]["model"] == "Ridge"


def test_classification_comparison_includes_auc_and_handles_multiclass(clf_df):
    X, y, _ = lab.prepare_xy(clf_df, "label")
    board = lab.compare_models(X, y, "classification", FAST_CLF, folds=3)
    assert board.iloc[0]["F1 (weighted)"] > 0.9 and board.iloc[0]["ROC AUC"] > 0.9
    y3 = pd.Series(np.where(X["a"] > 0.5, "hi", np.where(X["a"] < -0.5, "lo", "mid")), index=X.index)
    board3 = lab.compare_models(X, y3, "classification", ["Logistic Regression"], folds=3)
    assert board3.iloc[0]["ROC AUC"] > 0.9


# ── fugas y balance ──────────────────────────────────────────────────────────
def test_leakage_detects_derived_numeric_column(reg_df):
    X, y, _ = lab.prepare_xy(reg_df, "y")
    X = X.assign(leak=y * 2 + 1)
    found = {w["column"] for w in lab.detect_leakage(X, y, "regression")}
    assert "leak" in found


def test_leakage_detects_categorical_copy_of_label(clf_df):
    X, y, _ = lab.prepare_xy(clf_df, "label")
    X = X.assign(outcome_code=y.map({"up": "U", "down": "D"}))
    assert "outcome_code" in {w["column"] for w in lab.detect_leakage(X, y, "classification")}


def test_leakage_ignores_honest_features(reg_df):
    X, y, _ = lab.prepare_xy(reg_df, "y")
    assert lab.detect_leakage(X, y, "regression") == []


def test_class_balance():
    info = lab.class_balance(pd.Series(["a"] * 95 + ["b"] * 5))
    assert info["imbalanced"] and info["majority"] == "a" and info["minority_share"] == pytest.approx(0.05)
    assert not lab.class_balance(pd.Series(["a"] * 50 + ["b"] * 50))["imbalanced"]


# ── ajuste ───────────────────────────────────────────────────────────────────
def test_tune_model_returns_params_and_scores(reg_df):
    X, y, _ = lab.prepare_xy(reg_df, "y")
    pipe, params, tuned, base = lab.tune_model(X, y, "regression", "Ridge", n_iter=4, folds=3)
    assert "alpha" in params and np.isfinite(tuned) and np.isfinite(base)
    assert pipe.set_params(model__alpha=params["alpha"]).fit(X, y).predict(X).shape == (len(X),)


def test_tune_model_errors(reg_df):
    X, y, _ = lab.prepare_xy(reg_df, "y")
    with pytest.raises(ValueError, match="Unknown"):
        lab.tune_model(X, y, "regression", "Nope")
    with pytest.raises(ValueError, match="no hyperparameters"):
        lab.tune_model(X, y, "regression", "Linear Regression")


# ── explicación y gráficos ───────────────────────────────────────────────────
def test_explain_row_blames_the_dominant_feature(reg_df):
    X, y, _ = lab.prepare_xy(reg_df, "y")
    pipe = lab.make_pipeline(X, lab.candidates("regression")["Ridge"][0]).fit(X, y)
    row = X.head(1).copy()
    row["a"] = 5.0
    effects = lab.explain_row(pipe, X, row, "regression")
    assert effects.iloc[0]["feature"] == "a" and effects.iloc[0]["effect"] > 5
    assert set(effects["feature"]) == set(X.columns)


def test_partial_dependence_numeric_and_categorical(reg_df, clf_df):
    X, y, _ = lab.prepare_xy(reg_df, "y")
    pipe = lab.make_pipeline(X, lab.candidates("regression")["Ridge"][0]).fit(X, y)
    assert len(lab.partial_dependence_figure(pipe, X, "a", "regression").data) == 1
    assert len(lab.partial_dependence_figure(pipe, X, "c", "regression").data) == 1
    Xc, yc, _ = lab.prepare_xy(clf_df, "label")
    clf = lab.make_pipeline(Xc, lab.candidates("classification")["Logistic Regression"][0]).fit(Xc, yc)
    assert len(lab.partial_dependence_figure(clf, Xc, "a", "classification").data) == 1


# ── orquestación ─────────────────────────────────────────────────────────────
def test_run_model_lab_regression(reg_df):
    res = lab.run_model_lab(reg_df, "y", models=FAST, folds=3, tune=True, n_iter=3)
    assert res["task_type"] == "regression" and res["best_model"] in ("Linear Regression", "Ridge")
    assert res["metrics"]["R2"] > 0.9 and res["baseline_cv"] < 0.1
    assert {"leaderboard", "importance", "learning_curve", "performance", "residuals", "residual_hist"} <= set(res["plots"])
    assert res["n_train"] + res["n_test"] == len(reg_df)
    assert res["pipeline"].predict(res["X_test"]).shape == (res["n_test"],)
    assert list(res["feature_importance"])[0] in ("a", "c")
    assert "cross-validation" in res["narrative"]


def test_run_model_lab_classification_with_imbalance(clf_df):
    df = pd.concat([clf_df[clf_df["label"] == "down"], clf_df[clf_df["label"] == "up"].head(8)])
    res = lab.run_model_lab(df, "label", models=FAST_CLF, folds=3, tune=False, balance=True)
    assert res["balance"]["imbalanced"] and "rarest class" in res["narrative"]
    assert {"roc", "pr", "calibration", "performance"} <= set(res["plots"])
    assert res["tuning"] is None


def test_run_model_lab_reports_leaks(reg_df):
    res = lab.run_model_lab(reg_df.assign(leak=reg_df["y"] * 3), "y", models=FAST, folds=3, tune=False)
    assert res["leakage"] and "leak" in res["narrative"]


def test_run_model_lab_keeps_default_when_tuning_does_not_help(reg_df, monkeypatch):
    monkeypatch.setattr(lab, "tune_model", lambda X, y, task, name, *a, **k:
                        (lab.make_pipeline(X, lab.candidates(task)[name][0]), {"alpha": 1.0}, 0.1, 0.9))
    res = lab.run_model_lab(reg_df, "y", models=["Baseline (mean)", "Ridge"], folds=3, tune=True)
    assert res["tuning"]["improved"] is False and "did not improve" in res["narrative"]


@pytest.mark.parametrize("kwargs,match", [({"folds": 2}, "folds"), ({"test_size": 0.9}, "test_size")])
def test_run_model_lab_validates_arguments(reg_df, kwargs, match):
    with pytest.raises(ValueError, match=match):
        lab.run_model_lab(reg_df, "y", **kwargs)
