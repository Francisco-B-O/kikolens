"""Laboratorio de modelos: comparación con validación cruzada, ajuste de hiperparámetros, detección de
fugas de datos y diagnósticos. Todo el preprocesado vive dentro del pipeline, así que nunca se filtra
información del conjunto de validación al ajustar imputadores, escaladores o codificadores."""
import time
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import loguniform, randint, spearmanr, uniform
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_recall_curve, r2_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (KFold, RandomizedSearchCV, StratifiedKFold, cross_validate, learning_curve,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from kikolens.analyze.quality import infer_types
from kikolens.utils.dtypes import categorical_columns, expand_datetime_columns

MAX_CLASSES = 50
MAX_CATEGORIES = 30
LOWER_IS_BETTER = {"RMSE", "MAE", "Brier"}
PRIMARY_METRIC = {"regression": "R2", "classification": "F1 (weighted)"}


# ── Preparación ──────────────────────────────────────────────────────────────
def detect_task(y: pd.Series) -> str:
    return "regression" if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y) and y.nunique() > 20 \
        else "classification"


def prepare_xy(df: pd.DataFrame, target: str, drop=(), task=None):
    """Separa X e y. Devuelve (X, y, info) con las columnas descartadas y el motivo."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    df, converted = infer_types(df)
    if pd.api.types.is_datetime64_any_dtype(df[target]):
        raise ValueError("A date column cannot be used as the target. Choose a numeric or categorical column.")

    data = df.dropna(subset=[target]).reset_index(drop=True)
    y = data[target]
    task = task or detect_task(y)
    if task == "classification" and y.nunique() > MAX_CLASSES:
        raise ValueError(f"'{target}' has {y.nunique():,} distinct values, too many to classify. "
                         "Choose a numeric column or one with fewer categories.")
    if task == "classification" and y.nunique() < 2:
        raise ValueError(f"Target '{target}' contains a single class; classification needs at least 2.")
    if len(data) < 30:
        raise ValueError("At least 30 rows with a known target are required.")

    features = data.drop(columns=[target])
    date_cols = list(features.select_dtypes(include=["datetime", "datetimetz"]).columns)
    derived = {f"{c}_{part}" for c in date_cols for part in ("days", "month", "weekday")}
    X = expand_datetime_columns(features)
    dropped = []
    for col in list(X.columns):
        reason = None
        if col in drop:
            reason = "excluded by you"
        elif X[col].nunique(dropna=True) <= 1:
            reason = "constant"
        elif col not in derived and X[col].nunique(dropna=True) == len(X) and not pd.api.types.is_float_dtype(X[col]):
            reason = "unique identifier"
        elif col in categorical_columns(X) and X[col].nunique() > max(MAX_CATEGORIES, 0.5 * len(X)):
            reason = f"too many distinct values ({X[col].nunique():,})"
        if reason:
            dropped.append({"column": col, "reason": reason})
            X = X.drop(columns=[col])
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")
    return X, y, {"task": task, "dropped": dropped, "converted": converted}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cats = categorical_columns(X)
    nums = [c for c in X.columns if c not in cats]
    steps = []
    if nums:
        steps.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), nums))
    if cats:
        steps.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=MAX_CATEGORIES,
                                     sparse_output=False))]), cats))
    return ColumnTransformer(steps, verbose_feature_names_out=False)


# ── Candidatos ───────────────────────────────────────────────────────────────
def _scaled(regressor):
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def candidates(task: str, seed: int = 42, balance: bool = False) -> dict:
    """nombre -> (estimador, distribuciones de hiperparámetros con prefijo `model__`)."""
    cw = "balanced" if balance else None
    if task == "regression":
        return {
            "Baseline (mean)": (DummyRegressor(), {}),
            "Linear Regression": (LinearRegression(), {}),
            "Ridge": (Ridge(), {"model__alpha": loguniform(1e-3, 1e3)}),
            "Lasso": (Lasso(max_iter=5000), {"model__alpha": loguniform(1e-4, 1e1)}),
            "Elastic Net": (ElasticNet(max_iter=5000), {"model__alpha": loguniform(1e-4, 1e1),
                                                        "model__l1_ratio": uniform(0.05, 0.9)}),
            "KNN": (KNeighborsRegressor(), {"model__n_neighbors": randint(2, 30),
                                            "model__weights": ["uniform", "distance"]}),
            "Decision Tree": (DecisionTreeRegressor(random_state=seed),
                              {"model__max_depth": randint(2, 16), "model__min_samples_leaf": randint(1, 20)}),
            "Random Forest": (RandomForestRegressor(n_estimators=150, random_state=seed),
                              {"model__n_estimators": randint(80, 400), "model__max_depth": [None, 6, 10, 16],
                               "model__min_samples_leaf": randint(1, 10), "model__max_features": [1.0, 0.7, 0.5]}),
            "Extra Trees": (ExtraTreesRegressor(n_estimators=150, random_state=seed),
                            {"model__n_estimators": randint(80, 400), "model__max_depth": [None, 8, 16],
                             "model__min_samples_leaf": randint(1, 10)}),
            "Gradient Boosting": (GradientBoostingRegressor(random_state=seed),
                                  {"model__n_estimators": randint(60, 400), "model__learning_rate": loguniform(0.01, 0.3),
                                   "model__max_depth": randint(2, 6), "model__subsample": uniform(0.6, 0.4)}),
            "Hist Gradient Boosting": (HistGradientBoostingRegressor(random_state=seed),
                                       {"model__learning_rate": loguniform(0.01, 0.3), "model__max_iter": randint(60, 400),
                                        "model__max_depth": [None, 3, 5, 8], "model__l2_regularization": loguniform(1e-4, 10)}),
            "Neural Network": (_scaled(MLPRegressor(max_iter=500, early_stopping=True, random_state=seed)),
                               {"model__regressor__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                                "model__regressor__alpha": loguniform(1e-5, 1e-1),
                                "model__regressor__learning_rate_init": loguniform(1e-4, 1e-2)}),
        }
    return {
        "Baseline (majority)": (DummyClassifier(strategy="prior"), {}),
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight=cw), {"model__C": loguniform(1e-3, 1e2)}),
        "Naive Bayes": (GaussianNB(), {"model__var_smoothing": loguniform(1e-11, 1e-6)}),
        "KNN": (KNeighborsClassifier(), {"model__n_neighbors": randint(2, 30), "model__weights": ["uniform", "distance"]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=seed, class_weight=cw),
                          {"model__max_depth": randint(2, 16), "model__min_samples_leaf": randint(1, 20)}),
        "Random Forest": (RandomForestClassifier(n_estimators=150, random_state=seed, class_weight=cw),
                          {"model__n_estimators": randint(80, 400), "model__max_depth": [None, 6, 10, 16],
                           "model__min_samples_leaf": randint(1, 10), "model__max_features": ["sqrt", "log2", 0.5]}),
        "Extra Trees": (ExtraTreesClassifier(n_estimators=150, random_state=seed, class_weight=cw),
                        {"model__n_estimators": randint(80, 400), "model__max_depth": [None, 8, 16],
                         "model__min_samples_leaf": randint(1, 10)}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=seed),
                              {"model__n_estimators": randint(60, 400), "model__learning_rate": loguniform(0.01, 0.3),
                               "model__max_depth": randint(2, 6), "model__subsample": uniform(0.6, 0.4)}),
        "Hist Gradient Boosting": (HistGradientBoostingClassifier(random_state=seed, class_weight=cw),
                                   {"model__learning_rate": loguniform(0.01, 0.3), "model__max_iter": randint(60, 400),
                                    "model__max_depth": [None, 3, 5, 8], "model__l2_regularization": loguniform(1e-4, 10)}),
        "Neural Network": (MLPClassifier(max_iter=500, early_stopping=True, random_state=seed),
                           {"model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                            "model__alpha": loguniform(1e-5, 1e-1), "model__learning_rate_init": loguniform(1e-4, 1e-2)}),
    }


def make_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(X)), ("model", clone(estimator))])


# ── Métricas ─────────────────────────────────────────────────────────────────
def _scorers(task: str, n_classes: int) -> dict:
    if task == "regression":
        return {"R2": "r2", "RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"}
    scorers = {"Accuracy": "accuracy", "Balanced accuracy": "balanced_accuracy", "F1 (weighted)": "f1_weighted"}
    scorers["ROC AUC"] = "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
    return scorers


def _splitter(task: str, y: pd.Series, folds: int, seed: int):
    if task == "classification":
        folds = max(2, min(folds, int(y.value_counts().min())))
        return StratifiedKFold(folds, shuffle=True, random_state=seed)
    return KFold(folds, shuffle=True, random_state=seed)


def holdout_metrics(pipe, X, y, task) -> dict:
    pred = pipe.predict(X)
    if task == "regression":
        return {"R2": float(r2_score(y, pred)), "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
                "MAE": float(mean_absolute_error(y, pred))}
    out = {"Accuracy": float(accuracy_score(y, pred)), "Balanced accuracy": float(balanced_accuracy_score(y, pred)),
           "F1 (weighted)": float(f1_score(y, pred, average="weighted"))}
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        try:
            if proba.shape[1] == 2:
                out["ROC AUC"] = float(roc_auc_score(y, proba[:, 1]))
                out["Brier"] = float(brier_score_loss(y, proba[:, 1], pos_label=pipe.classes_[1]))
            else:
                out["ROC AUC"] = float(roc_auc_score(y, proba, multi_class="ovr", average="weighted", labels=pipe.classes_))
        except ValueError:
            pass
    return out


# ── Fugas de datos ───────────────────────────────────────────────────────────
def detect_leakage(X: pd.DataFrame, y: pd.Series, task: str, seed: int = 42, max_columns: int = 25) -> list:
    """Avisa de columnas que predicen el objetivo de forma sospechosamente perfecta."""
    warnings_out = []
    sample = X.sample(min(len(X), 2000), random_state=seed)
    ys = y.loc[sample.index]
    target_codes = ys.astype("category").cat.codes if task == "classification" else ys

    ranked = []
    for col in X.columns:
        s = sample[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.nunique() > 1:
            rho = spearmanr(s, target_codes, nan_policy="omit").statistic
            if np.isfinite(rho) and abs(rho) >= 0.98:
                warnings_out.append({"column": col, "severity": "high", "message":
                                     f"'{col}' has a rank correlation of {rho:.3f} with the target. It may be derived from it."})
        ranked.append(col)

    flagged = {w["column"] for w in warnings_out}
    for col in ranked[:max_columns]:
        if col in flagged:
            continue
        pipe = make_pipeline(sample[[col]], DecisionTreeRegressor(max_depth=4, random_state=seed)
                             if task == "regression" else DecisionTreeClassifier(max_depth=4, random_state=seed))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = cross_validate(pipe, sample[[col]], ys, cv=3,
                                       scoring="r2" if task == "regression" else "accuracy")["test_score"].mean()
        except Exception:
            continue
        baseline = 0.0 if task == "regression" else float(ys.value_counts(normalize=True).iloc[0])
        if score >= 0.98 and score - baseline > 0.05:
            warnings_out.append({"column": col, "severity": "high", "message":
                                 f"'{col}' alone predicts the target almost perfectly (score {score:.3f}). "
                                 "Check whether it is only known after the outcome."})
    return warnings_out


def class_balance(y: pd.Series) -> dict:
    shares = y.value_counts(normalize=True)
    return {"classes": int(len(shares)), "majority": str(shares.index[0]), "majority_share": float(shares.iloc[0]),
            "minority": str(shares.index[-1]), "minority_share": float(shares.iloc[-1]),
            "imbalanced": bool(shares.iloc[-1] < 0.10)}


# ── Comparación y ajuste ─────────────────────────────────────────────────────
def compare_models(X, y, task, models=None, folds=5, seed=42, balance=False, progress=None) -> pd.DataFrame:
    """Validación cruzada de cada candidato. Un modelo que falla no detiene a los demás."""
    pool = candidates(task, seed, balance)
    names = [m for m in (models or pool) if m in pool]
    if not names:
        raise ValueError("No valid models selected.")
    cv = _splitter(task, y, folds, seed)
    scorers = _scorers(task, y.nunique() if task == "classification" else 0)

    rows = []
    for i, name in enumerate(names):
        if progress:
            progress(i / len(names), name)
        estimator, _ = pool[name]
        started = time.perf_counter()
        row = {"model": name}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = cross_validate(make_pipeline(X, estimator), X, y, cv=cv, scoring=scorers, error_score=np.nan)
            for metric in scorers:
                values = res[f"test_{metric}"]
                sign = -1 if metric in LOWER_IS_BETTER else 1
                row[metric] = float(np.nanmean(values) * sign) if not np.all(np.isnan(values)) else np.nan
                row[f"{metric} std"] = float(np.nanstd(values)) if not np.all(np.isnan(values)) else np.nan
            row["fit time (s)"] = float(np.mean(res["fit_time"]))
            row["status"] = "ok" if not np.isnan(row[PRIMARY_METRIC[task]]) else "failed"
        except Exception as e:  # pragma: no cover - defensivo
            row["status"] = f"failed: {e}"
        row["total time (s)"] = time.perf_counter() - started
        rows.append(row)
    if progress:
        progress(1.0, "done")

    board = pd.DataFrame(rows)
    primary = PRIMARY_METRIC[task]
    if primary not in board:
        raise ValueError("No model could be evaluated.")
    return board.sort_values(primary, ascending=False, na_position="last").reset_index(drop=True)


def tune_model(X, y, task, name, n_iter=20, folds=5, seed=42, balance=False, scoring=None):
    """Búsqueda aleatoria de hiperparámetros. Devuelve (mejor pipeline sin ajustar, parámetros, puntuación CV, puntuación base)."""
    pool = candidates(task, seed, balance)
    if name not in pool:
        raise ValueError(f"Unknown model '{name}'.")
    estimator, space = pool[name]
    if not space:
        raise ValueError(f"'{name}' has no hyperparameters to tune.")
    scoring = scoring or ("r2" if task == "regression" else "f1_weighted")
    cv = _splitter(task, y, folds, seed)
    base_score = float(np.mean(cross_validate(make_pipeline(X, estimator), X, y, cv=cv, scoring=scoring)["test_score"]))
    search = RandomizedSearchCV(make_pipeline(X, estimator), space, n_iter=n_iter, cv=cv, scoring=scoring,
                                random_state=seed, error_score=np.nan, refit=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    params = search.best_params_
    best = make_pipeline(X, estimator).set_params(**params)
    return best, {k.removeprefix("model__"): v for k, v in params.items()}, float(search.best_score_), base_score


# ── Explicación ──────────────────────────────────────────────────────────────
def feature_importance(pipe, X, y, task, seed=42, repeats=5) -> dict:
    scoring = "r2" if task == "regression" else "f1_weighted"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm = permutation_importance(pipe, X, y, n_repeats=repeats, random_state=seed, scoring=scoring)
    return dict(sorted(zip(X.columns, perm.importances_mean.tolist()), key=lambda kv: kv[1], reverse=True))


def explain_row(pipe, X_reference: pd.DataFrame, row: pd.DataFrame, task: str) -> pd.DataFrame:
    """Oclusión: cuánto cambia la predicción al sustituir cada variable por un valor típico (mediana o moda)."""
    def score(frame):
        if task == "classification" and hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(frame)
            return float(probs[0].max()) if probs.shape[1] > 2 else float(probs[0, -1])
        return float(pipe.predict(frame)[0])

    base = score(row)
    out = []
    for col in X_reference.columns:
        neutral = row.copy()
        s = X_reference[col]
        typical = s.median() if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) else s.mode().iloc[0]
        neutral[col] = typical
        out.append({"feature": col, "value": row[col].iloc[0], "typical": typical, "effect": base - score(neutral)})
    return pd.DataFrame(out).sort_values("effect", key=abs, ascending=False).reset_index(drop=True)


# ── Gráficos ─────────────────────────────────────────────────────────────────
def leaderboard_figure(board: pd.DataFrame, task: str) -> go.Figure:
    metric = PRIMARY_METRIC[task]
    ok = board[board["status"] == "ok"].iloc[::-1]
    fig = go.Figure(go.Bar(x=ok[metric], y=ok["model"], orientation="h",
                           error_x=dict(type="data", array=ok[f"{metric} std"]), name=metric))
    fig.update_layout(title=f"Cross-validated {metric} (bars show the spread across folds)")
    return fig


def diagnostics(pipe, X_test, y_test, task) -> dict:
    figs = {}
    pred = pipe.predict(X_test)
    if task == "regression":
        lo, hi = float(min(y_test.min(), pred.min())), float(max(y_test.max(), pred.max()))
        f = go.Figure(go.Scatter(x=y_test, y=pred, mode="markers", name="Predictions"))
        f.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect fit", line=dict(dash="dash")))
        f.update_layout(title="Predicted vs actual", xaxis_title="Actual", yaxis_title="Predicted")
        figs["performance"] = f
        resid = np.asarray(y_test) - pred
        r = go.Figure(go.Scatter(x=pred, y=resid, mode="markers", name="Residuals"))
        r.add_hline(y=0, line_dash="dash")
        r.update_layout(title="Residuals vs predicted", xaxis_title="Predicted", yaxis_title="Residual")
        figs["residuals"] = r
        h = go.Figure(go.Histogram(x=resid, nbinsx=30, name="Residuals"))
        h.update_layout(title="Residual distribution", xaxis_title="Residual")
        figs["residual_hist"] = h
        return figs

    cm = confusion_matrix(y_test, pred, labels=pipe.classes_)
    f = go.Figure(go.Heatmap(z=cm, x=[str(c) for c in pipe.classes_], y=[str(c) for c in pipe.classes_],
                             colorscale="Greens", text=cm, texttemplate="%{text}"))
    f.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual")
    f.update_yaxes(autorange="reversed")
    figs["performance"] = f
    if not hasattr(pipe, "predict_proba"):
        return figs

    proba = pipe.predict_proba(X_test)
    classes = list(pipe.classes_)
    roc = go.Figure()
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_test, proba[:, 1], pos_label=classes[1])
        roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC {roc_auc_score(y_test, proba[:, 1]):.3f}"))
        prec, rec, _ = precision_recall_curve(y_test, proba[:, 1], pos_label=classes[1])
        pr = go.Figure(go.Scatter(x=rec, y=prec, mode="lines",
                                  name=f"AP {average_precision_score(y_test, proba[:, 1], pos_label=classes[1]):.3f}"))
        pr.update_layout(title="Precision-recall", xaxis_title="Recall", yaxis_title="Precision")
        figs["pr"] = pr
        frac, mean_pred = calibration_curve((np.asarray(y_test) == classes[1]).astype(int), proba[:, 1], n_bins=8,
                                            strategy="quantile")
        cal = go.Figure(go.Scatter(x=mean_pred, y=frac, mode="lines+markers", name="Model"))
        cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated", line=dict(dash="dash")))
        cal.update_layout(title="Calibration", xaxis_title="Predicted probability", yaxis_title="Observed frequency")
        figs["calibration"] = cal
    else:
        for i, c in enumerate(classes):
            truth = (np.asarray(y_test) == c).astype(int)
            if truth.sum() and truth.sum() < len(truth):
                fpr, tpr, _ = roc_curve(truth, proba[:, i])
                roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{c} (AUC {roc_auc_score(truth, proba[:, i]):.2f})"))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dot")))
    roc.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    figs["roc"] = roc
    return figs


def learning_curve_figure(pipe, X, y, task, seed=42) -> go.Figure:
    if len(X) > 4000:
        idx = X.sample(4000, random_state=seed).index
        X, y = X.loc[idx], y.loc[idx]
    cv = _splitter(task, y, 3, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sizes, train, val = learning_curve(pipe, X, y, cv=cv, train_sizes=np.linspace(0.2, 1.0, 5),
                                           scoring="r2" if task == "regression" else "f1_weighted", error_score=np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=train.mean(axis=1), mode="lines+markers", name="Training"))
    fig.add_trace(go.Scatter(x=sizes, y=val.mean(axis=1), mode="lines+markers", name="Validation"))
    fig.update_layout(title="Learning curve (a gap means overfitting; both low means underfitting)",
                      xaxis_title="Training rows", yaxis_title="R2" if task == "regression" else "F1 (weighted)")
    return fig


def partial_dependence_figure(pipe, X: pd.DataFrame, feature: str, task: str) -> go.Figure:
    kind_kwargs = {"response_method": "predict_proba"} if task == "classification" and hasattr(pipe, "predict_proba") else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pd_res = partial_dependence(pipe, X, [feature], grid_resolution=30, **kind_kwargs)
    values = pd_res["grid_values"][0]
    avg = np.asarray(pd_res["average"])
    avg = avg[-1] if avg.ndim > 1 and task == "classification" else avg[0]
    categorical = not pd.api.types.is_numeric_dtype(X[feature])
    fig = go.Figure(go.Bar(x=[str(v) for v in values], y=avg) if categorical else go.Scatter(x=values, y=avg, mode="lines"))
    fig.update_layout(title=f"Average prediction as '{feature}' changes", xaxis_title=feature, yaxis_title="Prediction")
    return fig


# ── Orquestación ─────────────────────────────────────────────────────────────
def run_model_lab(df, target, models=None, folds=5, tune=True, n_iter=20, test_size=0.2, balance=False,
                  drop=(), seed=42, progress=None, task=None) -> dict:
    """Pipeline completo: separa un test final, compara por CV, ajusta el mejor, evalúa y explica."""
    if not 3 <= folds <= 10:
        raise ValueError("folds must be between 3 and 10")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4")

    X, y, info = prepare_xy(df, target, drop=drop, task=task)
    task = info["task"]
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)

    notes = detect_leakage(X_train, y_train, task, seed)
    balance_info = class_balance(y) if task == "classification" else None
    board = compare_models(X_train, y_train, task, models, folds, seed, balance, progress)
    ranked = board[(board["status"] == "ok") & ~board["model"].str.startswith("Baseline")]
    if ranked.empty:
        raise ValueError("No model could be evaluated.")
    best_name = ranked.iloc[0]["model"]

    pool = candidates(task, seed, balance)
    tuning = None
    unfitted = make_pipeline(X_train, pool[best_name][0])
    if tune and pool[best_name][1]:
        tuned, params, tuned_score, base_score = tune_model(X_train, y_train, task, best_name, n_iter, folds, seed, balance)
        tuning = {"params": params, "tuned_cv": tuned_score, "default_cv": base_score,
                  "improved": tuned_score > base_score + 1e-9}
        if tuning["improved"]:  # si el ajuste no mejora se conserva la configuración por defecto
            unfitted = tuned
    pipe = clone(unfitted).fit(X_train, y_train)

    metrics = holdout_metrics(pipe, X_test, y_test, task)
    baseline_name = next((m for m in board["model"] if m.startswith("Baseline")), None)
    importance = feature_importance(pipe, X_test, y_test, task, seed)
    top = list(importance.items())[:10][::-1]
    imp_fig = go.Figure(go.Bar(x=[v for _, v in top], y=[k for k, _ in top], orientation="h"))
    imp_fig.update_layout(title="Permutation importance on the test set (top 10)")

    plots = {"leaderboard": leaderboard_figure(board, task), "importance": imp_fig,
             "learning_curve": learning_curve_figure(clone(unfitted), X_train, y_train, task, seed)}
    plots.update(diagnostics(pipe, X_test, y_test, task))

    primary = PRIMARY_METRIC[task]
    return {
        "task_type": task, "target": target, "leaderboard": board, "best_model": best_name, "pipeline": pipe,
        "model": pipe, "tuning": tuning, "metrics": metrics, "primary_metric": primary,
        "baseline_cv": float(board.loc[board["model"] == baseline_name, primary].iloc[0]) if baseline_name else None,
        "feature_importance": importance, "leakage": notes, "dropped": info["dropped"], "balance": balance_info,
        "converted": info["converted"], "plots": plots, "folds": folds, "n_train": len(X_train), "n_test": len(X_test),
        "features": list(X.columns), "X_train": X_train, "X_test": X_test, "y_test": y_test,
        "narrative": _narrative(task, target, best_name, metrics, primary, board, tuning, notes, balance_info, len(X_train), folds)
        + _converted_note(info["converted"]),
    }


def _converted_note(converted) -> str:
    if not converted:
        return ""
    names = {"numeric": "numbers", "datetime": "dates"}
    return "\n\nRead as " + ", ".join(f"{names[kind]}: {col}" for col, kind in converted) + " (they were stored as text)."


def _narrative(task, target, best, metrics, primary, board, tuning, leakage, balance, n_train, folds) -> str:
    cv = board.loc[board["model"] == best, primary].iloc[0]
    text = (f"### {best}\nBest of {int((board['status'] == 'ok').sum())} models by {folds}-fold cross-validation on "
            f"{n_train:,} training rows ({primary} {cv:.3f}). On the untouched test set: "
            + ", ".join(f"{k} {v:.3f}" for k, v in metrics.items()) + ".")
    if tuning:
        text += (f"\n\nTuning {'improved' if tuning['improved'] else 'did not improve'} it "
                 f"({tuning['default_cv']:.3f} to {tuning['tuned_cv']:.3f} cross-validated).")
    if leakage:
        text += f"\n\n**{len(leakage)} possible data leak(s) found.** Results may be too good to be true."
    if balance and balance["imbalanced"]:
        text += f"\n\nThe rarest class is only {balance['minority_share']:.1%} of the rows; prefer balanced accuracy or F1 over accuracy."
    return text
