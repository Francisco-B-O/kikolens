"""Selección automática de variables. Puntúa cada columna con cuatro métodos distintos (información mutua, un modelo lineal con
penalización L1, la importancia por permutación de un modelo de árboles y la importancia de un bosque aleatorio), las ordena por
consenso y mide con validación cruzada cuánto rinde un modelo con las k mejores. Recomienda el menor k que no empeora de forma
apreciable el mejor resultado (la regla del error estándar)."""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from kikolens.analyze.modellab import _splitter, build_preprocessor, make_pipeline, prepare_xy

METHODS = ("mutual information", "L1 model", "permutation", "random forest")


def _encoded(X: pd.DataFrame) -> pd.DataFrame:
    """Columnas numéricas tal cual y categóricas como códigos, rellenando vacíos: solo para puntuar, no para modelar."""
    out = X.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].fillna(out[col].median())
        else:
            out[col] = pd.factorize(out[col].astype(str))[0]
    return out.astype(float)


def _by_column(names, values, columns) -> pd.Series:
    """Suma la importancia de las columnas derivadas (indicadores de categoría) en su columna original."""
    out = pd.Series(0.0, index=columns)
    ordered = sorted(columns, key=len, reverse=True)
    for name, value in zip(names, values):
        owner = next((c for c in ordered if name == c or name.startswith(f"{c}_")), None)
        if owner is not None:
            out[owner] += abs(float(value))
    return out


def score_columns(X: pd.DataFrame, y: pd.Series, task: str, seed: int = 42) -> pd.DataFrame:
    """Importancia de cada columna según cada método (mayor es más importante)."""
    columns = list(X.columns)
    enc = _encoded(X)
    discrete = [not pd.api.types.is_numeric_dtype(X[c]) for c in columns]
    mi = (mutual_info_classif if task == "classification" else mutual_info_regression)(enc, y, discrete_features=discrete, random_state=seed)
    out = pd.DataFrame(index=columns)
    out["mutual information"] = mi

    prep = build_preprocessor(X).fit(X)
    Z = prep.transform(X)
    names = list(prep.get_feature_names_out())
    Z = StandardScaler().fit_transform(Z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == "regression":
            coef = np.abs(LassoCV(cv=3, random_state=seed, max_iter=3000).fit(Z, y).coef_)
        else:
            coef = np.abs(LogisticRegression(penalty="l1", solver="saga", C=0.3, max_iter=2000, random_state=seed).fit(Z, y).coef_).sum(axis=0)
    out["L1 model"] = _by_column(names, coef, columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed,
                                                        stratify=y if task == "classification" and y.value_counts().min() >= 2 else None)
    model = make_pipeline(X_train, HistGradientBoostingRegressor(max_iter=80, random_state=seed) if task == "regression"
                          else HistGradientBoostingClassifier(max_iter=80, random_state=seed)).fit(X_train, y_train)
    perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=seed)
    out["permutation"] = np.clip(perm.importances_mean, 0, None)

    forest = make_pipeline(X_train, RandomForestRegressor(n_estimators=120, random_state=seed, n_jobs=1) if task == "regression"
                           else RandomForestClassifier(n_estimators=120, random_state=seed, n_jobs=1)).fit(X_train, y_train)
    names_rf = list(forest.named_steps["prep"].get_feature_names_out())
    out["random forest"] = _by_column(names_rf, forest.named_steps["model"].feature_importances_, columns)
    return out


def select_features(df: pd.DataFrame, target: str, drop=(), folds: int = 3, seed: int = 42, max_columns: int = 60, progress=None) -> dict:
    """Ranking por consenso y curva de rendimiento. Devuelve la tabla, el conjunto recomendado y las columnas que se pueden quitar."""
    X, y, info = prepare_xy(df, target, drop=drop)
    task = info["task"]
    if X.shape[1] < 2:
        raise ValueError("At least two usable feature columns are needed to choose among them.")
    if X.shape[1] > max_columns:
        raise ValueError(f"There are {X.shape[1]} feature columns; choose at most {max_columns} (leave some out first).")
    if len(X) > 5000:
        keep = X.sample(5000, random_state=seed).index
        X, y = X.loc[keep], y.loc[keep]
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    scores = score_columns(X, y, task, seed)
    ranks = scores.rank(ascending=False, method="average")
    table = scores.copy()
    table["average rank"] = ranks.mean(axis=1)
    table = table.sort_values("average rank").reset_index().rename(columns={"index": "column"})
    table["consensus"] = range(1, len(table) + 1)
    order = table["column"].tolist()

    cv = _splitter(task, y, folds, seed)
    scoring = "r2" if task == "regression" else "f1_weighted"
    curve = []
    for k in range(1, len(order) + 1):
        if progress:
            progress(k / len(order), f"{k} column(s)")
        subset = order[:k]
        pipe = make_pipeline(X[subset], HistGradientBoostingRegressor(max_iter=60, random_state=seed) if task == "regression"
                             else HistGradientBoostingClassifier(max_iter=60, random_state=seed))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = cross_val_score(pipe, X[subset], y, cv=cv, scoring=scoring, error_score=np.nan)
        curve.append({"columns": k, "score": float(np.nanmean(values)), "std": float(np.nanstd(values)), "added": order[k - 1]})
    curve = pd.DataFrame(curve)
    best = curve["score"].idxmax()
    limit = curve.loc[best, "score"] - curve.loc[best, "std"] / np.sqrt(max(folds, 1))
    chosen = int(curve[curve["score"] >= limit]["columns"].min())
    recommended, removable = order[:chosen], order[chosen:]
    metric = "R2" if task == "regression" else "F1"
    return {"table": table, "curve": curve, "recommended": recommended, "removable": removable, "task": task, "metric": metric,
            "best_score": float(curve["score"].max()), "score_recommended": float(curve.loc[chosen - 1, "score"]),
            "figure": curve_figure(curve, chosen, metric),
            "summary": (f"{len(recommended)} of {len(order)} columns are enough: {metric} {curve.loc[chosen - 1, 'score']:.3f} against "
                        f"{curve['score'].max():.3f} with the best subset. " +
                        (f"You can leave out {', '.join(removable[:6])}{' and others' if len(removable) > 6 else ''}." if removable else "Every column helps."))}


def curve_figure(curve: pd.DataFrame, chosen: int, metric: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=curve["columns"], y=curve["score"], mode="lines+markers", name=metric,
                               error_y=dict(type="data", array=curve["std"], visible=True), text=curve["added"],
                               hovertemplate="%{x} columns (last added: %{text})<br>%{y:.3f}<extra></extra>"))
    fig.add_vline(x=chosen, line_dash="dash", annotation_text=f"Recommended: {chosen}")
    fig.update_layout(title=f"Cross-validated {metric} using the best k columns", xaxis_title="Columns used", yaxis_title=metric)
    return fig
