"""Redes neuronales (perceptrón multicapa) para regresión y clasificación tabular."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from kikolens.analyze.automl import AutoMLExplainer

ACTIVATIONS = ("relu", "tanh", "logistic", "identity")
SOLVERS = ("adam", "sgd", "lbfgs")
PRESETS = {
    "Small (32)": (32,),
    "Medium (64, 32)": (64, 32),
    "Deep (128, 64, 32)": (128, 64, 32),
    "Wide (256, 128)": (256, 128),
}


def parse_hidden_layers(text) -> tuple:
    """Convierte '64, 32' (o una secuencia) en una tupla de enteros positivos."""
    if isinstance(text, (tuple, list)):
        layers = tuple(int(n) for n in text)
    else:
        try:
            layers = tuple(int(p) for p in str(text).replace(";", ",").split(",") if p.strip())
        except ValueError:
            raise ValueError(f"Invalid layer sizes: {text!r}. Use integers like '64, 32'.")
    if not layers or any(n <= 0 for n in layers):
        raise ValueError("Hidden layers must be a non-empty list of positive integers.")
    return layers


def count_parameters(model) -> int:
    return int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))


def architecture_figure(layer_sizes, max_nodes=12) -> go.Figure:
    """Dibuja la red: una columna de nodos por capa, con las conexiones entre capas vecinas."""
    labels = ["Input"] + [f"Hidden {i + 1}" for i in range(len(layer_sizes) - 2)] + ["Output"]
    shown = [min(n, max_nodes) for n in layer_sizes]
    coords = []
    for x, n in enumerate(shown):
        ys = np.linspace(-1, 1, n) * (n / max(shown))
        coords.append([(x, y) for y in ys])

    fig = go.Figure()
    edge_x, edge_y = [], []
    for left, right in zip(coords[:-1], coords[1:]):
        for (x0, y0) in left:
            for (x1, y1) in right:
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="skip",
                             line=dict(width=0.5, color="rgba(128,128,128,0.25)"), showlegend=False))
    for pts, size, label, n_shown in zip(coords, layer_sizes, labels, shown):
        suffix = f" ({size}, showing {n_shown})" if size > n_shown else f" ({size})"
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers",
                                 marker=dict(size=14), name=label + suffix, hoverinfo="name"))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), title="Network architecture")
    return fig


def _build_model(task, hidden_layers, activation, solver, learning_rate, alpha, max_iter, early_stopping, seed):
    kwargs = dict(
        hidden_layer_sizes=hidden_layers, activation=activation, solver=solver, alpha=alpha,
        max_iter=max_iter, random_state=seed,
        early_stopping=early_stopping and solver != "lbfgs",
    )
    if solver != "lbfgs":
        kwargs["learning_rate_init"] = learning_rate
    if task == "regression":
        # El objetivo se estandariza por dentro; predict() devuelve la escala original.
        return TransformedTargetRegressor(regressor=MLPRegressor(**kwargs), transformer=StandardScaler())
    return MLPClassifier(**kwargs)


def _training_figure(model, task) -> go.Figure:
    fig = go.Figure()
    curve = getattr(model, "loss_curve_", None)
    if curve:
        fig.add_trace(go.Scatter(y=curve, mode="lines", name="Training loss"))
    val = getattr(model, "validation_scores_", None)
    if val is not None:
        label = "Validation R2" if task == "regression" else "Validation accuracy"
        fig.add_trace(go.Scatter(y=val, mode="lines", name=label, yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
    fig.update_layout(title="Training progress", xaxis_title="Epoch", yaxis_title="Loss")
    return fig


def run_neural_network(
    df: pd.DataFrame,
    target_col: str,
    hidden_layers=(64, 32),
    activation: str = "relu",
    solver: str = "adam",
    learning_rate: float = 0.001,
    alpha: float = 0.0001,
    max_iter: int = 300,
    early_stopping: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Entrena un MLP con el mismo preprocesado que AutoML y devuelve un resultado compatible."""
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {ACTIVATIONS}")
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}")
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    hidden_layers = parse_hidden_layers(hidden_layers)

    explainer = AutoMLExplainer(df, target_col)
    explainer.analyze_task()
    task = explainer.task_type
    X, y, imputer, scaler = explainer.preprocess()
    if len(X) < 20:
        raise ValueError("At least 20 rows are required to train a neural network.")
    if task == "classification" and len(np.unique(y)) < 2:
        raise ValueError(f"Target '{target_col}' contains a single class; classification needs at least 2.")

    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    model = _build_model(task, hidden_layers, activation, solver, learning_rate, alpha,
                         max_iter, early_stopping, random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    net = model.regressor_ if task == "regression" else model

    if task == "regression":
        metrics = {
            "R2": float(r2_score(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAE": float(mean_absolute_error(y_test, y_pred)),
        }
        perf = go.Figure(go.Scatter(x=y_test, y=y_pred, mode="markers", name="Predictions"))
        lo, hi = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
        perf.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect fit",
                                  line=dict(dash="dash")))
        perf.update_layout(title="Prediction accuracy", xaxis_title="Actual", yaxis_title="Predicted")
    else:
        metrics = {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "F1 (weighted)": float(f1_score(y_test, y_pred, average="weighted")),
        }
        cm = confusion_matrix(y_test, y_pred)
        perf = go.Figure(go.Heatmap(z=cm, colorscale="Greens", text=cm, texttemplate="%{text}"))
        perf.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual")

    perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=random_state)
    importance = dict(sorted(zip(explainer.training_columns, perm.importances_mean.tolist()),
                             key=lambda kv: kv[1], reverse=True))
    top = list(importance.items())[:10][::-1]
    imp_fig = go.Figure(go.Bar(x=[v for _, v in top], y=[k for k, _ in top], orientation="h"))
    imp_fig.update_layout(title="Permutation importance (top 10)")

    n_out = 1 if task == "regression" or len(net.classes_) <= 2 else len(net.classes_)
    sizes = [X.shape[1], *hidden_layers, n_out]
    n_params = count_parameters(net)
    architecture = " > ".join(str(s) for s in sizes)

    narrative = (
        f"### Neural network (MLP): {architecture}\n"
        f"**Task:** {task} on `{target_col}` · **Activation:** {activation} · **Solver:** {solver} · "
        f"**Parameters:** {n_params:,}\n\n"
        f"Trained for {net.n_iter_} epochs"
        f"{' (early stopping on a 10% validation split)' if net.early_stopping else ''}. "
        f"Inputs were median-imputed, one-hot encoded and standardized."
    )

    return {
        "metrics": metrics, "narrative": narrative, "model": model, "imputer": imputer, "scaler": scaler,
        "columns": explainer.training_columns, "categorical_mappings": explainer.categorical_mappings,
        "task_type": task, "feature_importance": importance, "target": target_col,
        "plots": {"performance": perf, "importance": imp_fig,
                  "training": _training_figure(net, task), "architecture": architecture_figure(sizes)},
        "n_parameters": n_params, "n_iter": int(net.n_iter_), "architecture": sizes,
        "loss_curve": list(getattr(net, "loss_curve_", []) or []),
    }
