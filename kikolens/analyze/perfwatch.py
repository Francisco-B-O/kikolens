"""¿Sigue siendo bueno el modelo? Cuando ya se conoce el resultado real de lo que el modelo predijo, compara su acierto por
periodo con el que tuvo al probarlo y avisa si se deteriora. Complementa a la monitorización de datos: la deriva de las
entradas es una pista, el error real es la prueba."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

from kikolens.analyze.predict import predict_frame

PERIODS = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q"}
ERROR_FACTOR = 1.5       # un error medio 1,5 veces mayor que el de la prueba es un aviso
ACCURACY_DROP = 0.10


def _metrics(task: str, actual: pd.Series, predicted: pd.Series) -> dict:
    if task == "regression":
        err = actual.to_numpy(float) - predicted.to_numpy(float)
        var = float(np.var(actual.to_numpy(float)))
        return {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())), "bias": float(err.mean()),
                "R2": float(1 - (err ** 2).mean() / var) if var > 0 else np.nan}
    a, p = actual.astype(str), predicted.astype(str)
    return {"accuracy": float(accuracy_score(a, p)), "F1": float(f1_score(a, p, average="weighted"))}


def performance_over_time(bundle: dict, df: pd.DataFrame, date: str | None = None, period: str = "Month",
                          apply_recipe: bool = False, min_rows: int = 20, chunks: int = 5) -> dict:
    """Acierto del modelo en cada periodo (o en `chunks` tramos consecutivos si no hay fecha)."""
    target, task = bundle["target"], bundle["task"]
    if target not in df.columns:
        raise ValueError(f"The data needs the real values in a column named '{target}'.")
    if period not in PERIODS:
        raise ValueError(f"Period must be one of {list(PERIODS)}.")
    data = df.dropna(subset=[target]).reset_index(drop=True)
    if len(data) < 2 * min_rows:
        raise ValueError(f"At least {2 * min_rows} rows with a known result are needed.")
    scored, report = predict_frame(bundle, data, apply_recipe=apply_recipe, interval=False)
    predicted = scored[f"predicted_{target}"]
    if date:
        if date not in data.columns:
            raise ValueError(f"Column '{date}' not found.")
        when = pd.to_datetime(data[date], errors="coerce")
        if when.isna().all():
            raise ValueError(f"'{date}' has no valid dates.")
        label = when.dt.to_period(PERIODS[period]).astype(str).where(when.notna())
    else:
        label = pd.Series(np.minimum(np.arange(len(data)) * chunks // len(data), chunks - 1), index=data.index).map(lambda i: f"part {i + 1}")
    rows = []
    for key in sorted(label.dropna().unique()):
        mask = (label == key).to_numpy()
        row = {"period": key, "rows": int(mask.sum())}
        if mask.sum() >= 2:
            row.update(_metrics(task, data.loc[mask, target], predicted[mask]))
        rows.append(row)
    table = pd.DataFrame(rows)
    table["enough rows"] = table["rows"] >= min_rows

    base = bundle.get("metrics", {})
    key = "MAE" if task == "regression" else "accuracy"
    base_value = base.get(key) if task == "regression" else base.get("Accuracy")
    usable = table[table["enough rows"]]
    alerts = []
    if base_value is not None and len(usable):
        if task == "regression":
            bad = usable[usable[key] > ERROR_FACTOR * base_value]
            alerts += [f"{r['period']}: average error {r[key]:.4g}, {r[key] / base_value:.1f} times the {base_value:.4g} seen in testing."
                       for _, r in bad.iterrows()]
        else:
            bad = usable[usable[key] < base_value - ACCURACY_DROP]
            alerts += [f"{r['period']}: accuracy {r[key]:.1%}, {base_value - r[key]:.0%} points below the {base_value:.1%} seen in testing."
                       for _, r in bad.iterrows()]
    trend = None
    if len(usable) >= 4:
        rho, p = stats.spearmanr(np.arange(len(usable)), usable[key])
        worse = rho > 0 if task == "regression" else rho < 0
        if p < 0.05 and worse:
            trend = {"rho": float(rho), "p": float(p)}
            alerts.append(f"The {key} gets steadily worse over time (rank correlation {rho:+.2f}, p = {p:.3f}).")
    status = "alert" if alerts else "ok"
    overall = _metrics(task, data[target], predicted)
    return {"table": table, "overall": overall, "baseline": base_value, "metric": key, "alerts": alerts, "trend": trend,
            "status": status, "input_report": report, "figure": figure(table, key, base_value),
            "summary": (f"Over {len(data):,} rows, the model's {key} is {overall[key]:.4g}"
                        + (f" against {base_value:.4g} in testing" if base_value is not None else "")
                        + (". It is holding up." if status == "ok" else f". {len(alerts)} warning(s): the model may need retraining."))}


def figure(table: pd.DataFrame, key: str, baseline) -> go.Figure:
    shown = table[table["enough rows"]]
    fig = go.Figure(go.Scatter(x=shown["period"], y=shown[key], mode="lines+markers", name=key))
    if baseline is not None:
        fig.add_hline(y=baseline, line_dash="dot", annotation_text="In testing")
    fig.update_layout(title=f"{key} by period", xaxis_title="Period", yaxis_title=key)
    return fig
