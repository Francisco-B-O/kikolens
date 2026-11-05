"""Piezas comunes a las páginas de AutoML y redes neuronales."""
import io

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kikolens.dashboard import components as ui


def metrics_row(res: dict, extra=()) -> None:
    items = [(k, f"{v:.4f}") for k, v in res["metrics"].items() if isinstance(v, (int, float))]
    ui.stats(items + list(extra))


def export_button(res: dict, key: str) -> None:
    buffer = io.BytesIO()
    joblib.dump(res["model"], buffer)
    st.download_button("Download model (.pkl)", buffer.getvalue(), f"{res['target']}_model.pkl", key=key)


def _base_row(res: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Fila de partida: mediana de cada variable numérica y 0 en las columnas dummy."""
    values = {}
    for col in res["columns"]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            values[col] = float(df[col].median())
        else:
            values[col] = 0.0
    return pd.DataFrame([values], columns=res["columns"])


def predict_row(res: dict, row: pd.DataFrame) -> float:
    return res["model"].predict(res["scaler"].transform(res["imputer"].transform(row)))[0]


def explain_and_simulate(res: dict, df: pd.DataFrame, key: str) -> None:
    st.markdown("#### Explain and simulate")
    tab_row, tab_sim = st.tabs(["Explain a row", "What-if simulator"])

    with tab_row:
        idx = st.number_input("Row index", 0, max(len(df) - 1, 0), 0, key=f"{key}_row")
        if st.button("Explain this row", key=f"{key}_explain"):
            row = df.iloc[int(idx)]
            data = []
            for feature, importance in res["feature_importance"].items():
                base = feature.split("_")[0] if feature not in df.columns else feature
                if base in df.columns and pd.api.types.is_numeric_dtype(df[base]) and pd.notna(row[base]):
                    std = df[base].std() or 1.0
                    data.append({"Factor": base, "Impact": float((row[base] - df[base].mean()) / std * importance)})
            if data:
                frame = pd.DataFrame(data).drop_duplicates("Factor").sort_values("Impact")
                fig = go.Figure(go.Bar(x=frame["Impact"], y=frame["Factor"], orientation="h"))
                fig.update_layout(title="Approximate contribution of each factor (standardized value x importance)")
                ui.chart(fig, 360)
            else:
                ui.note("No numeric features available to explain this row.", "low")

    with tab_sim:
        top = list(res["feature_importance"].keys())[:5]
        if not top:
            ui.note("No features available for simulation.", "low")
            return
        base = _base_row(res, df)
        inputs = []  # (nombre, es_categórica); una variable categórica aporta varias columnas dummy
        for feature in top:
            categorical = next((c for c in res["categorical_mappings"] if feature.startswith(f"{c}_")), None)
            entry = (categorical, True) if categorical else (feature, False)
            if entry not in inputs and (categorical or (feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]))):
                inputs.append(entry)
        if not inputs:
            ui.note("None of the most influential features can be simulated directly.", "low")
            return
        chosen, cols = {}, st.columns(len(inputs))
        for i, (name, is_categorical) in enumerate(inputs):
            if is_categorical:
                chosen[name] = cols[i].selectbox(name, res["categorical_mappings"][name], key=f"{key}_sim_{name}")
                continue
            lo, hi = float(df[name].min()), float(df[name].max())
            if lo < hi:
                chosen[name] = cols[i].slider(name, lo, hi, float(df[name].median()), key=f"{key}_sim_{name}")
            else:
                chosen[name] = cols[i].number_input(name, value=lo, key=f"{key}_sim_{name}")
        if st.button("Predict", key=f"{key}_simulate", type="primary"):
            row = base.copy()
            for name, value in chosen.items():
                if name in row.columns:
                    row[name] = value
                else:
                    for col in [c for c in row.columns if c.startswith(f"{name}_")]:
                        row[col] = 0.0
                    dummy = f"{name}_{value}"
                    if dummy in row.columns:
                        row[dummy] = 1.0
            prediction = predict_row(res, row)
            label = f"{prediction:.4f}" if res["task_type"] == "regression" else str(prediction)
            st.metric("Predicted outcome", label)
