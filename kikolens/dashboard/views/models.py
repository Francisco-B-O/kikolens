import joblib
import pandas as pd
import streamlit as st

from kikolens.analyze.automl import run_automl_analysis
from kikolens.dashboard import components as ui
from kikolens.dashboard.views import _modeling as modeling

TITLE = "Models"
SUBTITLE = "Classical machine learning, chosen and explained automatically."

ALGORITHMS = ["Random Forest", "Gradient Boosting", "Decision Tree", "Ridge/Logistic", "Lasso/KNN"]


def render(df: pd.DataFrame) -> None:
    mode = st.radio("Mode", ["Automatic", "Choose algorithm", "Import trained model"], horizontal=True, key="aml_mode")
    target = st.selectbox("Column to predict", df.columns, index=len(df.columns) - 1, key="aml_target")

    manual_type, imported = None, None
    if mode == "Choose algorithm":
        manual_type = st.selectbox("Algorithm", ALGORITHMS, key="aml_algo")
    elif mode == "Import trained model":
        upload = st.file_uploader("Upload a .pkl file", type=["pkl"], key="aml_pkl")
        st.caption("Only load files you created yourself. A .pkl file can run code when opened.")
        if upload:
            try:
                imported = joblib.load(upload)
            except Exception as e:
                ui.note(f"Could not read the model: {e}", "high")

    if st.button("Train", key="aml_train", type="primary"):
        if mode == "Import trained model" and imported is None:
            ui.note("Upload a valid model first.", "medium")
        else:
            try:
                algorithm = "Random Forest" if mode == "Import trained model" else manual_type
                with st.spinner("Training..."):
                    res = run_automl_analysis(df, target, manual_type=algorithm)
                    if imported is not None:
                        res["model"] = imported
                        res["narrative"] = "### Imported model\nPredictions use your uploaded model. The metrics and charts above come from a Random Forest trained just now on this data, not from your model."
                st.session_state.automl_res = res
            except ValueError as e:
                ui.note(str(e), "high")

    res = st.session_state.get("automl_res")
    if not res or res.get("target") != target:
        return

    modeling.metrics_row(res, [("Task", res["task_type"]), ("Features", len(res["columns"]))])
    st.markdown(res["narrative"])
    left, right = st.columns(2)
    with left:
        ui.chart(res["plots"]["performance"])
    with right:
        ui.chart(res["plots"]["importance"])
    modeling.export_button(res, "aml_export")
    st.divider()
    modeling.explain_and_simulate(res, df, "aml")
