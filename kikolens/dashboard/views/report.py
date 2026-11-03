import pandas as pd
import streamlit as st

from kikolens.dashboard import components as ui
from kikolens.report.builder import build_report

TITLE = "Report"
SUBTITLE = "One self-contained page you can share or print to PDF."


def render(df: pd.DataFrame) -> None:
    name = st.text_input("Report title", st.session_state.get("filename") or "dataset", key="rp_name")
    include_models = st.checkbox("Include models trained in this session", True, key="rp_models")
    with_ai = st.checkbox("Add commentary from the local language model", False, key="rp_ai")

    available = {}
    if include_models:
        if "automl_res" in st.session_state:
            available["Classical model"] = st.session_state.automl_res
        if "lab_res" in st.session_state:
            available["Model lab"] = st.session_state.lab_res
        if "nn_res" in st.session_state:
            available["Neural network"] = st.session_state.nn_res
        st.caption("Models included: " + (", ".join(available) if available else "none yet. Train one first."))

    if st.button("Build report", key="rp_go", type="primary"):
        ai_text = None
        if with_ai:
            from kikolens.ai.insights import get_ai_insights
            with st.spinner("Waiting for the local model..."):
                ai_text = get_ai_insights(df)
        with st.spinner("Building report..."):
            st.session_state.rp_html = build_report(df, name=name, models=available, ai_text=ai_text)

    page = st.session_state.get("rp_html")
    if page:
        ui.note(f"Report ready ({len(page) / 1e6:.1f} MB). Open it in a browser and print to save as PDF.", "ok")
        st.download_button("Download report (HTML)", page.encode("utf-8"), "kikolens_report.html", "text/html", key="rp_dl")
