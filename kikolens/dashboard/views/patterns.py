import pandas as pd
import streamlit as st

from kikolens.ai.insights import get_ai_insights
from kikolens.analyze.discovery import run_discovery_engine
from kikolens.dashboard import components as ui

TITLE = "Patterns"
SUBTITLE = "Correlations, outliers and imbalances found automatically."

ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


@st.cache_data(show_spinner="Looking for patterns...")
def _discover(df: pd.DataFrame):
    return run_discovery_engine(df)


def render(df: pd.DataFrame) -> None:
    insights = sorted(_discover(df), key=lambda i: ORDER.get(i["severity"], 9))
    if not insights:
        ui.note("No significant patterns or anomalies detected.", "ok")
    else:
        counts = pd.Series([i["severity"] for i in insights]).value_counts()
        ui.stats([("Findings", len(insights)), ("High", int(counts.get("high", 0))),
                  ("Medium", int(counts.get("medium", 0))), ("Low / info", int(counts.get("low", 0) + counts.get("info", 0)))])
        for ins in insights:
            kind = {"high": "high", "medium": "medium", "low": "low"}.get(ins["severity"], "accent")
            clean = ins["message"].replace("**", "")
            ui.note(clean, kind, detail=ins["detail"], bold=ins["type"].split(" ", 1)[-1])

    st.markdown("#### Written summary")
    st.caption("Asks the local language model for three takeaways. Nothing leaves this machine.")
    if st.button("Generate summary", key="pat_ai"):
        with st.spinner("Waiting for the local model..."):
            st.session_state.pat_ai_text = get_ai_insights(df)
    if st.session_state.get("pat_ai_text"):
        st.markdown(st.session_state.pat_ai_text)
