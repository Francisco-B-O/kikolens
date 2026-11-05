import pandas as pd
import streamlit as st

from kikolens.analyze import anomalies as an
from kikolens.dashboard import components as ui
from kikolens.dashboard.views import _spikes
from kikolens.dashboard import state

TITLE = "Anomalies"
SUBTITLE = "Find the rows that do not look like the rest, and see why."

HELP = {
    "Isolation Forest": "Isolates rows with random splits; odd rows need fewer. Good general choice.",
    "Local Outlier Factor": "Compares each row with its neighbours. Finds odd rows inside dense groups.",
    "Robust z-score": "Looks at one column at a time, using the median. Easiest to explain.",
    "Mahalanobis distance": "Distance from the centre taking correlations into account. Best for related columns.",
}


def _rows(df: pd.DataFrame) -> None:
    usable = an.usable_columns(df)
    if not usable:
        ui.note("Anomaly detection needs numeric columns that vary.", "low")
        return
    left, right = st.columns([3, 2], gap="large")
    with left:
        columns = st.multiselect("Columns to examine", usable, default=usable[:12], key="an_cols")
        method = st.radio("Method", an.METHODS, horizontal=True, key="an_method")
        st.caption(HELP[method])
    with right:
        share = st.slider("Share of rows to flag", 0.1, 20.0, 2.0, 0.1, key="an_share", format="%.1f%%") / 100
        st.caption("This is the most unusual slice, not a promise that those rows are errors.")

    if st.button("Find anomalies", key="an_go", type="primary"):
        try:
            with st.spinner("Searching..."):
                st.session_state.an_res = an.detect_anomalies(df, columns, method, share)
        except ValueError as e:
            ui.note(str(e), "high")
            return

    result = st.session_state.get("an_res")
    if not result or len(result["scores"]) != len(df):
        return
    ui.stats([("Flagged rows", f"{result['n_flagged']:,}"), ("Share", f"{result['flags'].mean():.1%}"),
              ("Method", result["method"]), ("Columns used", len(result["columns"]))])
    st.markdown(result["summary"])
    a, b = st.columns(2)
    with a:
        ui.chart(an.anomaly_map(df, result), 380)
    with b:
        ui.chart(an.score_histogram(result), 380)

    st.markdown("#### Flagged rows")
    ui.table(result["table"], height=min(420, 60 + 35 * len(result["table"])))
    c1, c2, c3 = st.columns([1, 1, 3])
    ui.download_csv(result["table"].reset_index(names="row"), "anomalies.csv", "Download flagged rows")
    if c1.button("Remove flagged rows", key="an_drop", help="Drops them from the working data. You can undo it."):
        kept = df[~result["flags"]].reset_index(drop=True)
        state.apply_change(kept, f"Removed {result['n_flagged']:,} anomalous rows ({result['method']}).")
        st.session_state.pop("an_res", None)
        st.rerun()
    st.caption("Removing rows from the dashboard is not stored in the recipe: review them first, they may be real events.")


def render(df: pd.DataFrame) -> None:
    rows, periods = st.tabs(["Unusual rows", "Unusual periods"])
    with rows:
        _rows(df)
    with periods:
        _spikes.render(df)
