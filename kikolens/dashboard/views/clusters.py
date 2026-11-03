import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from kikolens.analyze.clustering import run_clustering_analysis, suggest_k
from kikolens.dashboard import components as ui

TITLE = "Segments"
SUBTITLE = "Group similar rows together with k-means."


@st.cache_data(show_spinner="Comparing cluster counts...")
def _suggest(df: pd.DataFrame) -> pd.DataFrame:
    return suggest_k(df)


def render(df: pd.DataFrame) -> None:
    if df.select_dtypes("number").shape[1] < 2:
        ui.note("At least two numeric columns are needed.", "medium")
        return

    with st.expander("Help me choose the number of segments"):
        try:
            scores = _suggest(df)
        except ValueError as e:
            ui.note(str(e), "medium")
        else:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=scores["k"], y=scores["inertia"], name="Inertia (lower is tighter)"), secondary_y=False)
            fig.add_trace(go.Scatter(x=scores["k"], y=scores["silhouette"], name="Silhouette (higher is better)"), secondary_y=True)
            fig.update_xaxes(title="Number of segments", dtick=1)
            ui.chart(fig, 320)
            best = int(scores.loc[scores["silhouette"].idxmax(), "k"])
            st.caption(f"Best silhouette at k = {best}. Use it as a hint, not a rule.")

    k = st.slider("Number of segments", 2, 8, 3, key="cl_k")
    if st.button("Segment", key="cl_go", type="primary"):
        with st.spinner("Segmenting..."):
            st.session_state.cluster_res = (k, run_clustering_analysis(df, k))

    saved = st.session_state.get("cluster_res")
    if not saved:
        return
    k_used, res = saved
    sizes = res["df_with_clusters"]["Kiko_Cluster"].value_counts().sort_index()
    ui.stats([(f"Segment {i}", f"{n:,}", f"{n / len(df):.0%} of rows") for i, n in sizes.items()])
    ui.chart(res["fig"], 480)
    st.markdown("#### Segment profiles (column means)")
    ui.table(res["profiles"].round(3))
    ui.download_csv(res["df_with_clusters"], "segmented.csv", "Download data with segment labels (CSV)")
