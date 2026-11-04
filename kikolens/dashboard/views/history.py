import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kikolens.analyze import runs as rn
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Run history"
SUBTITLE = "Every model lab training is recorded here, so you can compare attempts and never lose the best one."


def render(df: pd.DataFrame) -> None:
    frame = rn.list_runs()
    if frame.empty:
        ui.note("No runs yet. Train something in the Model lab and it will appear here.", "low")
        return
    targets = ["(all)"] + sorted(frame["target"].unique())
    target = st.selectbox("Column predicted", targets, key="rh_t")
    shown = frame if target == "(all)" else frame[frame["target"] == target]
    metrics = [m for m in ("R2", "F1 (weighted)", "Accuracy") if m in shown.columns and shown[m].notna().any()]
    ui.stats([("Runs", f"{len(shown):,}"), ("Different columns predicted", shown["target"].nunique()),
              ("Best", f"{shown[metrics[0]].max():.4f}" if metrics else "n/a", metrics[0] if metrics else "")])
    if metrics and len(shown) > 1:
        metric = metrics[0]
        ordered = shown.sort_values("created")
        fig = go.Figure(go.Scatter(x=ordered["created"], y=ordered[metric], mode="lines+markers", text=ordered["model"], hovertemplate="%{text}<br>%{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=ordered["created"], y=ordered[metric].cummax(), mode="lines", name="Best so far", line=dict(dash="dot")))
        fig.update_layout(title=f"{metric} of each run over time", showlegend=False)
        ui.chart(fig, 320)
    ui.heat_table(shown.round(4), metrics, hide_index=True)
    st.markdown(f"#### {t('Compare runs')}")
    chosen = st.multiselect("Runs to compare", list(shown["id"]), format_func=lambda i: f"{i} - {shown.set_index('id').loc[i, 'model']} ({shown.set_index('id').loc[i, 'created'][:16]})", key="rh_c")
    if len(chosen) >= 2:
        table = rn.compare(chosen)
        ui.table(table[table["differs"]].drop(columns="differs").astype(str))
    elif chosen:
        st.caption(t("Pick at least two."))
    with st.expander(t("Delete a run")):
        victim = st.selectbox("Run", list(shown["id"]), key="rh_d")
        if st.button("Delete", key="rh_del"):
            rn.delete_run(victim)
            st.rerun()
