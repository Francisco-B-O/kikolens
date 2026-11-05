import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kikolens.analyze import monitor as mt
from kikolens.analyze import predict as pr
from kikolens.dashboard import components as ui
from kikolens.utils.loaders import load_data

TITLE = "Monitoring"
SUBTITLE = "Watch a data source over time: drift, quality, predictions and alerts."

STATUS_KIND = {"ok": "ok", "warning": "medium", "alert": "high", "failed": "high"}


def _create(df: pd.DataFrame) -> None:
    with st.expander("Create a monitor from the data open now", expanded=mt.list_monitors().empty):
        c1, c2 = st.columns(2)
        name = c1.text_input("Name", key="mt_name")
        source = c2.text_input("Path of the file to watch (optional)", key="mt_source",
                               help="A local file or link that gets updated. Leave empty to check files you upload here.")
        models = ["(none)"] + pr.list_models()["file"].tolist()
        model = st.selectbox("Score with a saved model", models, key="mt_model")
        c1, c2, c3 = st.columns(3)
        level = c1.selectbox("Alert when drift reaches", ["high", "medium"], key="mt_level")
        drop = c2.number_input("Quality drop to alert (points)", 1.0, 100.0, 10.0, key="mt_drop")
        change = c3.number_input("Row count change to flag", 0.05, 5.0, 0.5, 0.05, key="mt_rows")
        if st.button("Create monitor", key="mt_create", type="primary"):
            try:
                mt.create_monitor(name.strip() or "monitor", source.strip() or "(uploaded)", st.session_state.original_df,
                                  st.session_state.recipe, None if model == "(none)" else model,
                                  {"drift_level": level, "quality_drop": drop, "rows_change": change})
            except ValueError as e:
                ui.note(str(e), "high")
                return
            st.rerun()


def _show_record(record: dict) -> None:
    ui.note(f"Status: {record['status']}" + (f" · {record['rows']:,} rows" if record.get("rows") is not None else ""),
            STATUS_KIND[record["status"]])
    for alert in record["alerts"]:
        ui.note(alert["message"], alert["severity"])
    if not record["alerts"]:
        st.caption("No alerts.")


def _trend(frame: pd.DataFrame) -> None:
    if len(frame) < 2:
        st.caption("Run the check a few times to see how things move.")
        return
    labels = pd.to_datetime(frame["time"]).dt.strftime("%b %d %H:%M")
    panels = [("drift share", "Drift share (0 to 1)", [0, 1]), ("quality", "Quality score", [0, 100]),
              ("prediction", "Average prediction", None)]
    shown = [(col, title, rng) for col, title, rng in panels if pd.to_numeric(frame[col], errors="coerce").notna().sum() >= 2]
    for column, (col, title, rng) in zip(st.columns(len(shown)), shown):
        fig = go.Figure(go.Scatter(x=labels, y=pd.to_numeric(frame[col], errors="coerce"), mode="lines+markers", name=title))
        fig.update_layout(title=title, showlegend=False)
        fig.update_xaxes(type="category")
        if rng:
            fig.update_yaxes(range=rng)
        with column:
            ui.chart(fig, 280)


def render(df: pd.DataFrame) -> None:
    _create(df)
    monitors = mt.list_monitors()
    if monitors.empty:
        return
    name = st.selectbox("Monitor", monitors["name"].tolist(), key="mt_pick")
    mon = mt.load_monitor(name)
    ui.stats([("Source", mon.source[-28:]), ("Model", mon.model or "none"), ("Drift alert at", mon.thresholds["drift_level"]),
              ("Baseline quality", f"{mon.baseline_quality:.0f}" if mon.baseline_quality is not None else "n/a")])

    file = st.file_uploader("Data to check now", key="mt_file")
    c1, c2 = st.columns([1, 1])
    run_source = c1.button("Check the watched source", key="mt_run_src", disabled=mon.source.startswith("("))
    run_upload = c2.button("Check the uploaded file", key="mt_run_up", type="primary", disabled=file is None)
    if run_source or run_upload:
        try:
            data = load_data(file) if run_upload else None
            st.session_state.mt_record = mt.run_check(mon, data)
        except Exception as e:
            ui.note(str(e), "high")
    record = st.session_state.get("mt_record")
    if record and record["monitor"] == mon.name:
        _show_record(record)

    frame = mt.history(mon.name)
    st.markdown("#### History")
    if frame.empty:
        st.caption("Nothing checked yet.")
    else:
        _trend(frame)
        ui.table(frame.iloc[::-1], hide_index=True)
    with st.expander("Schedule it"):
        every = st.radio("How often", list(mt.EVERY), horizontal=True, key="mt_every")
        st.code(mt.schedule_hint(mon, every), language="bash")
        st.caption("KikoLens does not install anything. Run these commands yourself to schedule the check.")
    if st.button("Delete this monitor", key="mt_del"):
        mt.delete_monitor(mon.name)
        st.session_state.pop("mt_record", None)
        st.rerun()
