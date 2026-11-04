import pandas as pd
import streamlit as st

from kikolens.analyze import champion
from kikolens.analyze import perfwatch as pw
from kikolens.analyze import predict as pr
from kikolens.analyze.timeseries import detect_datetime_columns
from kikolens.dashboard import components as ui
from kikolens.dashboard.views.predict import _registry_choices, _session_bundles
from kikolens.utils.loaders import load_data

TITLE = "Model health"
SUBTITLE = "Is a model still as good as when you tested it? Score newer data whose real results are known."


def _compare(df: pd.DataFrame) -> None:
    options, saved = _session_bundles(df), _registry_choices()
    labels = list(options) + list(saved)
    if len(labels) < 2:
        ui.note("Save at least two models (or train one in this session and save another) to compare them.", "low")
        return
    chosen = st.multiselect("Models to compare", labels, default=labels[:2], key="mh_cmp")
    upload = st.file_uploader("Data with the real results", type=["csv", "xlsx", "parquet", "json"], key="mh_cmp_upload")
    if st.button("Compare", key="mh_cmp_go", type="primary", disabled=len(chosen) < 2):
        try:
            data = load_data(upload) if upload is not None else df
            bundles = {c: options[c] if c in options else pr.load_bundle(pr.resolve_model(saved[c])) for c in chosen}
            st.session_state.mh_cmp_res = champion.compare_saved_models(bundles, data)
        except ValueError as e:
            st.session_state.pop("mh_cmp_res", None)
            ui.note(str(e), "high")
    res = st.session_state.get("mh_cmp_res")
    if res:
        ui.note(res["summary"], "ok" if res["reliable"] else "medium")
        ui.table(res["table"].round(4), hide_index=True)


def render(df: pd.DataFrame) -> None:
    health, compare = st.tabs(["Check a model", "Compare models"])
    with health:
        _health(df)
    with compare:
        _compare(df)


def _health(df: pd.DataFrame) -> None:
    options = _session_bundles(df)
    saved = _registry_choices()
    labels = list(options) + list(saved)
    if not labels:
        ui.note("Train a model first (Models, Model lab or Neural network) or save one, then come back with newer data.", "low")
        return
    choice = st.selectbox("Model", labels, key="mh_model")
    try:
        bundle = options[choice] if choice in options else pr.load_bundle(pr.resolve_model(saved[choice]))
    except ValueError as e:
        ui.note(str(e), "high")
        return
    target = bundle["target"]
    st.caption(f"This model predicts '{target}'. The data needs that column filled with what really happened.")
    upload = st.file_uploader("Data with the real results", type=["csv", "xlsx", "parquet", "json"], key="mh_upload")
    data = df
    if upload is not None:
        try:
            data = load_data(upload)
        except Exception as e:
            ui.note(str(e), "high")
            return
    elif target not in df.columns:
        ui.note(f"The open data has no '{target}' column. Upload a file that has it.", "low")
        return
    dates = ["(none, split in order)"] + detect_datetime_columns(data)
    c1, c2 = st.columns(2)
    date = c1.selectbox("Date", dates, key="mh_date")
    period = c2.selectbox("Period", list(pw.PERIODS), index=1, key="mh_period", disabled=date == dates[0])
    raw = st.checkbox("Apply the cleaning steps used for training first", False, key="mh_raw",
                      help="Use this when the file is raw, as it was before any cleaning.")
    if st.button("Check the model", key="mh_go", type="primary"):
        try:
            st.session_state.mh_res = pw.performance_over_time(bundle, data, None if date == dates[0] else date, period, raw)
        except ValueError as e:
            st.session_state.pop("mh_res", None)
            ui.note(str(e), "high")
    res = st.session_state.get("mh_res")
    if not res:
        return
    ui.note(res["summary"], "ok" if res["status"] == "ok" else "high")
    for line in res["alerts"]:
        ui.note(line, "medium")
    ui.chart(res["figure"], 360)
    ui.table(res["table"].drop(columns=["enough rows"]).round(4), hide_index=True)
    st.caption("Periods with fewer than 20 rows are shown in the table but not judged.")
