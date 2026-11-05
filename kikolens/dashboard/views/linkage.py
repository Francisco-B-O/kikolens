import pandas as pd
import streamlit as st

from kikolens.analyze import linkage as lk
from kikolens.dashboard import components as ui
from kikolens.dashboard import state
from kikolens.utils.loaders import load_data

TITLE = "Match records"
SUBTITLE = "Join two tables whose names are not spelled the same way: accents, word order, typos."


def render(df: pd.DataFrame) -> None:
    upload = st.file_uploader("File to match against", type=["csv", "xlsx", "parquet", "json"], key="lk_file")
    if upload is None:
        st.caption("Upload the second table. Each row of the open data looks for its best partner in it.")
        return
    try:
        other = load_data(upload)
    except Exception as e:
        ui.note(str(e), "high")
        return
    c1, c2, c3 = st.columns(3)
    left_on = c1.selectbox("Name column in the open data", list(df.columns), key="lk_l")
    right_on = c2.selectbox("Name column in the uploaded file", list(other.columns), key="lk_r")
    threshold = c3.slider("Minimum similarity", 0.5, 1.0, 0.85, 0.01, key="lk_t")
    key = (left_on, right_on, threshold, upload.name)
    if st.button("Find matches", key="lk_go", type="primary"):
        try:
            st.session_state.lk_res = (key, lk.link_records(df, other, left_on, right_on, threshold), other)
        except ValueError as e:
            st.session_state.pop("lk_res", None)
            ui.note(str(e), "high")
    cached = st.session_state.get("lk_res")
    if not cached or cached[0] != key:
        return
    res = cached[1]
    ui.note(res["summary"], "accent")
    ui.stats([("Matched", f"{res['matched']:,}"), ("Not matched", f"{res['unmatched']:,}"), ("Need a look", f"{res['ambiguous']:,}")])
    shown = res["matches"][~res["matches"]["exact"]].drop(columns=["left row", "right row", "exact"])
    if len(shown):
        st.markdown("#### Matches that are not exact")
        ui.table(shown.head(200), hide_index=True)
    if st.button("Add the matching columns to my data", key="lk_apply",
                 help="Adds the columns of the uploaded file (ending in _right) and the score. You can undo it."):
        state.apply_change(lk.merge_linked(df, cached[2], res["matches"]), f"Matched records with {upload.name}")
        st.session_state.pop("lk_res", None)
        st.rerun()
