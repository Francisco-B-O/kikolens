import pandas as pd
import streamlit as st

from kikolens.analyze import segments as sg
from kikolens.dashboard import components as ui
from kikolens.dashboard import state
from kikolens.dashboard.i18n import t

TITLE = "Segment explorer"
SUBTITLE = "Find groups of similar rows with a Gaussian mixture, hierarchical clustering or DBSCAN, and read each group in words."

METHODS = {"gmm": "Gaussian mixture (finds the number of groups)", "hierarchical": "Hierarchical (dendrogram)", "dbscan": "DBSCAN (free shapes and noise)"}


def render(df: pd.DataFrame) -> None:
    numeric = list(df.select_dtypes("number").columns)
    if len(numeric) < 2:
        ui.note("This needs at least two numeric columns.", "low")
        return
    method = st.radio("Method", list(METHODS), format_func=lambda m: t(METHODS[m]), horizontal=True, key="sx_m")
    columns = st.multiselect("Columns to compare rows by", numeric, default=numeric[:6], key="sx_c")
    options = {}
    if method == "hierarchical":
        options["n_groups"] = st.slider("Groups", 2, 12, 4, key="sx_n")
    elif method == "gmm":
        options["k_max"] = st.slider("Most groups to try", 2, 12, 8, key="sx_k")
    else:
        options["min_samples"] = st.slider("Neighbours needed to form a group", 3, 30, 5, key="sx_ms")
    try:
        res = {"gmm": sg.gmm_auto, "hierarchical": sg.hierarchical, "dbscan": sg.dbscan_auto}[method](df, columns or None, **options)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    ui.note(res["summary"], "accent")
    charts = ([res["map"]] if "map" in res else []) + [res["figure"]]      # DBSCAN solo tiene el mapa
    for column, fig in zip(st.columns(len(charts)), charts):
        with column:
            ui.chart(fig, 380)
    ui.table(res["profiles"].round(3), hide_index=True)
    if st.button("Add the group as a column", key="sx_apply", help="Adds a 'group' column to your data. You can undo it."):
        state.apply_change(df.assign(group=res["labels"]), f"Added groups ({res['method']})")
        st.rerun()
