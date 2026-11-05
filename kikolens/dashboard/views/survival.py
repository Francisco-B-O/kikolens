import pandas as pd
import streamlit as st

from kikolens.analyze import survival as sv
from kikolens.dashboard import components as ui

TITLE = "Survival"
SUBTITLE = "How long until something happens (churn, failure, a purchase), counting cases that have not happened yet."


def render(df: pd.DataFrame) -> None:
    numeric = list(df.select_dtypes("number").columns)
    flags = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    if not numeric or not flags:
        ui.note("This needs a numeric duration column and a column saying whether the event happened (two values, such as 0/1).", "low")
        return
    c1, c2, c3 = st.columns(3)
    duration = c1.selectbox("Time until the event or until last seen", numeric, key="sv_d")
    event = c2.selectbox("Did the event happen?", [c for c in flags if c != duration], key="sv_e",
                         help="1 or yes when it happened; 0 or no when the case is still waiting.")
    groups = ["(everyone together)"] + [c for c in df.columns if 2 <= df[c].nunique() <= 8 and c not in (duration, event)]
    group = c3.selectbox("Compare groups by", groups, key="sv_g")
    if st.button("Draw the curves", key="sv_go", type="primary"):
        try:
            st.session_state.sv_res = ((duration, event, group),
                                       sv.survival_by_group(df, duration, event, None if group == groups[0] else group))
        except ValueError as e:
            ui.note(str(e), "high")
            return
    cached = st.session_state.get("sv_res")
    if not cached or cached[0] != (duration, event, group):
        return
    res = cached[1]
    ui.note(res["summary"], "accent")
    ui.chart(res["figure"], 420)
    if res["test"]:
        ui.stats([("Log-rank statistic", f"{res['test']['statistic']:.2f}"), ("Groups", len(res["test"]["groups"])),
                  ("p-value", "< 0.0001" if res["test"]["p_value"] < 1e-4 else f"{res['test']['p_value']:.4f}")])
    table = pd.DataFrame({"group": list(res["medians"]), "median time": [m if m is not None else float("nan")
                                                                         for m in res["medians"].values()]})
    ui.table(table.round(3), hide_index=True)
