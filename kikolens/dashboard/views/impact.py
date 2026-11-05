import pandas as pd
import streamlit as st

from kikolens.analyze import causal
from kikolens.analyze import timeseries as ts
from kikolens.dashboard import components as ui

TITLE = "Impact"
SUBTITLE = "Did a change (a campaign, a price, a new feature) move the numbers? Compare with what would have happened."


def _impact(df: pd.DataFrame) -> None:
    dates = ts.detect_datetime_columns(df)
    numeric = list(df.select_dtypes("number").columns)
    if not dates or not numeric:
        ui.note("This needs a date column and a numeric column to follow.", "low")
        return
    c1, c2, c3 = st.columns(3)
    date = c1.selectbox("Date", dates, key="im_d")
    value = c2.selectbox("Measure", numeric, key="im_v")
    agg = c3.selectbox("Combine rows of the same period by", ["mean", "sum"], key="im_a")
    try:
        series = ts.prepare_series(df, date, value, agg=agg)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    when = st.date_input("When did the change happen?", value=series.index[int(len(series) * 0.75)].date(),
                         min_value=series.index[0].date(), max_value=series.index[-1].date(), key="im_w")
    others = [c for c in numeric if c != value]
    controls = st.multiselect("Control series (measures the change should not have touched)", others, key="im_c",
                              help="Optional but much better: columns that move like the measure but were not affected.")
    if st.button("Estimate the effect", key="im_go", type="primary"):
        control = None
        if controls:
            control = pd.concat([ts.prepare_series(df, date, c, agg=agg).rename(c) for c in controls], axis=1)
        try:
            st.session_state.im_res = ((date, value, str(when), tuple(controls)),
                                       causal.causal_impact(series, pd.Timestamp(when), control))
        except ValueError as e:
            st.session_state.pop("im_res", None)
            ui.note(str(e), "high")
    cached = st.session_state.get("im_res")
    if cached and cached[0] == (date, value, str(when), tuple(controls)):
        res = cached[1]
        ui.note(res["summary"], "ok" if res["detected"] else "low")
        ui.stats([("Average effect", f"{res['average_effect']:+.4g}"), ("Relative", f"{res['relative_effect']:+.1%}"),
                  ("Total", f"{res['total_effect']:+.4g}"), ("p-value", f"{res['p_value']:.3f}")])
        ui.chart(res["figure"], 400)


def _did(df: pd.DataFrame) -> None:
    flags = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    numeric = list(df.select_dtypes("number").columns)
    if len(flags) < 2 or not numeric:
        ui.note("This needs a numeric outcome and two columns with two values each: treated or not, and before or after.", "low")
        return
    c1, c2, c3 = st.columns(3)
    outcome = c1.selectbox("Outcome", numeric, key="im_o")
    treated = c2.selectbox("Treated or not", flags, key="im_t")
    post = c3.selectbox("Before or after", [c for c in flags if c != treated], key="im_p")
    if st.button("Estimate", key="im_dgo", type="primary"):
        try:
            st.session_state.im_did = ((outcome, treated, post), causal.diff_in_diff(df, outcome, treated, post))
        except ValueError as e:
            st.session_state.pop("im_did", None)
            ui.note(str(e), "high")
    cached = st.session_state.get("im_did")
    if cached and cached[0] == (outcome, treated, post):
        res = cached[1]
        ui.note(res["summary"], "ok" if res["p_value"] < 0.05 else "low")
        ui.table(res["means"].round(4))


def render(df: pd.DataFrame) -> None:
    tabs = st.tabs(["Before and after a date", "Treated and control groups"])
    with tabs[0]:
        _impact(df)
    with tabs[1]:
        _did(df)
