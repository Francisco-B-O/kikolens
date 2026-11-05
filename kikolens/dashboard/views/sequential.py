import pandas as pd
import streamlit as st

from kikolens.analyze import sequential as sq
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Sequential tests"
SUBTITLE = "A/B tests you can check after every visitor without cheating, a planner, and a bandit simulator."


def _live(df: pd.DataFrame) -> None:
    st.caption(t("Unlike a classic test, this one stays valid however often you look at it, so you can stop as soon as the evidence is enough."))
    two = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    numeric = list(df.select_dtypes("number").columns)
    if not two or not numeric:
        ui.note("This needs a column with exactly two groups and a numeric outcome (use 0/1 for conversions).", "low")
        return
    a, b, c, d = st.columns(4)
    group = a.selectbox("Group column (two values)", two, key="sq_g")
    control = b.selectbox("Control group", sorted(df[group].dropna().astype(str).unique()), key="sq_c")
    outcome = c.selectbox("Outcome", [x for x in numeric if x != group], key="sq_o")
    order = d.selectbox("Order of arrival", ["(row order)"] + list(df.columns), key="sq_ord")
    alpha = st.select_slider("Significance level", [0.01, 0.025, 0.05, 0.1], 0.05, key="sq_a")
    try:
        res = sq.from_frame(df, group, outcome, control, None if order.startswith("(") else order, alpha=alpha)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    ui.note(res["summary"], "ok" if res["stopped_at"] else "accent")
    ui.stats([("Decision", res["decision"]), ("Observations per group", f"{res['n']:,}"), ("Difference", f"{res['difference']:+.4g}"),
              ("Always-valid p", f"{res['p_value']:.3g}"), ("Classic p (only valid if looked at once)", f"{res['classic_p']:.3g}")])
    ui.chart(res["figure"], 340)


def _plan() -> None:
    c1, c2, c3 = st.columns(3)
    base = c1.number_input("Current rate", 0.001, 0.99, 0.10, 0.01, key="sq_pb", format="%.3f")
    effect = c2.number_input("Smallest change worth detecting", -0.5, 0.5, 0.02, 0.005, key="sq_pe", format="%.3f")
    runs = c3.select_slider("Simulations", [40, 100, 200], 100, key="sq_pr")
    if st.button("Plan it", key="sq_pgo", type="primary"):
        try:
            with st.spinner(t("Simulating...")):
                st.session_state.sq_plan = sq.planning(base, effect, runs=runs, max_n=12000)
        except ValueError as e:
            st.session_state.pop("sq_plan", None)
            ui.note(str(e), "high")
    res = st.session_state.get("sq_plan")
    if res:
        ui.note(res["summary"], "accent")
        ui.stats([("Fixed test", f"{res['fixed_n']:,}"), ("Sequential, median", f"{res['median_stop']:,}"), ("Sequential, 90% of runs", f"{res['p90_stop']:,}"),
                  ("False alarms with no effect", f"{res['false_positive_rate']:.1%}")])


def _bandit() -> None:
    rates = st.text_input("True conversion rates, one per variant (comma separated)", "0.04, 0.06, 0.09", key="sq_br")
    visitors = st.slider("Visitors", 500, 20000, 5000, 500, key="sq_bn")
    try:
        res = sq.thompson([float(x) for x in rates.split(",")], visitors)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    ui.note(res["summary"], "accent")
    ui.chart(res["figure"], 340)
    st.caption(t("In real life the true rates are unknown; this shows what a bandit would have done if they were these."))


def render(df: pd.DataFrame) -> None:
    tabs = st.tabs(["Live test", "Planner", "Bandit simulator"])
    with tabs[0]:
        _live(df)
    with tabs[1]:
        _plan()
    with tabs[2]:
        _bandit()
