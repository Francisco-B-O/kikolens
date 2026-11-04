import pandas as pd
import streamlit as st

from kikolens.analyze import customers as cu
from kikolens.dashboard import components as ui

TITLE = "Customers"
SUBTITLE = "Retention by cohort, valuable customer segments, conversion funnels and what is bought together."


def _run(key, fn, *args, **kwargs):
    if st.button("Run", key=f"{key}_go", type="primary"):
        try:
            st.session_state[key] = fn(*args, **kwargs)
        except ValueError as e:
            st.session_state.pop(key, None)
            ui.note(str(e), "high")
    return st.session_state.get(key)


def _dates(df):
    from kikolens.analyze.timeseries import detect_datetime_columns
    return detect_datetime_columns(df)


def _cohorts(df: pd.DataFrame) -> None:
    dates = _dates(df)
    if not dates:
        ui.note("Retention needs a date column and a column that identifies the customer or user.", "low")
        return
    c1, c2, c3 = st.columns(3)
    user = c1.selectbox("Customer or user", df.columns, key="cu_cu")
    date = c2.selectbox("Date of activity", dates, key="cu_cd")
    period = c3.selectbox("Cohort period", list(cu.PERIODS), index=2, key="cu_cp")
    res = _run("cu_cohort", cu.cohort_retention, df, user, date, period)
    if res:
        ui.note(res["summary"], "accent")
        ui.chart(res["figure"], 120 + 32 * len(res["matrix"]))
        ui.table((res["matrix"] * 100).round(1))


def _rfm(df: pd.DataFrame) -> None:
    dates = _dates(df)
    numeric = list(df.select_dtypes("number").columns)
    if not dates or not numeric:
        ui.note("RFM needs a customer column, a date and an amount.", "low")
        return
    c1, c2, c3 = st.columns(3)
    customer = c1.selectbox("Customer", df.columns, key="cu_rc")
    date = c2.selectbox("Date", dates, key="cu_rd")
    amount = c3.selectbox("Amount spent", numeric, key="cu_ra")
    res = _run("cu_rfm", cu.rfm, df, customer, date, amount)
    if res:
        ui.note(res["summary"], "accent")
        ui.table(res["segments"].round(3), hide_index=True)
        with st.expander("Every customer"):
            ui.table(res["customers"].round(2), hide_index=True)
        ui.download_csv(res["customers"], "customer_segments.csv", "Download customers with segments")


def _funnel(df: pd.DataFrame) -> None:
    cats = [c for c in df.columns if 2 <= df[c].nunique() <= 20]
    if not cats:
        ui.note("A funnel needs a column that names the stage each event reached (2 to 20 stages).", "low")
        return
    c1, c2 = st.columns(2)
    user = c1.selectbox("User", df.columns, key="cu_fu")
    stage = c2.selectbox("Stage", cats, key="cu_fs")
    order = st.multiselect("Stages in order, first to last", sorted(df[stage].dropna().astype(str).unique()), key="cu_fo")
    if len(order) < 2:
        st.caption("Pick at least two stages, in the order users go through them.")
        return
    res = _run("cu_funnel", cu.funnel, df, user, stage, order)
    if res:
        ui.note(res["summary"], "accent")
        ui.chart(res["figure"], 360)
        ui.table(res["table"].round(3), hide_index=True)


def _basket(df: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)
    transaction = c1.selectbox("Order or basket id", df.columns, key="cu_bt")
    item = c2.selectbox("Item", [c for c in df.columns if c != transaction], key="cu_bi")
    s1, s2 = st.columns(2)
    support = s1.slider("Minimum share of baskets", 0.005, 0.3, 0.02, 0.005, key="cu_bs")
    confidence = s2.slider("Minimum confidence", 0.1, 0.9, 0.3, 0.05, key="cu_bc")
    res = _run("cu_basket", cu.association_rules, df, transaction, item, support, confidence)
    if res:
        ui.note(res["summary"], "accent")
        ui.table(res["rules"].round(3), hide_index=True)
        st.caption("Lift above 1 means the items are bought together more often than chance would give.")


def render(df: pd.DataFrame) -> None:
    tabs = st.tabs(["Retention", "Segments (RFM)", "Funnel", "Bought together"])
    for tab, body in zip(tabs, (_cohorts, _rfm, _funnel, _basket)):
        with tab:
            body(df)
