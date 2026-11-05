import pandas as pd
import streamlit as st

from kikolens.analyze import marketing as mk
from kikolens.analyze import timeseries as ts
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Marketing mix"
SUBTITLE = "What each channel really adds to sales, how to split a budget, and how much demand falls when the price goes up."


def _mix(df: pd.DataFrame) -> None:
    st.caption(t("Spend on a channel keeps working for a while and each extra unit adds less. The model estimates both, along with trend and season, and checks itself on the last periods."))
    dates = ts.detect_datetime_columns(df)
    numeric = list(df.select_dtypes("number").columns)
    if not dates or len(numeric) < 2:
        ui.note("This needs a date column, a numeric outcome (sales) and numeric spend columns.", "low")
        return
    c1, c2 = st.columns(2)
    date = c1.selectbox("Date", dates, key="mm_d")
    target = c2.selectbox("Outcome (sales, sign-ups...)", numeric, key="mm_t")
    rest = [c for c in numeric if c != target]
    channels = st.multiselect("Channels (spend columns)", rest, default=rest[:3], key="mm_c")
    controls = st.multiselect("Other things that move the outcome (price, weather, holidays...)", [c for c in rest if c not in channels], key="mm_x")
    seasonality = st.checkbox("Take trend and season out", True, key="mm_s")
    signature = (date, target, tuple(channels), tuple(controls), seasonality)
    if st.button("Estimate", key="mm_go", type="primary"):
        try:
            st.session_state["mm_res"] = (signature, mk.marketing_mix(df, date, target, channels, controls, seasonality))
        except ValueError as e:
            st.session_state.pop("mm_res", None)
            ui.note(str(e), "high")
    cached = st.session_state.get("mm_res")
    if not cached or cached[0] != signature:
        return
    res = cached[1]
    ui.note(res["summary"], "ok")
    ui.stats([("Fit (R²)", f"{res['r2']:.0%}"), ("Error on last periods", f"{res['holdout_error']:.0%}"), ("From the baseline", f"{res['baseline_share']:.0%}")])
    for note in res["notes"]:
        ui.note(note, "medium")
    ui.table(res["table"].round(4))
    ui.chart(res["figure"], 380)
    a, b = st.columns(2)
    with a:
        ui.chart(res["curves_figure"], 340)
    with b:
        ui.chart(res["fit_figure"], 340)
    st.subheader(t("Split the budget"))
    total = sum(v["current"] for v in res["_steady"].values())
    c1, c2, c3 = st.columns(3)
    budget = c1.number_input("Spend per period", value=float(round(total, 2)), min_value=0.0, key=f"mm_b_{signature}")
    low = c2.slider("Lowest share of today's spend per channel", 0.0, 1.0, 0.5, key="mm_lo")
    high = c3.slider("Highest multiple of today's spend per channel", 1.0, 4.0, 2.0, key="mm_hi")
    try:
        plan = mk.optimize_budget(res, budget, low, high)
    except ValueError as e:
        ui.note(str(e), "medium")
        return
    ui.note(plan["summary"], "accent")
    ui.chart(plan["figure"], 320)
    ui.table(plan["table"].round(3))


def _elasticity(df: pd.DataFrame) -> None:
    st.caption(t("A 1% higher price changes the quantity sold by the elasticity, in %. Prices are often raised when demand is strong, which hides the effect: an instrument fixes that."))
    numeric = list(df.select_dtypes("number").columns)
    if len(numeric) < 2:
        ui.note("This needs a numeric quantity and a numeric price.", "low")
        return
    c1, c2 = st.columns(2)
    quantity = c1.selectbox("Quantity sold", numeric, key="pe_q")
    price = c2.selectbox("Price", [c for c in numeric if c != quantity], key="pe_p")
    rest = [c for c in numeric if c not in (quantity, price)]
    controls = st.multiselect("Controls (promotions, season, competitor price...)", rest, key="pe_c")
    instrument = st.multiselect("Instrument (a cost or tax that moves the price but not demand)", [c for c in rest if c not in controls], key="pe_i")
    groups = [c for c in df.columns if 2 <= df[c].nunique() <= 30 and c not in numeric]
    group = st.selectbox("One elasticity per", ["(none)"] + groups, key="pe_g")
    cost = st.number_input("Unit cost (0 to look at revenue instead of margin)", min_value=0.0, value=0.0, key="pe_u")
    signature = (quantity, price, tuple(controls), tuple(instrument), group, cost)
    if st.button("Estimate", key="pe_go", type="primary"):
        try:
            st.session_state["pe_res"] = (signature, mk.price_elasticity(df, quantity, price, controls, None if group.startswith("(") else group, instrument or None, cost or None))
        except ValueError as e:
            st.session_state.pop("pe_res", None)
            ui.note(str(e), "high")
    cached = st.session_state.get("pe_res")
    if not cached or cached[0] != signature:
        return
    res = cached[1]
    ui.note(res["summary"], "ok" if res["elasticity"] < 0 else "low")
    ui.stats([("Elasticity", f"{res['elasticity']:+.2f}"), ("95% range", f"{res['interval'][0]:+.2f} to {res['interval'][1]:+.2f}"), ("Rows", f"{res['n']:,}")])
    for note in res["notes"]:
        ui.note(note, "medium")
    ui.chart(res["figure"], 360)
    if res["groups"] is not None and len(res["groups"]):
        ui.table(res["groups"].round(3))


def render(df: pd.DataFrame) -> None:
    tabs = st.tabs(["Channels and budget", "Price elasticity"])
    with tabs[0]:
        _mix(df)
    with tabs[1]:
        _elasticity(df)
