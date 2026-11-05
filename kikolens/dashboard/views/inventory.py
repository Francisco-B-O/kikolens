import pandas as pd
import streamlit as st

from kikolens.analyze import inventory as inv
from kikolens.analyze import timeseries as ts
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Inventory"
SUBTITLE = "Which items matter, how much to order and when, and whether the policy really gives the service you want."


def _run(key: str, signature, fn):
    if st.button("Calculate", key=f"{key}_go", type="primary"):
        try:
            st.session_state[key] = (signature, fn())
        except ValueError as e:
            st.session_state.pop(key, None)
            ui.note(str(e), "high")
    cached = st.session_state.get(key)
    return cached[1] if cached and cached[0] == signature else None


def _notes(res: dict) -> None:
    for note in res.get("notes", []):
        ui.note(note, "medium")


def _items(df: pd.DataFrame) -> None:
    dates = ts.detect_datetime_columns(df)
    numeric = list(df.select_dtypes("number").columns)
    if not numeric:
        ui.note("This needs a column with the item and a numeric quantity sold.", "low")
        return
    c1, c2, c3, c4 = st.columns(4)
    item = c1.selectbox("Item", [c for c in df.columns if c not in numeric] or list(df.columns), key="iv_item")
    quantity = c2.selectbox("Quantity sold", numeric, key="iv_qty")
    date = c3.selectbox("Date (needed for XYZ)", ["(none)"] + dates, key="iv_date")
    value = c4.selectbox("Value of each sale (optional)", ["(none)"] + [c for c in numeric if c != quantity], key="iv_value")
    freq = st.radio("Period", ["D", "W", "M"], index=1, horizontal=True, key="iv_freq", format_func=lambda f: {"D": "Day", "W": "Week", "M": "Month"}[f])
    date = None if date.startswith("(") else date
    value = None if value.startswith("(") else value
    res = _run("iv_abc", (item, quantity, date, value, freq), lambda: inv.classify_items(df, item, quantity, date, value, freq))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    if res["matrix"] is not None:
        st.caption(t("Items by value class (rows) and demand variation (columns)"))
        ui.table(res["matrix"])
    ui.chart(res["figure"], 300)
    ui.table(res["table"].round(3))
    ui.download_csv(res["table"], "item_classification.csv", "Download the classification (CSV)")


def _plan(df: pd.DataFrame) -> None:
    dates = ts.detect_datetime_columns(df)
    numeric = list(df.select_dtypes("number").columns)
    if not dates or not numeric:
        ui.note("This needs an item column, a date column and a numeric quantity sold.", "low")
        return
    c1, c2, c3 = st.columns(3)
    item = c1.selectbox("Item", [c for c in df.columns if c not in numeric and c not in dates] or list(df.columns), key="ip_item")
    date = c2.selectbox("Date", dates, key="ip_date")
    quantity = c3.selectbox("Quantity sold", numeric, key="ip_qty")
    d1, d2, d3, d4 = st.columns(4)
    lead = d1.number_input("Delivery time (days)", 0.5, 365.0, 7.0, key="ip_lead")
    lead_sd = d2.number_input("Its variation (days)", 0.0, 100.0, 0.0, key="ip_leadsd")
    service = d3.slider("Service level", 0.5, 0.999, 0.95, 0.005, key="ip_service")
    freq = d4.radio("Demand period", ["D", "W"], horizontal=True, key="ip_freq", format_func=lambda f: {"D": "Day", "W": "Week"}[f])
    e1, e2 = st.columns(2)
    order_cost = e1.number_input("Cost of placing an order (0 to skip the order quantity)", 0.0, 1e9, 0.0, key="ip_oc")
    holding = e2.number_input("Cost of holding one unit for a year", 0.0, 1e9, 0.0, key="ip_hc")
    res = _run("ip_res", (item, date, quantity, lead, lead_sd, service, freq, order_cost, holding),
               lambda: inv.plan(df, item, date, quantity, lead, service, order_cost or None, holding or None, lead_sd, freq))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    ui.table(res["table"].round(2))
    ui.download_csv(res["table"], "reorder_policies.csv", "Download the policies (CSV)")


def _once(df: pd.DataFrame) -> None:
    st.caption(t("For something sold only once (a season, a perishable): the best quantity balances the cost of leftovers against the profit lost by running out."))
    c1, c2, c3 = st.columns(3)
    cost = c1.number_input("Cost per unit", 0.01, 1e9, 4.0, key="nv_cost")
    price = c2.number_input("Selling price", 0.01, 1e9, 10.0, key="nv_price")
    salvage = c3.number_input("Value of an unsold unit", 0.0, 1e9, 2.0, key="nv_salvage")
    mode = st.radio("Demand", ["Mean and spread", "Past sales in the data"], horizontal=True, key="nv_mode")
    samples = mean = sd = None
    if mode == "Mean and spread":
        a, b = st.columns(2)
        mean = a.number_input("Mean demand", 0.01, 1e9, 100.0, key="nv_mean")
        sd = b.number_input("Standard deviation", 0.0, 1e9, 20.0, key="nv_sd")
    else:
        numeric = list(df.select_dtypes("number").columns)
        if not numeric:
            ui.note("There is no numeric column with past demand.", "low")
            return
        samples = df[st.selectbox("Column with the demand of past seasons", numeric, key="nv_col")]
    res = _run("nv_res", (cost, price, salvage, mean, sd, mode), lambda: inv.newsvendor(cost, price, salvage, mean, sd, samples))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    ui.stats([("Order", f"{res['quantity']:,.1f}"), ("Chance of covering demand", f"{res['probability_cover']:.0%}"), ("Expected profit", f"{res['expected_profit']:,.1f}"),
              ("Expected leftover", f"{res['expected_left_over']:,.1f}")])
    _notes(res)
    ui.chart(res["figure"], 340)


def _policy() -> None:
    st.caption(t("Reorder when stock plus what is on order falls to the reorder point. The simulation shows the service this really gives, not only what the formula promises."))
    c1, c2, c3, c4 = st.columns(4)
    mean = c1.number_input("Mean demand per day", 0.01, 1e9, 10.0, key="rp_mean")
    sd = c2.number_input("Standard deviation per day", 0.0, 1e9, 3.0, key="rp_sd")
    lead = c3.number_input("Delivery time (days)", 1, 365, 4, key="rp_lead")
    lead_sd = c4.number_input("Its variation (days)", 0.0, 100.0, 0.0, key="rp_leadsd")
    d1, d2 = st.columns(2)
    service = d1.slider("Service level wanted", 0.5, 0.999, 0.95, 0.005, key="rp_service")
    quantity = d2.number_input("Units per order", 0.5, 1e9, 60.0, key="rp_q")
    e1, e2 = st.columns(2)
    order_cost = e1.number_input("Cost of placing an order (to suggest the units per order)", 0.0, 1e9, 0.0, key="rp_oc")
    holding = e2.number_input("Cost of holding one unit for a year", 0.0, 1e9, 0.0, key="rp_hc")
    if order_cost > 0 and holding > 0:
        suggestion = inv.eoq(mean * 365, order_cost, holding)
        st.caption(t("Economic order quantity with these costs:") + f" {suggestion['quantity']:,.0f} " + t("units, about") + f" {suggestion['orders_per_year']:,.1f} " + t("orders a year."))

    def both():
        policy = inv.reorder_policy(mean, sd, lead, lead_sd, service, quantity)
        return policy, inv.simulate_policy(mean, sd, int(lead), quantity, policy["reorder_point"], lead_sd)

    res = _run("rp_res", (mean, sd, lead, lead_sd, service, quantity), both)
    if res is None:
        return
    policy, sim = res
    ui.note(policy["summary"], "ok")
    ui.stats([("Safety stock", f"{policy['safety_stock']:,.1f}"), ("Reorder point", f"{policy['reorder_point']:,.1f}"), ("Fill rate (formula)", f"{policy['fill_rate']:.1%}"),
              ("Average stock", f"{sim['average_on_hand']:,.1f}")])
    _notes(policy)
    ui.note(t("Simulated:") + " " + sim["summary"], "accent")


def render(df: pd.DataFrame) -> None:
    tabs = st.tabs(["Which items matter", "Plan for every item", "Sold once", "Reorder policy"])
    with tabs[0]:
        _items(df)
    with tabs[1]:
        _plan(df)
    with tabs[2]:
        _once(df)
    with tabs[3]:
        _policy()
