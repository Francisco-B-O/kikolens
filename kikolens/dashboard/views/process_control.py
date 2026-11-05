import pandas as pd
import streamlit as st

from kikolens.analyze import spc
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Process control"
SUBTITLE = "Is a process stable or did something change? Control charts, capability against a specification, and small drifts."

CHART_NAMES = {"individuals": "One value per point", "xbar": "Means of subgroups", "p": "Share defective", "u": "Defects per unit", "c": "Number of defects"}


def _run(key: str, signature, fn):
    if st.button("Run", key=f"{key}_go", type="primary"):
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


def _chart(df: pd.DataFrame, numeric: list) -> None:
    c1, c2, c3 = st.columns(3)
    kind = c1.selectbox("Type of chart", list(CHART_NAMES), format_func=lambda k: CHART_NAMES[k], key="pc_kind")
    value = c2.selectbox("Value" if kind in ("individuals", "xbar", "c") else "Count of defective or defects", numeric, key="pc_value")
    order = c3.selectbox("Time order", ["(row order)"] + [c for c in df.columns if c != value], key="pc_order")
    subgroup = size = None
    if kind == "xbar":
        subgroup = st.selectbox("Column that identifies each subgroup", [c for c in df.columns if c != value], key="pc_sub")
    if kind in ("p", "u"):
        size = st.selectbox("How many were examined (or units)", [c for c in numeric if c != value], key="pc_size")
    d1, d2, d3 = st.columns(3)
    use_base = d1.checkbox("Set the limits with the first points only", key="pc_usebase", help="Then the rest are monitored against those fixed limits.")
    baseline = int(d2.number_input("How many first points", spc.MIN_POINTS, max(len(df), spc.MIN_POINTS), min(len(df), 40), key="pc_base", disabled=not use_base))
    rules = tuple(d3.multiselect("Signal rules", [1, 2, 3, 4, 5, 6], default=[1, 4], key="pc_rules",
                                 help="1: a point beyond 3 sigma. 2: 2 of 3 beyond 2 sigma. 3: 4 of 5 beyond 1 sigma. 4: 9 in a row on one side. 5: 6 going up or down. 6: 14 alternating."))
    signature = (kind, value, order, subgroup, size, use_base, baseline, rules)
    res = _run("pc_res", signature, lambda: spc.control_chart(df, value, kind, None if order.startswith("(") else order, subgroup, size, baseline if use_base else None, True, rules or (1,)))
    if res is None:
        return
    ui.note(res["summary"], "ok" if res["in_control"] else "high")
    _notes(res)
    ui.chart(res["figure"], 420)
    if len(res["signals"]):
        shown = res["signals"][["position", "rule", "meaning", "value", "limit_phase"]].copy()
        shown["position"] += 1
        ui.table(shown)


def _capability(df: pd.DataFrame, numeric: list) -> None:
    value = st.selectbox("Measure", numeric, key="pcap_value")
    c1, c2, c3 = st.columns(3)
    lsl = c1.text_input("Lower specification limit", key="pcap_lsl")
    usl = c2.text_input("Upper specification limit", key="pcap_usl")
    target = c3.text_input("Target (optional)", key="pcap_target")

    def number(text, name):
        if not text.strip():
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"The {name} must be a number.") from None

    res = _run("pcap_res", (value, lsl, usl, target), lambda: spc.capability(df[value], number(lsl, "lower limit"), number(usl, "upper limit"), number(target, "target")))
    if res is None:
        return
    ui.note(res["summary"], "ok" if res["band"] == "capable" else "medium" if res["band"] == "marginal" else "high")
    ui.stats([("Cp", "-" if res["cp"] is None else f"{res['cp']:.2f}"), ("Cpk", f"{res['cpk']:.2f}"), ("Pp", "-" if res["pp"] is None else f"{res['pp']:.2f}"), ("Ppk", f"{res['ppk']:.2f}")])
    _notes(res)
    ui.chart(res["figure"], 360)
    st.caption(t("Cpk of 1.33 or more is usually called capable, between 1 and 1.33 marginal, and below 1 not capable."))


def _small_shifts(df: pd.DataFrame, numeric: list) -> None:
    value = st.selectbox("Measure", numeric, key="pcs_value")
    c1, c2 = st.columns(2)
    baseline = int(c1.number_input("Points that define normal", spc.MIN_POINTS, max(len(df), spc.MIN_POINTS), max(spc.MIN_POINTS, len(df) // 3), key="pcs_base"))
    lam = c2.slider("Weight of each new point in the moving average", 0.05, 0.5, 0.2, 0.05, key="pcs_lam")

    def both():
        return spc.cusum(df[value], baseline=baseline), spc.ewma(df[value], lam, baseline=baseline)

    res = _run("pcs_res", (value, baseline, lam), both)
    if res is None:
        return
    one, two = res
    ui.note("CUSUM: " + one["summary"], "ok" if one["signal_at"] is None else "high")
    ui.chart(one["figure"], 300)
    ui.note("EWMA: " + two["summary"], "ok" if two["signal_at"] is None else "high")
    ui.chart(two["figure"], 300)


def render(df: pd.DataFrame) -> None:
    numeric = list(df.select_dtypes("number").columns)
    if not numeric or len(df) < spc.MIN_POINTS:
        ui.note(f"This needs a numeric column with at least {spc.MIN_POINTS} rows, in time order.", "low")
        return
    tabs = st.tabs(["Control chart", "Capability", "Small shifts"])
    with tabs[0]:
        _chart(df, numeric)
    with tabs[1]:
        _capability(df, numeric)
    with tabs[2]:
        _small_shifts(df, numeric)
