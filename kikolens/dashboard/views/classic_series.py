import pandas as pd
import streamlit as st

from kikolens.analyze import classic as cl
from kikolens.analyze import timeseries as ts
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Classical series"
SUBTITLE = "Is it stationary, what repeats, and an ARIMA forecast that has to beat repeating the last season."


def _series(df: pd.DataFrame):
    dates = ts.detect_datetime_columns(df)
    numeric = list(df.select_dtypes("number").columns)
    if not dates or not numeric:
        ui.note("A date column and a numeric column are needed.", "low")
        return None
    a, b, c = st.columns(3)
    date = a.selectbox("Date column", dates, key="cs_date")
    value = b.selectbox("Value", numeric, key="cs_value")
    agg = c.selectbox("Combine rows of the same period by", ["mean", "sum"], key="cs_agg")
    try:
        return ts.prepare_series(df, date, value, agg=agg), (date, value, agg)
    except ValueError as e:
        ui.note(str(e), "high")
        return None


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


def _stationarity(series, tag) -> None:
    st.caption(t("ADF assumes the series has a unit root; KPSS assumes it does not. Reading them together avoids the mistakes each makes alone."))
    trend = st.checkbox("Allow a trend", key="cs_trend")
    res = _run("cs_stat", (tag, trend), lambda: cl.stationarity(series, "ct" if trend else "c"))
    if res is None:
        return
    ui.note(res["summary"], "ok" if res["differences"] == 0 else "accent")
    _notes(res)
    ui.table(res["table"].round(4))


def _autocorrelation(series, tag) -> None:
    res = _run("cs_acf", tag, lambda: cl.acf_pacf(series))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    ui.chart(res["figure"], 380)


def _stl(series, tag) -> None:
    guess = ts.guess_period(series)
    c1, c2 = st.columns(2)
    period = int(c1.number_input("Observations per season", 2, 400, int(guess or 12), key="cs_period"))
    robust = c2.checkbox("Robust to outliers", True, key="cs_robust")
    res = _run("cs_stl", (tag, period, robust), lambda: cl.stl(series, period, robust))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    ui.stats([("Seasonal strength", f"{res['seasonal_strength']:.2f}"), ("Trend strength", f"{res['trend_strength']:.2f}"), ("Unusual points", f"{len(res['unusual'])}")])
    _notes(res)
    ui.chart(res["figure"], 640)
    ui.download_csv(res["components"].reset_index(names="date"), "stl_components.csv", "Download the components (CSV)")


def _arima(series, tag) -> None:
    guess = ts.guess_period(series)
    c1, c2, c3 = st.columns(3)
    horizon = int(c1.number_input("Periods to forecast", 1, 500, 12, key="cs_hor"))
    period = int(c2.number_input("Observations per season (0 for none)", 0, 400, int(guess or 0), key="cs_aperiod"))
    manual = c3.checkbox("Choose the orders myself", key="cs_manual")
    order, seasonal = None, (0, 0, 0)
    if manual:
        o1, o2, o3, s1, s2, s3 = st.columns(6)
        order = (int(o1.number_input("p", 0, 5, 1, key="cs_p")), int(o2.number_input("d", 0, 2, 1, key="cs_d")), int(o3.number_input("q", 0, 5, 1, key="cs_q")))
        seasonal = (int(s1.number_input("P", 0, 5, 0, key="cs_P")), int(s2.number_input("D", 0, 1, 0, key="cs_D")), int(s3.number_input("Q", 0, 5, 0, key="cs_Q")))

    def fit():
        if order is None:
            return cl.auto_arima(series, period or None, horizon)
        return cl.arima(series, order, seasonal, period or None, horizon)

    res = _run("cs_arima", (tag, horizon, period, order, seasonal), fit)
    if res is None:
        return
    ui.note(res["summary"], "ok")
    skill = res.get("backtest")
    items = [("Model", res["label"]), ("AICc", f"{res['aicc']:.1f}"), ("Residual check (Ljung-Box p)", f"{res['ljung_box']['p_value']:.3f}")]
    if skill:
        items.append(("Error on unseen stretches", f"{skill['model_mae']:.4g}"))
        items.append(("Repeating the last season", f"{skill['naive_mae']:.4g}"))
    ui.stats(items)
    _notes(res)
    ui.chart(res["figure"], 400)
    ui.table(res["forecast"].round(4))
    if "candidates" in res:
        with st.expander(t("Models tried")):
            ui.table(res["candidates"].round(2))


def render(df: pd.DataFrame) -> None:
    found = _series(df)
    if found is None:
        return
    series, tag = found
    tabs = st.tabs(["Stationarity", "Autocorrelation", "STL decomposition", "ARIMA"])
    for tab, fn in zip(tabs, (_stationarity, _autocorrelation, _stl, _arima)):
        with tab:
            fn(series, tag)
