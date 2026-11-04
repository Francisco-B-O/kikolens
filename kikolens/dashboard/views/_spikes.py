"""Pestaña de anomalías en series temporales dentro de la sección Anomalías."""
import pandas as pd
import streamlit as st

from kikolens.analyze import timeseries as ts
from kikolens.analyze import tsanomaly as ta
from kikolens.dashboard import components as ui


def render(df: pd.DataFrame) -> None:
    dates = ts.detect_datetime_columns(df)
    numeric = list(df.select_dtypes("number").columns)
    if not dates or not numeric:
        ui.note("Unusual periods need a date column and a numeric column.", "low")
        return
    st.caption("Periods that stand out once the trend and the weekly, monthly or yearly pattern are taken out.")
    c1, c2, c3 = st.columns(3)
    date = c1.selectbox("Date", dates, key="sp_d")
    value = c2.selectbox("Measure", numeric, key="sp_v")
    agg = c3.selectbox("Combine rows of the same period by", ["mean", "sum"], key="sp_a")
    threshold = st.slider("Sensitivity (lower finds more)", 2.0, 8.0, 3.5, 0.5, key="sp_t")
    try:
        res = ta.from_frame(df, date, value, agg=agg, threshold=threshold)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    ui.note(res["summary"], "ok" if not len(res["anomalies"]) else "medium")
    ui.chart(res["figure"], 400)
    if len(res["anomalies"]):
        ui.table(res["anomalies"].round(3), hide_index=True)
        ui.download_csv(res["anomalies"], "unusual_periods.csv", "Download the unusual periods")
