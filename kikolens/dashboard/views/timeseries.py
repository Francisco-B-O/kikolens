import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from kikolens.analyze import timeseries as ts
from kikolens.dashboard import components as ui
from kikolens.dashboard.theme import palette

TITLE = "Time series"
SUBTITLE = "Trend, seasonality and a forecast checked against a simple baseline."


def render(df: pd.DataFrame) -> None:
    date_cols = ts.detect_datetime_columns(df)
    numeric = df.select_dtypes("number").columns.tolist()
    if not date_cols or not numeric:
        ui.note("A date column and a numeric column are needed.", "medium",
                detail="Text columns that look like dates are detected automatically. "
                       "Convert a column in Quality and cleaning if yours was missed.")
        return

    c1, c2, c3, c4 = st.columns(4)
    date_col = c1.selectbox("Date column", date_cols, key="ts_date")
    value_col = c2.selectbox("Value", numeric, key="ts_value")
    freq_label = c3.selectbox("Period", ["Auto"] + list(ts.FREQUENCIES), key="ts_freq")
    agg = c4.selectbox("Combine rows by", ts.AGGREGATIONS, key="ts_agg")
    horizon = st.slider("Periods to forecast", 1, 120, 12, key="ts_h")

    if st.button("Analyze", key="ts_go", type="primary"):
        try:
            series = ts.prepare_series(df, date_col, value_col,
                                       freq=None if freq_label == "Auto" else ts.FREQUENCIES[freq_label], agg=agg)
            st.session_state.ts_res = {
                "series": series, "parts": ts.decompose(series), "fc": ts.forecast(series, horizon),
                "trend": ts.trend_summary(series), "key": (date_col, value_col),
            }
        except ValueError as e:
            st.session_state.pop("ts_res", None)
            ui.note(str(e), "high")

    res = st.session_state.get("ts_res")
    if not res or res["key"] != (date_col, value_col):
        return

    series, parts, fc, trend = res["series"], res["parts"], res["fc"], res["trend"]
    change = f"{trend['change_pct']:+.1f}% start to end" if trend["change_pct"] is not None else ""
    beats = fc["metrics"]["Beats naive"]
    ui.stats([
        ("Trend", trend["direction"].capitalize(), change),
        ("Seasonal period", parts["period"] or "none",
         f"strength {parts['seasonal_strength']:.0%}" if parts["seasonal_strength"] is not None else ""),
        ("Backtest error (MAE)", f"{fc['metrics']['MAE']:.4g}", f"naive: {fc['metrics']['Naive MAE']:.4g}"),
        ("Periods", len(series)),
    ])
    if beats:
        ui.note("The forecast beat a naive 'repeat the last value' baseline on held-out data.", "ok")
    else:
        ui.note("The forecast did not beat a naive baseline on held-out data. Treat it with caution.", "medium",
                detail="This usually means the series is noisy or has no stable pattern.")

    p = palette()
    forecast = fc["forecast"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, name="Observed", mode="lines", line=dict(color=p["ink"], width=1.6)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["upper"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["lower"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=p["accent_soft"], name="95% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["forecast"], name="Forecast", mode="lines",
                             line=dict(color=p["accent"], width=2, dash="dash")))
    fig.update_layout(title=f"{value_col} over time")
    ui.chart(fig, 440)

    st.markdown("#### Components")
    comp = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         subplot_titles=("Trend", "Seasonal pattern", "Remainder"))
    for row, key in enumerate(("trend", "seasonal", "resid"), start=1):
        comp.add_trace(go.Scatter(x=series.index, y=parts[key], showlegend=False,
                                  line=dict(color=p["colorway"][row - 1], width=1.4)), row=row, col=1)
    ui.chart(comp, 620)

    with st.expander("Forecast table"):
        table = forecast.rename_axis("date").reset_index()
        ui.table(table.round(4), hide_index=True)
        ui.download_csv(table, "forecast.csv", "Download forecast (CSV)")
