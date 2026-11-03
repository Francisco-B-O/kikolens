import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from kikolens.analyze import forecasting as fx
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

    st.divider()
    _advanced(df, series, date_col, value_col, agg, horizon)


def _advanced(df: pd.DataFrame, series: pd.Series, date_col: str, value_col: str, agg: str, horizon: int) -> None:
    st.markdown("#### Go further")
    a, b, c = st.tabs(["Compare models", "Changes in the series", "One forecast per group"])
    with a:
        st.caption("Each model is trained on the past and tested on the stretch that follows, several times. The most accurate wins.")
        events = st.text_input("Special dates (optional, comma separated, e.g. 2024-11-29, 2024-12-24)", key="ts_events")
        if st.button("Compare models", key="ts_compare", type="primary"):
            try:
                dates = [d.strip() for d in events.split(",") if d.strip()]
                with st.spinner("Testing the models..."):
                    st.session_state.ts_models = (value_col, fx.forecast_models(series, horizon, events=pd.to_datetime(dates) if dates else None))
            except (ValueError, TypeError) as e:
                ui.note(str(e), "high")
        cached = st.session_state.get("ts_models")
        if cached and cached[0] == value_col:
            res = cached[1]
            ui.note(res["summary"], "ok" if res["beats_naive"] else "medium")
            ui.chart(fx.forecast_figure(series, res), 420)
            ui.table(res["leaderboard"].round(4), hide_index=True)
            table = res["forecast"].rename_axis("date").reset_index()
            ui.download_csv(table, "forecast_best_model.csv", "Download this forecast (CSV)")
    with b:
        sensitivity = st.slider("Sensitivity", 0.3, 3.0, 1.0, 0.1, key="ts_sens", help="Lower finds more changes.")
        try:
            points = fx.detect_changepoints(series, 5, sensitivity=sensitivity)
        except ValueError as e:
            ui.note(str(e), "low")
        else:
            ui.note(f"{len(points)} change(s) found." if points else "No clear level shift or trend change.", "medium" if points else "ok")
            ui.chart(fx.changepoint_figure(series, points), 340)
            if points:
                ui.table(pd.DataFrame(points).round(3), hide_index=True)
    with c:
        groups = [col for col in df.columns if col not in (date_col, value_col) and 2 <= df[col].nunique() <= 60]
        if not groups:
            ui.note("This needs a column that splits the rows into groups (such as a region or a product).", "low")
        else:
            group = st.selectbox("Group by", groups, key="ts_group")
            if st.button("Forecast every group", key="ts_by_group"):
                try:
                    with st.spinner("Forecasting each group..."):
                        st.session_state.ts_groups = (group, fx.forecast_by_group(df, date_col, value_col, group, horizon, agg=agg))
                except ValueError as e:
                    ui.note(str(e), "high")
            cached = st.session_state.get("ts_groups")
            if cached and cached[0] == group:
                res = cached[1]
                ui.table(res["summary"].round(3), hide_index=True)
                if len(res["skipped"]):
                    ui.note(f"{len(res['skipped'])} group(s) had too little data and were skipped.", "low")
                ui.download_csv(res["forecast"], "forecast_by_group.csv", "Download all forecasts (CSV)")
