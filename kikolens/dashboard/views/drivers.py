import pandas as pd
import streamlit as st

from kikolens.analyze import drivers
from kikolens.analyze.timeseries import detect_datetime_columns
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Why it changed"
SUBTITLE = "A figure moved between two periods: which regions, channels or products explain it?"


def _span(period) -> tuple:
    """Las fechas elegidas incluyen el último día; el periodo termina al comenzar el siguiente."""
    return pd.Timestamp(period[0]), pd.Timestamp(period[1]) + pd.Timedelta(days=1)


def render(df: pd.DataFrame) -> None:
    dates = detect_datetime_columns(df)
    numeric = list(df.select_dtypes("number").columns)
    if not dates:
        ui.note("This needs a date column to tell the two periods apart.", "low")
        return
    c1, c2, c3 = st.columns(3)
    date = c1.selectbox("Date", dates, key="dr_d")
    agg = c2.selectbox("Measure", ["sum", "mean", "count"], key="dr_a",
                       format_func=lambda a: t({"sum": "Total of a column", "mean": "Average of a column", "count": "Number of rows"}[a]))
    metric = c3.selectbox("Column", numeric, key="dr_m", disabled=agg == "count") if numeric else None
    if agg != "count" and not numeric:
        ui.note("There is no numeric column to measure.", "low")
        return
    try:
        (b0, b1), (a0, a1) = drivers.default_periods(df[date])
    except ValueError as e:
        ui.note(str(e), "high")
        return
    p1, p2 = st.columns(2)
    before = p1.date_input("Earlier period", (b0.date(), (b1 - pd.Timedelta(days=1)).date()), key=f"dr_b_{date}")
    after = p2.date_input("Later period", (a0.date(), (a1 - pd.Timedelta(days=1)).date()), key=f"dr_f_{date}")
    if len(before) != 2 or len(after) != 2:
        st.caption("Pick the start and the end of each period.")
        return
    dims = [c for c in df.columns if c not in (date, metric) and 2 <= df[c].nunique() <= drivers.MAX_VALUES
            and not pd.api.types.is_float_dtype(df[c])]
    chosen = st.multiselect("Break it down by", dims, default=dims[:4], key="dr_dims")
    if st.button("Explain the change", key="dr_go", type="primary"):
        try:
            st.session_state.dr_res = drivers.why_changed(df, metric, date, _span(before), _span(after), chosen or None, agg)
        except ValueError as e:
            st.session_state.pop("dr_res", None)
            ui.note(str(e), "high")
    res = st.session_state.get("dr_res")
    if not res:
        return
    ui.note(res["summary"], "accent")
    ui.stats([("Before", f"{res['total_before']:,.4g}"), ("After", f"{res['total_after']:,.4g}"),
              ("Change", f"{res['change']:+,.4g}"), ("Relative", f"{res['relative_change']:+.1%}")])
    ui.table(res["ranking"].round(3), hide_index=True)
    st.caption("Concentration is how much of the change sits in the single biggest value of each column.")
    dim = st.selectbox("Look at", list(res["tables"]), key="dr_look")
    ui.chart(drivers.waterfall_figure(res, dim), 380)
    ui.table(res["tables"][dim].round(4), hide_index=True)
    if res["pairs"] is not None:
        st.markdown("#### Combinations that moved most")
        ui.table(res["pairs"].round(3), hide_index=True)
