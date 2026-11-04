import pandas as pd
import streamlit as st

from kikolens.analyze import multivar as mv
from kikolens.analyze.timeseries import detect_datetime_columns
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Forecast with drivers"
SUBTITLE = "Forecast a series using other columns too, test whether they really help, and try what-if scenarios."


def render(df: pd.DataFrame) -> None:
    dates, numeric = detect_datetime_columns(df), list(df.select_dtypes("number").columns)
    if not dates or len(numeric) < 2:
        ui.note("This needs a date column and at least two numeric columns.", "low")
        return
    a, b, c = st.columns(3)
    date = a.selectbox("Date", dates, key="mv_d")
    target = b.selectbox("Column to forecast", numeric, key="mv_t")
    agg = c.selectbox("Combine rows of the same period by", ["mean", "sum"], key="mv_a")
    drivers = st.multiselect("Drivers", [x for x in numeric if x != target], default=[x for x in numeric if x != target][:2], key="mv_dr")
    tabs = st.tabs(["Forecast", "Does it help predict? (Granger)"])
    with tabs[0]:
        horizon = st.slider("Periods to forecast", 3, 60, 14, key="mv_h")
        scenario = {}
        if drivers:
            st.caption(t("What-if: how much higher or lower will each driver be in the future, compared with its last value? 1.0 means the same."))
            cols = st.columns(len(drivers))
            for col, d in zip(cols, drivers):
                factor = col.number_input(d, 0.1, 5.0, 1.0, 0.05, key=f"mv_f_{d}")
                if abs(factor - 1.0) > 1e-9:
                    scenario[d] = factor
        if st.button("Forecast", key="mv_go", type="primary"):
            try:
                with st.spinner(t("Testing on the past...")):
                    st.session_state.mv_res = mv.forecast_with_drivers(df, date, target, drivers, horizon, agg=agg, scenario=scenario)
            except ValueError as e:
                st.session_state.pop("mv_res", None)
                ui.note(str(e), "high")
        res = st.session_state.get("mv_res")
        if res:
            ui.note(res["summary"], "ok" if res["drivers_help"] else "medium")
            ui.stats([("Error with drivers", f"{res['mae_with_drivers']:.4g}"), ("Error alone", f"{res['mae_alone']:.4g}"), ("p-value", f"{res['p_value']:.3g}")])
            ui.chart(res["figure"], 400)
            ui.table(res["forecast"].round(3))
    with tabs[1]:
        if not drivers:
            st.caption(t("Pick at least one driver."))
        else:
            cause = st.selectbox("Does the past of", drivers, key="mv_gc")
            lag = st.slider("Most periods back", 1, 21, 7, key="mv_gl")
            try:
                res = mv.granger(df, date, cause, target, lag, agg)
            except ValueError as e:
                ui.note(str(e), "high")
            else:
                ui.note(res["summary"], "ok" if res["significant"] else "low")
                ui.chart(res["figure"], 320)
                ui.table(res["table"].round(4), hide_index=True)
