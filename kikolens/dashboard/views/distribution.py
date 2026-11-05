import pandas as pd
import streamlit as st

from kikolens.analyze import distributions as dist
from kikolens.dashboard import components as ui

TITLE = "Distribution fit"
SUBTITLE = "Which distribution describes a column best, and does it look normal?"


def render(df: pd.DataFrame) -> None:
    numeric = list(df.select_dtypes("number").columns)
    if not numeric:
        ui.note("This needs a numeric column.", "low")
        return
    c1, c2 = st.columns([1, 2])
    column = c1.selectbox("Column", numeric, key="di_c")
    names = c2.multiselect("Distributions to try", list(dist.CANDIDATES), default=list(dist.CANDIDATES), key="di_n")
    try:
        res = dist.fit_distributions(df[column], names or None)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    ui.note(res["summary"], "accent")
    ui.stats([("Best fit", res["best"]), ("Rows used", f"{res['n']:,}"), ("Skewness", f"{res['skewness']:+.2f}"),
              ("Excess kurtosis", f"{res['excess_kurtosis']:+.2f}")])
    left, right = st.columns(2)
    with left:
        ui.chart(res["figure"], 360)
    with right:
        ui.chart(res["qq"], 360)
    ui.table(res["table"].round(4), hide_index=True)
    st.caption("A lower AIC is better; a difference under 2 means the fits are practically tied. "
               "The Kolmogorov-Smirnov p-value is only a rough guide when the parameters were estimated from the same data.")
