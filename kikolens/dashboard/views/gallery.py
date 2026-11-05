import pandas as pd
import streamlit as st

from kikolens.analyze import visuals as vz
from kikolens.analyze.timeseries import detect_datetime_columns
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Chart gallery"
SUBTITLE = "Flows, hierarchies, profiles, correlation maps and calendars: charts that show what a bar chart cannot."


def _categories(df: pd.DataFrame) -> list:
    return [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 200 and not pd.api.types.is_float_dtype(df[c])]


def _numbers(df: pd.DataFrame) -> list:
    return list(df.select_dtypes("number").columns)


def _build(kind: str, df: pd.DataFrame):
    cats, nums = _categories(df), _numbers(df)
    if kind == "sankey":
        if len(cats) < 2:
            raise ValueError("This needs two columns with a few categories.")
        a, b, c = st.columns(3)
        source = a.selectbox("From", cats, key="gl_s")
        target = b.selectbox("To", [x for x in cats if x != source], key="gl_t")
        value = c.selectbox("Size by", ["(number of rows)"] + nums, key="gl_v")
        return vz.sankey(df, source, target, None if value.startswith("(") else value)
    if kind == "treemap":
        if not cats:
            raise ValueError("This needs at least one column with a few categories.")
        levels = st.multiselect("Levels, from the outside in", cats, default=cats[:2], key="gl_p", max_selections=3)
        value = st.selectbox("Size by", ["(number of rows)"] + nums, key="gl_tv")
        return vz.treemap(df, levels, None if value.startswith("(") else value)
    if kind == "radar":
        if not cats or len(nums) < 3:
            raise ValueError("This needs a column with categories and at least three numeric columns.")
        group = st.selectbox("Compare the groups of", cats, key="gl_g")
        metrics = st.multiselect("Measures", nums, default=nums[:6], key="gl_m")
        return vz.radar(df, group, metrics)
    if kind == "clustered_heatmap":
        method = st.radio("Correlation", ["spearman", "pearson", "kendall"], horizontal=True, key="gl_cm")
        return vz.clustered_heatmap(df, None, method)
    if kind == "correlation_network":
        return vz.correlation_network(df, None, st.slider("Show links of at least", 0.2, 0.95, 0.5, 0.05, key="gl_th"))
    if kind == "calendar_heatmap":
        dates = detect_datetime_columns(df)
        if not dates:
            raise ValueError("This needs a date column.")
        a, b, c = st.columns(3)
        date = a.selectbox("Date", dates, key="gl_cd")
        value = b.selectbox("Measure", ["(number of rows)"] + nums, key="gl_cv")
        agg = c.selectbox("Combine by", ["sum", "mean", "count"], key="gl_ca", disabled=value.startswith("("))
        return vz.calendar_heatmap(df, date, None if value.startswith("(") else value, agg)
    if kind == "ridgeline":
        if not cats or not nums:
            raise ValueError("This needs a numeric column and a column with categories.")
        a, b = st.columns(2)
        return vz.ridgeline(df, a.selectbox("Distribution of", nums, key="gl_rv"), b.selectbox("By", cats, key="gl_rg"))
    if kind == "parallel":
        if len(nums) < 3:
            raise ValueError("This needs at least three numeric columns.")
        columns = st.multiselect("Columns", nums, default=nums[:6], key="gl_pc", max_selections=10)
        color = st.selectbox("Colour by", ["(none)"] + list(df.columns), key="gl_pcol")
        return vz.parallel(df, columns, None if color == "(none)" else color)
    if not cats or not nums:
        raise ValueError("This needs a column with categories and a numeric column.")
    a, b, c, d = st.columns(4)
    x = a.selectbox("X", list(df.columns), key="gl_sx")
    y = b.selectbox("Y", nums, key="gl_sy")
    facet = c.selectbox("One panel per", cats, key="gl_sf")
    shape = d.selectbox("Chart", ["line", "bar", "scatter"], key="gl_sk")
    return vz.small_multiples(df, x, y, facet, shape)


def render(df: pd.DataFrame) -> None:
    names = list(vz.GALLERY)
    choice = st.selectbox("Chart", names, key="gl_kind", format_func=t)
    try:
        fig = _build(vz.GALLERY[choice], df)
    except ValueError as e:
        ui.note(str(e), "low")
        return
    ui.chart(fig, 520 if vz.GALLERY[choice] not in ("clustered_heatmap",) else 600)
    st.caption(t("Hover over the chart for details. Use the camera icon in its corner to save it as an image."))
