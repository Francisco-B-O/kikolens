import pandas as pd
import plotly.express as px
import streamlit as st

from kikolens.dashboard import components as ui

TITLE = "Explore"
SUBTITLE = "Build charts from any columns and filter what you see."

CHARTS = ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin", "Density heatmap"]
AGGREGATIONS = ["sum", "mean", "median", "count", "min", "max"]
MAX_POINTS = 20_000


def _filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("Filters", expanded=False):
        chosen = st.multiselect("Filter on columns", list(df.columns), key="ex_filter_cols")
        for col in chosen:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > 1:
                lo, hi = float(s.min()), float(s.max())
                a, b = st.slider(col, lo, hi, (lo, hi), key=f"ex_f_{col}")
                df = df[s.between(a, b)]
            elif pd.api.types.is_datetime64_any_dtype(s) and s.notna().any():
                span = st.date_input(col, (s.min().date(), s.max().date()), key=f"ex_f_{col}")
                if isinstance(span, (tuple, list)) and len(span) == 2:  # mientras se elige, llega una sola fecha
                    df = df[(s.dt.date >= span[0]) & (s.dt.date <= span[1])]
            else:
                values = st.multiselect(col, sorted(s.dropna().astype(str).unique())[:500], key=f"ex_f_{col}")
                if values:
                    df = df[s.astype(str).isin(values)]
    return df


def render(df: pd.DataFrame) -> None:
    view = _filters(df)
    if view.empty:
        ui.note("The filters leave no rows to plot.", "medium")
        return
    st.caption(f"{len(view):,} of {len(df):,} rows")

    ctrl, area = st.columns([1, 3])
    cols = list(view.columns)
    with ctrl:
        kind = st.selectbox("Chart", CHARTS, key="ex_kind")
        x = st.selectbox("X", cols, key="ex_x")
        needs_y = kind not in ("Histogram",)
        y = st.selectbox("Y", cols, index=min(1, len(cols) - 1), key="ex_y") if needs_y else None
        color = st.selectbox("Color", [None] + cols, key="ex_color")
        agg = st.selectbox("Aggregate", AGGREGATIONS, key="ex_agg") if kind == "Bar" else None

    plot = view
    if kind in ("Scatter", "Density heatmap") and len(plot) > MAX_POINTS:
        plot = plot.sample(MAX_POINTS, random_state=0)
        with ctrl:
            st.caption(f"Showing a random {MAX_POINTS:,} points.")

    try:
        if kind == "Scatter":
            fig = px.scatter(plot, x=x, y=y, color=color, opacity=0.7)
        elif kind == "Line":
            fig = px.line(plot.sort_values(x), x=x, y=y, color=color)
        elif kind == "Bar":
            keys = [x] + ([color] if color and color != x else [])
            if agg == "count":
                grouped = plot.groupby(keys, dropna=False).size().reset_index(name="count")
                fig = px.bar(grouped, x=x, y="count", color=color)
            else:
                grouped = plot.groupby(keys, dropna=False)[y].agg(agg).reset_index()
                fig = px.bar(grouped, x=x, y=y, color=color)
        elif kind == "Histogram":
            fig = px.histogram(plot, x=x, color=color, nbins=40)
        elif kind == "Box":
            fig = px.box(plot, x=x, y=y, color=color)
        elif kind == "Violin":
            fig = px.violin(plot, x=x, y=y, color=color, box=True)
        else:
            fig = px.density_heatmap(plot, x=x, y=y, nbinsx=40, nbinsy=40)
    except (ValueError, TypeError) as e:
        with area:
            ui.note(f"This combination cannot be drawn: {e}", "medium")
        return

    with area:
        ui.chart(fig, 480)
