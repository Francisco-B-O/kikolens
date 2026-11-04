import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from kikolens.analyze.stats import basic_statistics, correlation_matrix
from kikolens.dashboard import components as ui
from kikolens.dashboard.theme import palette

TITLE = "Statistics"
SUBTITLE = "Distributions, summaries and relationships between variables."


def _top_pairs(corr: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().rename("correlation").reset_index()
    pairs.columns = ["variable A", "variable B", "correlation"]
    return pairs.reindex(pairs["correlation"].abs().sort_values(ascending=False).index).head(n)


def render(df: pd.DataFrame) -> None:
    summary, dist, corr_tab = st.tabs(["Summary", "Distributions", "Correlation"])

    with summary:
        ui.table(basic_statistics(df).round(3))

    with dist:
        numeric = df.select_dtypes("number").columns.tolist()
        col = st.selectbox("Column", df.columns, key="st_col")
        if col in numeric:
            s = df[col].dropna()
            ui.stats([("Mean", f"{s.mean():.4g}"), ("Median", f"{s.median():.4g}"),
                      ("Std. dev.", f"{s.std():.4g}"), ("Skewness", f"{s.skew():.2f}"),
                      ("Min / Max", f"{s.min():.4g} / {s.max():.4g}")])
            a, b = st.columns(2)
            with a:
                ui.chart(px.histogram(s, nbins=40, title=f"Distribution of {col}").update_layout(showlegend=False), 340)
            with b:
                ui.chart(px.box(s, points="outliers", title="Spread and outliers").update_layout(showlegend=False), 340)
        else:
            counts = df[col].astype(str).value_counts().head(25).reset_index()
            counts.columns = [col, "count"]
            ui.chart(px.bar(counts, x="count", y=col, orientation="h", title=f"Most frequent values of {col}")
                     .update_yaxes(autorange="reversed"), 440)
            st.caption(f"{df[col].nunique():,} distinct values · {df[col].isna().sum():,} missing")

    with corr_tab:
        corr = correlation_matrix(df)
        if corr.empty or len(corr) < 2:
            ui.note("At least two numeric, non-constant columns are needed.", "low")
            return
        p = palette()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
            colorscale=[[0, p["colorway"][1]], [0.5, p["bg"]], [1, p["accent"]]],
            text=corr.round(2).values, texttemplate="%{text}", textfont=dict(size=11)))
        fig.update_yaxes(autorange="reversed")
        ui.chart(fig, 520)
        st.markdown("#### Strongest relationships")
        ui.table(_top_pairs(corr).round(3), hide_index=True)
