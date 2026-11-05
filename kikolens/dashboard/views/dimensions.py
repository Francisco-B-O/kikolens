import pandas as pd
import streamlit as st

from kikolens.analyze import dimension as dm
from kikolens.dashboard import components as ui

TITLE = "Dimensions"
SUBTITLE = "Squeeze many columns into a picture, and find the ones that say the same thing."


def _color_options(df: pd.DataFrame) -> list:
    return ["(none)"] + [c for c in df.columns if df[c].nunique() <= 12 or pd.api.types.is_numeric_dtype(df[c])]


def _pca(df: pd.DataFrame, columns: list) -> None:
    try:
        result = dm.pca_analysis(df, columns)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    ui.stats([("Columns", len(result["columns"])), ("For 90% of the variation",
                                                     f"{result['components_for_90'] or 'all'} component(s)"),
              ("First component", f"{result['explained'][0]:.0%}")])
    st.markdown(result["summary"])
    left, right = st.columns(2)
    with left:
        ui.chart(dm.scree_figure(result), 340)
    with right:
        first = result["names"][: min(2, len(result["names"]))]
        ui.chart(dm.loadings_figure(result, first), 340)
    for pc, features in result["top_features"].items():
        st.caption(f"{pc} is mostly {', '.join(features)}")
    color = st.selectbox("Color the points by", _color_options(df), key="dm_pca_color")
    names = result["names"]
    c1, c2 = st.columns(2)
    x = c1.selectbox("Horizontal", names, index=0, key="dm_px")
    y = c2.selectbox("Vertical", names, index=1, key="dm_py")
    ui.chart(dm.projection_figure(result["scores"], x, y, df[color] if color != "(none)" else None,
                                  "Rows on the main components"), 440)
    with st.expander("Loadings table"):
        ui.table(result["loadings"].round(3))


def _tsne(df: pd.DataFrame, columns: list) -> None:
    perplexity = st.slider("Perplexity", 5, 50, 30, key="dm_perp",
                           help="Roughly how many neighbours each point considers. Try several values.")
    color = st.selectbox("Color the points by", _color_options(df), key="dm_tsne_color")
    if st.button("Project", key="dm_tsne_go", type="primary"):
        try:
            with st.spinner("Projecting..."):
                st.session_state.dm_tsne = dm.tsne_embedding(df, columns, perplexity)
        except ValueError as e:
            ui.note(str(e), "high")
            return
    cached = st.session_state.get("dm_tsne")
    if not cached:
        return
    coords, note = cached
    if note:
        ui.note(note, "low")
    series = df[color].reindex(coords.index) if color != "(none)" else None
    ui.chart(dm.projection_figure(coords, "t-SNE 1", "t-SNE 2", series, "t-SNE projection"), 480)
    st.caption("Distances between far-apart clusters are not meaningful in t-SNE; look at which points stay together.")


def _redundancy(df: pd.DataFrame, columns: list) -> None:
    try:
        table = dm.vif_table(df, columns)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    bad = table[table["verdict"] != "fine"]
    ui.note(f"{len(bad)} column(s) can be mostly predicted from the others." if len(bad)
            else "No column is redundant with the rest.", "medium" if len(bad) else "ok")
    ui.table(table.round(3), hide_index=True)
    st.caption("Values above 5 mean a column is largely explained by the others; above 10 it is nearly a duplicate. "
               "Consider dropping one of each redundant group before fitting a linear model.")


def render(df: pd.DataFrame) -> None:
    usable = dm.numeric_columns(df)
    if len(usable) < 2:
        ui.note("This section needs at least two numeric columns that vary.", "low")
        return
    columns = st.multiselect("Columns", usable, default=usable[:30], key="dm_cols")
    tabs = st.tabs(["Main components (PCA)", "Map of the rows (t-SNE)", "Redundant columns"])
    with tabs[0]:
        _pca(df, columns)
    with tabs[1]:
        _tsne(df, columns)
    with tabs[2]:
        _redundancy(df, columns)
