import pandas as pd
import streamlit as st

from kikolens.analyze import panel as pn
from kikolens.analyze import transform as tf
from kikolens.dashboard import components as ui

TITLE = "Panel"
SUBTITLE = "Pin the numbers and charts you care about into one page you can export."


def _panel() -> pn.Panel:
    if st.session_state.get("panel") is None:
        st.session_state.panel = pn.Panel(st.session_state.get("filename") or "Panel")
    return st.session_state.panel


def _filters(df: pd.DataFrame, panel: pn.Panel) -> None:
    with st.expander(f"Filters for the whole panel ({len(panel.filters)})", expanded=bool(panel.filters)):
        for i, f in enumerate(panel.filters):
            c1, c2 = st.columns([6, 1])
            c1.caption(f"{f['column']} {f['op']} {f.get('value') or ''}")
            if c2.button("Remove", key=f"pn_fdel_{i}"):
                panel.filters.pop(i)
                st.rerun()
        a, b, c, d = st.columns([3, 2, 3, 1])
        col = a.selectbox("Column", df.columns, key="pn_fcol")
        op = b.selectbox("Condition", tf.FILTER_OPERATORS, key="pn_fop")
        value = None if op.startswith("is ") else c.text_input("Value", key="pn_fval")
        d.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        if d.button("Add", key="pn_fadd"):
            try:
                tf.filter_rows(df, col, op, value)
            except ValueError as e:
                ui.note(str(e), "high")
            else:
                panel.filters.append({"column": col, "op": op, "value": value})
                st.rerun()


def _new_tile(df: pd.DataFrame, panel: pn.Panel) -> None:
    nums = df.select_dtypes("number").columns.tolist()
    with st.expander("Add a tile", expanded=not panel.tiles):
        kind = st.radio("Type", pn.TILE_TYPES, horizontal=True, key="pn_type")
        title = st.text_input("Title", key="pn_title")
        tile = {"type": kind, "title": title}
        if kind == "metric":
            c1, c2 = st.columns(2)
            tile["agg"] = c1.selectbox("Calculation", pn.AGGS, index=pn.AGGS.index("count"), key="pn_magg")
            tile["column"] = c2.selectbox("Column", ["(all rows)"] + list(df.columns), key="pn_mcol")
            if tile["column"] == "(all rows)":
                tile["column"] = None
        elif kind == "chart":
            tile["kind"] = st.selectbox("Chart", pn.CHART_KINDS, key="pn_ckind")
            c1, c2, c3 = st.columns(3)
            tile["x"] = c1.selectbox("Horizontal / groups", df.columns, key="pn_cx")
            y = c2.selectbox("Vertical / value", ["(count rows)"] + nums, key="pn_cy")
            tile["y"] = None if y.startswith("(") else y
            tile["agg"] = c3.selectbox("Calculation", pn.AGGS, key="pn_cagg", disabled=tile["y"] is None)
        else:
            c1, c2 = st.columns(2)
            tile["columns"] = c1.multiselect("Columns (empty for all)", df.columns, key="pn_tcols")
            sort = c2.selectbox("Sort by", ["(none)"] + list(df.columns), key="pn_tsort")
            tile["sort"] = None if sort.startswith("(") else sort
            tile["n"] = int(st.number_input("Rows", 1, 200, 10, key="pn_tn"))
        if st.button("Add to panel", key="pn_add", type="primary"):
            try:
                panel.add(tile)
                panel.evaluate(df)
            except ValueError as e:
                ui.note(str(e), "high")
                return
            st.rerun()


def _render_tile(tile: dict, value, error, panel: pn.Panel) -> None:
    title = tile.get("title") or tile["type"].title()
    if error:
        ui.note(error, "high", bold=title)
    elif tile["type"] == "chart":
        ui.chart(value, 340, key=f"pn_fig_{tile['id']}")
    elif tile["type"] == "table":
        st.markdown(f"**{title}**")
        ui.table(value, height=260)
    a, b, c, _ = st.columns([1, 1, 1, 6])
    if a.button("Up", key=f"pn_up_{tile['id']}"):
        panel.move(tile["id"], -1)
        st.rerun()
    if b.button("Down", key=f"pn_dn_{tile['id']}"):
        panel.move(tile["id"], 1)
        st.rerun()
    if c.button("Remove", key=f"pn_rm_{tile['id']}"):
        panel.remove(tile["id"])
        st.rerun()


def render(df: pd.DataFrame) -> None:
    panel = _panel()
    c1, c2, c3 = st.columns([3, 1, 1])
    panel.title = c1.text_input("Panel title", panel.title, key="pn_name")
    c2.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    if c2.button("Suggest a panel", key="pn_suggest", help="Replaces the tiles with a starting point built from your columns."):
        st.session_state.panel = pn.suggest_panel(df, panel.title)
        st.rerun()
    upload = c3.file_uploader("Open", type=["json"], key="pn_open", label_visibility="collapsed")
    if upload is not None and st.session_state.get("pn_loaded") != upload.file_id:
        try:
            st.session_state.panel = pn.Panel.from_json(upload.getvalue().decode("utf-8"))
            st.session_state.pn_loaded = upload.file_id
        except (ValueError, UnicodeDecodeError) as e:
            ui.note(str(e), "high")
        else:
            st.rerun()

    _filters(df, panel)
    _new_tile(df, panel)
    if not panel.tiles:
        ui.note("The panel is empty. Add a tile above or press Suggest a panel.", "low")
        return

    data, results = panel.evaluate(df)
    metrics = [(t.get("title") or "Metric", pn.format_metric(v)) for t, v, e in results if t["type"] == "metric" and not e]
    if metrics:
        ui.stats(metrics)
    st.caption(f"Showing {len(data):,} of {len(df):,} rows.")
    others = [r for r in results if r[0]["type"] != "metric" or r[2]]
    for i in range(0, len(others), 2):
        for col, (tile, value, error) in zip(st.columns(2, gap="large"), others[i:i + 2]):
            with col:
                _render_tile(tile, value, error, panel)

    st.divider()
    a, b = st.columns(2)
    a.download_button("Download as a web page", pn.panel_html(df, panel).encode("utf-8"), "kikolens_panel.html", "text/html",
                      key="pn_html", help="One self-contained page with the charts. The data is not included.")
    b.download_button("Download panel definition (JSON)", panel.to_json().encode("utf-8"), "panel.json",
                      "application/json", key="pn_json")
