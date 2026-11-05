import pandas as pd
import streamlit as st

from kikolens.analyze import panel as pn
from kikolens.analyze import transform as tf
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

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


def _selection(state) -> dict | None:
    """Lo marcado en un gráfico, sea cual sea la forma en que Streamlit entregue el estado."""
    try:
        selection = state["selection"]
        return {"points": list(selection["points"] or []), "box": list(selection["box"] or [])}
    except (KeyError, TypeError, AttributeError):
        return None


def _sync_cross(df: pd.DataFrame, panel: pn.Panel) -> dict:
    """Recoge las selecciones hechas en los gráficos. Una selección vacía no borra la anterior (al cambiar los demás gráficos se puede perder el
    estado de la selección), se borra con el botón; así el filtro no desaparece solo."""
    cross = st.session_state.setdefault("pn_cross", {})
    ids = {t["id"] for t in panel.tiles}
    for gone in [k for k in cross if k not in ids]:
        cross.pop(gone)
    for tile in panel.tiles:
        if tile["type"] != "chart":
            continue
        filters = pn.selection_filters(tile, _selection(st.session_state.get(f"pn_fig_{tile['id']}")), df)
        if filters:
            cross[tile["id"]] = filters
    return cross


def _cross_bar(panel: pn.Panel, cross: dict) -> None:
    if not cross:
        return
    titles = {t["id"]: t.get("title") or "chart" for t in panel.tiles}
    st.caption(t("Selected in the charts:") + " " + " | ".join(f"{titles.get(k, 'chart')}: {pn.describe_filters(v)}" for k, v in cross.items()))
    if st.button("Clear the selections", key="pn_clear_cross"):
        for tile in panel.tiles:
            st.session_state.pop(f"pn_fig_{tile['id']}", None)
        st.session_state["pn_cross"] = {}
        st.rerun()


def _render_tile(tile: dict, value, error, panel: pn.Panel) -> None:
    title = tile.get("title") or tile["type"].title()
    if error:
        ui.note(error, "high", bold=title)
    elif tile["type"] == "chart":
        ui.chart(value, 340, key=f"pn_fig_{tile['id']}", select=True)
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
    c1, c2 = st.columns([4, 1])
    panel.title = c1.text_input("Panel title", panel.title, key="pn_name")
    c2.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    if c2.button("Suggest a panel", key="pn_suggest", help="Replaces the tiles with a starting point built from your columns."):
        st.session_state.panel = pn.suggest_panel(df, panel.title)
        st.rerun()
    with st.expander("Open a saved panel definition"):
        upload = st.file_uploader("Panel definition (JSON)", type=["json"], key="pn_open")
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

    cross = _sync_cross(df, panel)
    data, results = panel.evaluate(df, cross)
    metrics = [(t.get("title") or "Metric", pn.format_metric(v)) for t, v, e in results if t["type"] == "metric" and not e]
    if metrics:
        ui.stats(metrics)
    st.caption(f"Showing {len(data):,} of {len(df):,} rows.")
    st.caption("Click bars or slices, or drag a box on a chart, to filter the other tiles.")
    _cross_bar(panel, cross)
    others = [r for r in results if r[0]["type"] != "metric" or r[2]]
    for i in range(0, len(others), 2):
        for col, (tile, value, error) in zip(st.columns(2, gap="large"), others[i:i + 2]):
            with col:
                _render_tile(tile, value, error, panel)

    st.divider()
    a, b, c, d = st.columns(4)
    a.download_button("Web page", pn.panel_html(df, panel).encode("utf-8"), "kikolens_panel.html", "text/html",
                      key="pn_html", help="One self-contained page with the charts. The data is not included.")
    b.download_button("Panel definition (JSON)", panel.to_json().encode("utf-8"), "panel.json", "application/json", key="pn_json")
    _office(df, panel, cross, c, d)


def _office(df: pd.DataFrame, panel: pn.Panel, cross: dict, pptx_slot, docx_slot) -> None:
    """PowerPoint y Word se preparan al pedirlos (llevan un rato) y se recuerdan mientras no cambie el panel, los datos ni la selección."""
    try:
        from kikolens.report import office
    except ImportError:
        st.caption("PowerPoint and Word export needs  pip install kikolens[office]")
        return
    signature = (panel.to_json(), repr(sorted(cross.items())), df.shape, st.session_state.get("filename"))
    for slot, key, make, name, mime in (
            (pptx_slot, "pptx", office.panel_pptx, "kikolens_panel.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
            (docx_slot, "docx", office.panel_docx, "kikolens_panel.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")):
        ready = st.session_state.get(f"pn_{key}_ready")
        if ready and ready[0] == signature:
            slot.download_button("Download PowerPoint" if key == "pptx" else "Download Word", ready[1], name, mime, key=f"pn_{key}",
                                 help="Native, editable charts. Includes the current selection." if key == "pptx" else "Charts as images. Includes the current selection.")
        elif slot.button("Prepare PowerPoint" if key == "pptx" else "Prepare Word", key=f"pn_{key}_make"):
            try:
                st.session_state[f"pn_{key}_ready"] = (signature, make(df, panel, cross))
            except (ImportError, ValueError) as e:
                ui.note(str(e), "high")
            else:
                st.rerun()
