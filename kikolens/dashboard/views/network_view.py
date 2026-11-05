import pandas as pd
import streamlit as st

from kikolens.analyze import network as nw
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Network"
SUBTITLE = "Who is important, which groups exist, how well connected it is, and what happens if the key nodes disappear."


def _run(key: str, signature, fn):
    if st.button("Analyse", key=f"{key}_go", type="primary"):
        try:
            st.session_state[key] = (signature, fn())
        except ValueError as e:
            st.session_state.pop(key, None)
            ui.note(str(e), "high")
    cached = st.session_state.get(key)
    return cached[1] if cached and cached[0] == signature else None


def _notes(res: dict) -> None:
    for note in res.get("notes", []):
        ui.note(note, "medium")


def _overview(df, source, target, weight, directed, signature) -> None:
    res = _run("nw_res", signature, lambda: nw.analyze(df, source, target, weight, directed))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    ui.stats([("Nodes", f"{res['nodes']:,}"), ("Links", f"{res['edges']:,}"), ("In the largest piece", f"{res['giant_share']:.0%}"), ("Communities", f"{len(res['communities'])}"),
              ("Typical distance", f"{res['avg_distance']:.1f}")])
    _notes(res)
    ui.chart(res["figure"], 520)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.caption(t("Most important nodes (by PageRank)"))
        ui.table(res["table"].head(50).round(4))
    with c2:
        st.caption(t("Communities"))
        ui.table(res["communities"].round(4))
    ui.download_csv(res["table"], "network_nodes.csv", "Download every node (CSV)")


def _paths(df, source, target, weight, directed) -> None:
    nodes = sorted(pd.unique(pd.concat([df[source], df[target]]).dropna().astype(str)))
    if len(nodes) < 2:
        return
    c1, c2 = st.columns(2)
    start = c1.selectbox("From", nodes, key="nw_from")
    end = c2.selectbox("To", nodes, index=min(1, len(nodes) - 1), key="nw_to")
    res = _run("nw_path", (source, target, weight, directed, start, end), lambda: nw.shortest_path(df, source, target, start, end, weight, directed))
    if res is not None:
        ui.note(res["summary"], "ok" if res["connected"] else "medium")


def _links(df, source, target, weight) -> None:
    top = int(st.slider("Suggestions", 3, 50, 10, key="nw_top"))
    res = _run("nw_links", (source, target, weight, top), lambda: nw.suggest_links(df, source, target, weight, top))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    ui.table(res["table"].round(3))


def _robustness(df, source, target, weight) -> None:
    res = _run("nw_rob", (source, target, weight), lambda: nw.robustness(df, source, target, weight))
    if res is None:
        return
    ui.note(res["summary"], "medium" if res["fragile"] else "ok")
    _notes(res)
    ui.chart(res["figure"], 380)
    ui.table(res["table"].round(3))


def render(df: pd.DataFrame) -> None:
    columns = [c for c in df.columns if not pd.api.types.is_float_dtype(df[c])]
    if len(columns) < 2:
        ui.note("A network needs two columns that name the two ends of each link, such as customer and product, or sender and receiver.", "low")
        return
    c1, c2, c3, c4 = st.columns(4)
    source = c1.selectbox("Link starts at", columns, key="nw_src")
    target = c2.selectbox("Link ends at", [c for c in columns if c != source], key="nw_tgt")
    numeric = ["(none)"] + [c for c in df.select_dtypes("number").columns if c not in (source, target)]
    weight = c3.selectbox("Strength of each link", numeric, key="nw_w")
    directed = c4.checkbox("Links have a direction", key="nw_dir")
    weight = None if weight.startswith("(") else weight
    signature = (source, target, weight, directed)
    tabs = st.tabs(["Overview", "Paths", "Missing links", "Robustness"])
    with tabs[0]:
        _overview(df, source, target, weight, directed, signature)
    with tabs[1]:
        _paths(df, source, target, weight, directed)
    with tabs[2]:
        _links(df, source, target, weight)
    with tabs[3]:
        _robustness(df, source, target, weight)
