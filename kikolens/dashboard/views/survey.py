import pandas as pd
import streamlit as st

from kikolens.analyze import survey as sv
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Survey"
SUBTITLE = "Scale questions, reliability, NPS, cross-tabs and what matters most for the overall rating."


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


def _scale_columns(df: pd.DataFrame) -> list:
    found = []
    for c in df.columns:
        try:
            values, _ = sv.to_numeric_scale(df[c])
        except ValueError:
            continue
        s = values.dropna()
        if len(s) >= 20 and s.nunique() <= 11 and s.min() >= 0:
            found.append(c)
    return found


def _questions(df: pd.DataFrame) -> None:
    scale = _scale_columns(df)
    if not scale:
        ui.note("No scale questions were found: they are columns with a few numeric answers (1 to 5, 0 to 10) or texts such as 'Agree' and 'Strongly agree'.", "low")
        return
    items = st.multiselect("Questions", scale, default=scale[:8], key="sv_items")
    reverse = st.multiselect("Worded the other way round (reversed)", items, key="sv_rev")
    res = _run("sv_likert", (tuple(items), tuple(reverse)), lambda: sv.likert_summary(df, items, None, reverse))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    ui.chart(res["figure"], 120 + 44 * len(items))
    ui.table(res["table"].round(3))


def _reliability(df: pd.DataFrame) -> None:
    scale = _scale_columns(df)
    items = st.multiselect("Questions that measure the same thing", scale, default=scale[:5], key="sr_items")
    reverse = st.multiselect("Worded the other way round (reversed)", items, key="sr_rev")
    res = _run("sr_res", (tuple(items), tuple(reverse)), lambda: sv.reliability(df, items, reverse))
    if res is None:
        return
    ui.note(res["summary"], "ok" if res["alpha"] >= 0.7 else "medium")
    ui.stats([("Cronbach's alpha", f"{res['alpha']:.2f}"), ("95% range", f"{res['interval'][0]:.2f} to {res['interval'][1]:.2f}"), ("Average correlation", f"{res['mean_inter_item_correlation']:.2f}"),
              ("People", f"{res['n']:,}")])
    _notes(res)
    ui.table(res["table"].round(3))


def _nps(df: pd.DataFrame) -> None:
    numeric = [c for c in df.select_dtypes("number").columns if df[c].between(0, 10).all()]
    if not numeric:
        ui.note("NPS needs a column with a 0 to 10 answer to 'How likely are you to recommend us?'.", "low")
        return
    c1, c2 = st.columns(2)
    score = c1.selectbox("Recommendation (0 to 10)", numeric, key="sn_score")
    groups = [c for c in df.columns if 2 <= df[c].nunique() <= 15 and c != score]
    group = c2.selectbox("Compare groups", ["(none)"] + groups, key="sn_group")
    group = None if group.startswith("(") else group
    res = _run("sn_res", (score, group), lambda: sv.nps(df, score, group))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    ui.chart(res["figure"], 300)
    if group:
        ui.table(res["table"].round(3))


def _crosstab(df: pd.DataFrame) -> None:
    cols = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 15]
    if len(cols) < 2:
        ui.note("This needs two questions with a few different answers each.", "low")
        return
    c1, c2 = st.columns(2)
    row = c1.selectbox("Rows", cols, key="sc_row")
    column = c2.selectbox("Columns", [c for c in cols if c != row], key="sc_col")
    res = _run("sc_res", (row, column), lambda: sv.crosstab(df, row, column))
    if res is None:
        return
    ui.note(res["summary"], "ok" if res["p_value"] < 0.05 else "accent")
    _notes(res)
    st.caption(t("Share of each row"))
    ui.heat_table(res["row_percent"].round(3))
    st.caption(t("Adjusted residuals: beyond ±2 a cell has more (+) or fewer (-) answers than chance would give."))
    ui.heat_table(res["residuals"].round(2), cmap="RdBu_r")


def _drivers(df: pd.DataFrame) -> None:
    scale = _scale_columns(df)
    if len(scale) < 3:
        ui.note("This needs an overall rating and at least two questions that may explain it.", "low")
        return
    outcome = st.selectbox("Overall rating", scale, key="sk_outcome")
    items = st.multiselect("Questions that may drive it", [c for c in scale if c != outcome], default=[c for c in scale if c != outcome][:8], key="sk_items")
    res = _run("sk_res", (outcome, tuple(items)), lambda: sv.key_drivers(df, outcome, items))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    ui.chart(res["figure"], 460)
    ui.table(res["table"].round(3))


def _groups(df: pd.DataFrame) -> None:
    scale = _scale_columns(df)
    groups = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 12 and c not in scale]
    if not scale or not groups:
        ui.note("This needs scale questions and a column that splits the people into groups.", "low")
        return
    group = st.selectbox("Groups", groups, key="sg_group")
    items = st.multiselect("Questions to compare", scale, default=scale[:8], key="sg_items")
    res = _run("sg_res", (group, tuple(items)), lambda: sv.compare_groups(df, items, group))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    ui.table(res["table"].round(4))


def _quality(df: pd.DataFrame) -> None:
    scale = _scale_columns(df)
    if len(scale) < 5:
        ui.note("This needs at least five scale questions.", "low")
        return
    items = st.multiselect("Scale questions", scale, default=scale[:10], key="sq_items")
    seconds = st.selectbox("Seconds spent (optional)", ["(none)"] + list(df.select_dtypes("number").columns), key="sq_secs")
    res = _run("sq_res", (tuple(items), seconds), lambda: sv.response_quality(df, items, None if seconds.startswith("(") else seconds))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    ui.chart(res["figure"], 280)


def render(df: pd.DataFrame) -> None:
    tabs = st.tabs(["Questions", "Reliability", "NPS", "Cross-tab", "What matters most", "Between groups", "Answer quality"])
    for tab, body in zip(tabs, (_questions, _reliability, _nps, _crosstab, _drivers, _groups, _quality)):
        with tab:
            body(df)
