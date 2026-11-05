import pandas as pd
import streamlit as st

from kikolens.dashboard import components as ui
from kikolens.dashboard import i18n
from kikolens.learn import glossary as gl

TITLE = "Glossary"
SUBTITLE = "What the technical words mean, with what to watch out for. Search for a word or browse the list."


def render(df: pd.DataFrame) -> None:
    lang = i18n.current()
    query = st.text_input("Search", key="gl_query", placeholder="p-value, power, overfitting...")
    table = gl.search(query, lang)
    if not len(table):
        ui.note("Nothing matches that. Try a shorter word.", "low")
        return
    st.caption(f"{len(table)} " + i18n.t("terms"))
    for _, row in table.iterrows():
        entry = gl.get(row["key"], lang)
        with st.expander(entry["term"], expanded=len(table) == 1):
            st.write(entry["definition"])
            st.markdown("**" + i18n.t("Watch out") + ":** " + entry["watch_out"])
            if entry["see_also"]:
                st.caption(i18n.t("See also") + ": " + ", ".join(x["term"] for x in entry["see_also"]))
