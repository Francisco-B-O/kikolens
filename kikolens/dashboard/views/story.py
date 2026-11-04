import html

import pandas as pd
import streamlit as st

from kikolens.analyze import story as sty
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Data story"
SUBTITLE = "Your data told as a short story: what is there, what stands out, what moves together and what to do next."


def render(df: pd.DataFrame) -> None:
    key = (st.session_state.get("filename"), df.shape)
    cached = st.session_state.get("story_res")
    if st.button(t("Tell the story"), key="story_go", type="primary") or (cached and cached[0] != key):
        with st.spinner(t("Reading the data...")):
            try:
                st.session_state.story_res = (key, sty.build_story(df))
            except ValueError as e:
                st.session_state.pop("story_res", None)
                ui.note(str(e), "high")
                return
        cached = st.session_state.story_res
    if not cached:
        st.caption(t("Press the button and KikoLens will read the data and write the story."))
        return
    story = cached[1]
    for i, chapter in enumerate(story["chapters"]):
        st.markdown(f'<div class="kl-chapter"><small>{html.escape(t(chapter["kicker"]))}</small><h3>{html.escape(chapter["title"])}</h3></div>',
                    unsafe_allow_html=True)
        st.markdown(chapter["text"])
        if chapter["stats"]:
            ui.stats(chapter["stats"])
        if chapter["figure"] is not None:
            ui.chart(chapter["figure"], 340, key=f"story_fig_{i}")
        if isinstance(chapter["table"], pd.DataFrame) and len(chapter["table"]):
            ui.table(chapter["table"], hide_index=True)
    for note in story["skipped"]:
        st.caption(f"{t('Skipped')}: {note}")
    st.download_button(t("Download the story (Markdown)"), sty.story_markdown(story, str(st.session_state.get("filename") or "Data story")).encode("utf-8"),
                       "data_story.md", "text/markdown", key="story_dl")
