import time

import pandas as pd
import streamlit as st

from kikolens.ai import analyst as an
from kikolens.ai.client import LLM
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Analyst"
SUBTITLE = "Ask in your own words: it draws charts, runs queries and chains analyses, showing every step. The model runs on this computer."


def _llm() -> LLM:
    """El modelo de la sesión (en las pruebas se puede dejar uno ya preparado en `analyst_llm`)."""
    if st.session_state.get("analyst_llm") is None:
        st.session_state["analyst_llm"] = LLM()
    return st.session_state["analyst_llm"]


def _available(llm: LLM) -> bool:
    """¿Responde Ollama? Se recuerda unos segundos para no preguntar en cada interacción."""
    cached = st.session_state.get("analyst_ok")
    if cached and time.time() - cached[0] < 20:
        return cached[1]
    ok = llm.available()
    st.session_state["analyst_ok"] = (time.time(), ok)
    return ok


def _bot(df: pd.DataFrame, samples: bool) -> an.Analyst:
    key = (st.session_state.get("filename"), df.shape, samples)
    current = st.session_state.get("analyst_bot")
    if current is None or current[0] != key:
        st.session_state["analyst_bot"] = (key, an.Analyst(df, _llm(), samples))
    return st.session_state["analyst_bot"][1]


def _suggestions(df: pd.DataFrame) -> list:
    numbers = list(df.select_dtypes("number").columns)
    cats = [c for c in df.columns if 2 <= df[c].nunique() <= 20 and c not in numbers]
    out = []
    if numbers and cats:
        out.append(f"Show the total of {numbers[-1]} by {cats[0]}")
    if numbers:
        out.append(f"What drives {numbers[-1]}?")
    out += ["What stands out in this data?", "Are there any unusual rows?"]
    return out[:4]


def _show(turn: an.Turn, index: int) -> None:
    with st.chat_message("user"):
        st.markdown(turn.question)
    with st.chat_message("assistant"):
        if turn.error and turn.report is None:
            ui.note(turn.text, "medium")
        else:
            st.markdown(turn.text)
        if turn.report is not None and not turn.report.used_model:
            for note in turn.report.notes:
                st.caption(note)
        if turn.sql:
            st.code(turn.sql, language="sql")
        if turn.figure is not None:
            ui.chart(turn.figure, 380, key=f"analyst_fig_{index}")
        if turn.table is not None and len(turn.table):
            ui.table(turn.table.head(200), hide_index=True)
        if turn.spec:
            with st.expander(t("How the chart was built")):
                st.json(turn.spec)
        if turn.report is not None and turn.report.steps:
            with st.expander(f"{t('Steps taken')} ({len(turn.report.steps)})"):
                for i, step in enumerate(turn.report.steps, 1):
                    st.markdown(f"**{i}. {step.tool}** `{step.args}`")
                    if step.thought:
                        st.caption(step.thought)
                    ui.note(step.summary or step.error, "" if step.summary else "high")
                left, right = st.columns(2)
                left.download_button(t("Download the trail (Markdown)"), turn.report.markdown().encode("utf-8"), f"analysis_{index}.md", "text/markdown",
                                     key=f"analyst_md_{index}")
                right.download_button(t("Download as Python"), turn.report.python(str(st.session_state.get("filename") or "data.csv")).encode("utf-8"),
                                      f"analysis_{index}.py", "text/x-python", key=f"analyst_py_{index}")


def render(df: pd.DataFrame) -> None:
    llm = _llm()
    if not _available(llm):
        ui.note("The local language model is not running, so the analyst can only choose a basic analysis from your words. "
                "Start Ollama and pull a model (ollama pull llama3.2) for the full conversation.", "medium")
    with st.expander(t("Settings")):
        samples = st.checkbox("Send sample values of the data to the model", True, key="analyst_samples",
                              help="The model runs on this computer, so nothing leaves it either way. Turn this off to send only column names, types and ranges.")
        if st.button("Clear the conversation", key="analyst_clear"):
            st.session_state.pop("analyst_bot", None)
            st.rerun()
    bot = _bot(df, samples)
    if not bot.history:
        st.caption(t("Try one of these:"))
        for i, text in enumerate(_suggestions(df)):
            if st.button(text, key=f"analyst_sug_{i}"):
                st.session_state["analyst_pending"] = text
                st.rerun()
    question = st.chat_input(t("Ask about your data"), key="analyst_input") or st.session_state.pop("analyst_pending", None)
    if question:
        with st.spinner(t("Thinking...")):
            try:
                bot.ask(question)
            except ValueError as e:
                ui.note(str(e), "high")
    for i, turn in enumerate(bot.history):
        _show(turn, i)
