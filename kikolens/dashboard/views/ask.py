import pandas as pd
import streamlit as st

from kikolens.dashboard import components as ui
from kikolens.sql.generator import generate_sql_from_text
from kikolens.sql.runner import run_sql_query

TITLE = "Ask"
SUBTITLE = "Question the data in plain language, or write SQL yourself. Read-only."

EXAMPLES = ["How many rows are there?", "Show the 10 rows with the highest values", "Count rows per category"]


def _execute(df: pd.DataFrame, sql: str) -> None:
    try:
        result = run_sql_query(df, sql)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    st.caption(f"{len(result):,} row(s)")
    ui.table(result)
    ui.download_csv(result, "query_result.csv", "Download result (CSV)")


def render(df: pd.DataFrame) -> None:
    history = st.session_state.setdefault("chat_history", [])
    natural, manual = st.tabs(["Plain language", "SQL"])

    with natural:
        question = st.text_input("Your question", placeholder=EXAMPLES[1], key="ask_q")
        if st.button("Ask", key="ask_go", type="primary") and question.strip():
            with st.spinner("Writing the query..."):
                sql = generate_sql_from_text(df, question)
            if sql.startswith("-- Error"):
                ui.note(sql.removeprefix("-- "), "high", detail="Is Ollama running and the model downloaded?")
            else:
                history.append({"question": question, "sql": sql})
                st.session_state.ask_sql = sql
                st.session_state.ask_sql_edit = sql
        if st.session_state.get("ask_sql"):
            sql = st.text_area("Generated SQL (editable)", height=120, key="ask_sql_edit")
            _execute(df, sql)

    with manual:
        st.caption("The table is called `df`.")
        sql = st.text_area("Query", "SELECT * FROM df LIMIT 20", height=140, key="sql_manual")
        if st.button("Run", key="sql_run", type="primary"):
            _execute(df, sql)

    if history:
        with st.expander(f"Previous questions ({len(history)})"):
            for item in reversed(history[-10:]):
                st.markdown(f"**{item['question']}**")
                st.code(item["sql"], language="sql")
