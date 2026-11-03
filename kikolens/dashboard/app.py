import html
import os
from urllib.parse import quote

import streamlit as st

from kikolens.dashboard import components as ui
from kikolens.dashboard import state, theme
from kikolens.dashboard.views import (ask, clusters, explore, lab, models, neural, overview, patterns, predict, project, quality,
                                      report, statistics, timeseries, transform)
from kikolens.utils.loaders import load_data

st.set_page_config(page_title="KikoLens", page_icon=None, layout="wide", initial_sidebar_state="expanded")

PAGES = {
    "Overview": overview,
    "Explore": explore,
    "Quality and cleaning": quality,
    "Transform": transform,
    "Patterns": patterns,
    "Statistics": statistics,
    "Ask": ask,
    "Models": models,
    "Model lab": lab,
    "Neural network": neural,
    "Predict": predict,
    "Time series": timeseries,
    "Segments": clusters,
    "Report": report,
    "Project": project,
}
FILE_TYPES = ["csv", "tsv", "xlsx", "xls", "json", "ndjson", "jsonl", "parquet"]


def _database_uri() -> str | None:
    engine = st.sidebar.selectbox("Engine", ["SQLite", "PostgreSQL", "MySQL"], key="db_engine")
    if engine == "SQLite":
        path = st.sidebar.text_input("Path to the .db file", key="db_path").strip()
        return f"sqlite:///{path}" if path else None
    host = st.sidebar.text_input("Host", "localhost", key="db_host")
    port = st.sidebar.text_input("Port", "5432" if engine == "PostgreSQL" else "3306", key="db_port")
    user = st.sidebar.text_input("User", key="db_user").strip()
    password = st.sidebar.text_input("Password", type="password", key="db_pass")
    database = st.sidebar.text_input("Database", key="db_name").strip()
    if user and password and database:
        driver = "postgresql" if engine == "PostgreSQL" else "mysql+pymysql"
        return f"{driver}://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}/{database}"
    return None


def _source_picker():
    ui.label("Data", sidebar=True)
    mode = st.sidebar.selectbox("Source", ["File", "URL", "Database"], key="source_mode", label_visibility="collapsed")
    if mode == "File":
        upload = st.sidebar.file_uploader("File", type=FILE_TYPES, key="upload", label_visibility="collapsed")
        return upload, getattr(upload, "name", None)
    if mode == "URL":
        url = st.sidebar.text_input("Link to a CSV, JSON, Excel or Parquet file", key="source_url").strip()
        return (url or None), (url.rsplit("/", 1)[-1] if url else None)
    uri = _database_uri()
    return uri, ("database" if uri else None)


def _load_if_new(source, name) -> bool:
    """Carga la fuente cuando cambia. Devuelve True si hay que redibujar."""
    if source is None or st.session_state.filename == name:
        return False
    try:
        with st.spinner("Loading..."):
            df = load_data(source)
    except Exception as e:
        st.sidebar.error(str(e))
        return False
    state.set_dataset(df, name)
    return True


def _welcome() -> None:
    st.markdown(
        """<div class="kl-empty"><h1>KikoLens</h1>
        <p>Analyze, clean, model and forecast your data without it leaving this computer.</p>
        <ol class="kl-steps"><li>Choose a file, a link or a database in the sidebar.</li>
        <li>Check the quality of the data and fix what is wrong.</li>
        <li>Explore it, train a model or a neural network, and export a report.</li></ol></div>""",
        unsafe_allow_html=True,
    )


def main() -> None:
    state.init()
    theme.inject()

    st.sidebar.markdown('<div class="kl-brand">KikoLens<small>Private data analysis</small></div>', unsafe_allow_html=True)
    source, name = _source_picker()
    if _load_if_new(source, name):
        st.rerun()

    df = st.session_state.df
    if df is None:
        _welcome()
        return

    ui.label("Sections", sidebar=True)
    page = st.sidebar.radio("Section", list(PAGES), key="nav", label_visibility="collapsed")
    st.sidebar.markdown(
        f'<div class="kl-file">{html.escape(os.path.basename(str(st.session_state.filename)))}<br>'
        f"{len(df):,} rows · {len(df.columns)} columns</div>",
        unsafe_allow_html=True,
    )

    view = PAGES[page]
    ui.header(view.TITLE, view.SUBTITLE)
    view.render(df)


main()
