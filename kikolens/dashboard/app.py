import html
import os
from urllib.parse import quote

import streamlit as st

from kikolens.dashboard import components as ui
from kikolens.dashboard import state, theme
from kikolens.dashboard.views import (anomalies, ask, clusters, compare, dimensions, experiments, explore, lab, models, monitoring,
                                      neural, overview, patterns, predict, project, quality, panel, report, statistics,
                                      text, timeseries, transform)
from kikolens.utils.loaders import load_data

st.set_page_config(page_title="KikoLens", page_icon=None, layout="wide", initial_sidebar_state="expanded")

# Las secciones se agrupan por áreas para que la barra lateral no sea una lista interminable.
GROUPS = {
    "Data": {"Overview": overview, "Explore": explore, "Quality and cleaning": quality, "Transform": transform,
             "Patterns": patterns, "Statistics": statistics, "Ask": ask},
    "Analysis": {"Experiments": experiments, "Anomalies": anomalies, "Dimensions": dimensions,
                 "Compare data": compare, "Segments": clusters,
                 "Text": text},
    "Models": {"Models": models, "Model lab": lab, "Neural network": neural, "Time series": timeseries,
               "Predict": predict, "Monitoring": monitoring},
    "Share": {"Panel": panel, "Report": report, "Project": project},
}
PAGES = {name: view for pages in GROUPS.values() for name, view in pages.items()}
GROUP_OF = {name: group for group, pages in GROUPS.items() for name in pages}
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


def _go_to_group() -> None:
    st.session_state.nav = next(iter(GROUPS[st.session_state.nav_group]))


def _navigation() -> None:
    """Primero el área y debajo sus secciones. Si se abre una sección de otra área, el área se ajusta sola."""
    current = st.session_state.get("nav")
    if current in GROUP_OF:
        st.session_state["nav_group"] = GROUP_OF[current]
    elif "nav_group" not in st.session_state:
        st.session_state["nav_group"] = "Data"
    ui.label("Area", sidebar=True)
    group = st.sidebar.radio("Area", list(GROUPS), key="nav_group", horizontal=True, label_visibility="collapsed",
                             on_change=_go_to_group)
    if st.session_state.get("nav") not in GROUPS[group]:
        st.session_state["nav"] = next(iter(GROUPS[group]))
    ui.label("Sections", sidebar=True)
    st.sidebar.radio("Section", list(GROUPS[group]), key="nav", label_visibility="collapsed")


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

    _navigation()
    page = st.session_state.nav
    st.sidebar.markdown(
        f'<div class="kl-file">{html.escape(os.path.basename(str(st.session_state.filename)))}<br>'
        f"{len(df):,} rows · {len(df.columns)} columns</div>",
        unsafe_allow_html=True,
    )

    view = PAGES[page]
    ui.header(view.TITLE, view.SUBTITLE)
    view.render(df)


main()
