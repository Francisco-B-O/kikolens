import html
import os
from urllib.parse import quote

import streamlit as st

from kikolens.dashboard import components as ui
from kikolens.dashboard import state, theme
from kikolens.dashboard.views import (anomalies, ask, clusters, compare, dimensions, experiments, explore, export, lab, models, monitoring,
                                      neural, overview, patterns, predict, project, quality, panel, report, statistics,
                                      text, timeseries, transform)
from kikolens.utils.loaders import list_sheets, list_tables, load_data, load_text

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
    "Share": {"Panel": panel, "Report": report, "Export": export, "Project": project},
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
    """Devuelve (fuente, nombre, opciones de carga). El nombre cambia cuando cambia algo que obligue a recargar."""
    ui.label("Data", sidebar=True)
    mode = st.sidebar.selectbox("Source", ["File", "URL", "Paste", "Database"], key="source_mode", label_visibility="collapsed")
    if mode == "File":
        upload = st.sidebar.file_uploader("File", type=FILE_TYPES, key="upload", label_visibility="collapsed")
        options, name = {}, getattr(upload, "name", None)
        if upload is not None and name.lower().endswith((".xlsx", ".xls")):
            try:
                sheets = list_sheets(upload)
            except ValueError as e:
                st.sidebar.error(str(e))
                return None, None, {}
            if len(sheets) > 1:
                sheet = st.sidebar.selectbox("Sheet", sheets, key="xl_sheet")
                options, name = {"sheet": sheet}, f"{name} [{sheet}]"
        return upload, name, options
    if mode == "URL":
        url = st.sidebar.text_input("Link to a file or a shared Google Sheet", key="source_url",
                                    help="CSV, JSON, Excel or Parquet. s3:// and gs:// work if s3fs or gcsfs is installed.").strip()
        return (url or None), (url.rsplit("/", 1)[-1][:60] or "link" if url else None), {}
    if mode == "Paste":
        text = st.sidebar.text_area("Paste a table (with its header row)", key="paste_text", height=140)
        return (("paste", text) if text.strip() else None), (f"pasted data ({len(text)} characters)" if text.strip() else None), {}
    uri = _database_uri()
    options, name = {}, ("database" if uri else None)
    if uri:
        query = st.sidebar.text_area("Read-only query (optional)", key="db_query", height=70,
                                     help="Leave empty to pick a table.").strip()
        if query:
            options, name = {"query": query}, f"database query {abs(hash(query)) % 10**6}"
        else:
            try:
                tables = list_tables(uri)
            except Exception as e:
                st.sidebar.error(str(e))
                return None, None, {}
            if tables:
                table = st.sidebar.selectbox("Table", tables, key="db_table")
                options, name = {"table": table}, f"database: {table}"
    return uri, name, options


def _load_if_new(source, name, options=None) -> bool:
    """Carga la fuente cuando cambia. Devuelve True si hay que redibujar."""
    if source is None or st.session_state.filename == name:
        return False
    try:
        with st.spinner("Loading..."):
            if isinstance(source, tuple) and source[0] == "paste":
                df = load_text(source[1])
            else:
                df = load_data(source, **(options or {}))
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
    source, name, options = _source_picker()
    if _load_if_new(source, name, options):
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
