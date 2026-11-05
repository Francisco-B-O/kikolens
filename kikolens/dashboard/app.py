import html
import os
from urllib.parse import quote

import streamlit as st

from kikolens.dashboard import components as ui
from kikolens import config, demo
from kikolens.dashboard import i18n, state, theme
from kikolens.dashboard.views import (analyst, anomalies, ask, autopilot, causal_effect, clusters, compare, customers, dimensions, distribution, drivers, experiments, explore, export,
                                      features, gallery, health, history, home, impact, lab, linkage, marketing, models, monitoring, survival, multivariate, recommender, segment_explorer, selection, sequential, synthetic,
                                      neural, overview, patterns, predict, privacy, project, quality, panel, regression, report, simulation, statistics,
                                      story, text, timeseries, transform)
from kikolens.utils.loaders import list_sheets, list_tables, load_data, load_text

st.set_page_config(page_title="KikoLens", page_icon=None, layout="wide", initial_sidebar_state="expanded")

# Las secciones se agrupan por áreas para que la barra lateral no sea una lista interminable.
GROUPS = {
    "Data": {"Home": home, "Overview": overview, "Data story": story, "Autopilot": autopilot, "Explore": explore, "Chart gallery": gallery, "Quality and cleaning": quality, "Privacy": privacy, "Transform": transform,
             "Match records": linkage,
             "Patterns": patterns, "Statistics": statistics, "Ask": ask, "Analyst": analyst},
    "Analysis": {"Experiments": experiments, "Anomalies": anomalies, "Dimensions": dimensions, "Distribution fit": distribution,
                 "Compare data": compare, "Segments": clusters,
                 "Text": text, "Survival": survival, "Customers": customers, "Impact": impact, "Causal effect": causal_effect, "Simulation": simulation, "Marketing mix": marketing, "Sequential tests": sequential, "Segment explorer": segment_explorer,
                 "Recommendations": recommender,
                 "Why it changed": drivers},
    "Models": {"Models": models, "Model lab": lab, "Feature ideas": features, "Feature selection": selection, "Forecast with drivers": multivariate, "Run history": history, "Regression": regression, "Neural network": neural, "Time series": timeseries,
               "Predict": predict, "Model health": health, "Monitoring": monitoring},
    "Share": {"Panel": panel, "Report": report, "Export": export, "Synthetic data": synthetic, "Project": project},
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
    mode = st.sidebar.selectbox("Source", ["File", "URL", "Paste", "Database"], key="source_mode", label_visibility="collapsed",
                                format_func=i18n.t)
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


def _load_demo(name: str) -> None:
    state.set_dataset(demo.load_demo(name), f"{name} (example).csv")


def _welcome() -> None:
    t = i18n.t
    steps = ("Choose a file, a link or a database in the sidebar.", "Check the quality of the data and fix what is wrong.",
             "Explore it, train a model or a neural network, and export a report.")
    st.markdown(
        '<div class="kl-empty"><h1>KikoLens</h1>'
        f'<p>{html.escape(t("Analyze, clean, model and forecast your data without it leaving this computer."))}</p>'
        f'<ol class="kl-steps">{"".join(f"<li>{html.escape(t(step))}</li>" for step in steps)}</ol></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div style="text-align:center;margin-top:1.2rem" class="kl-label">{html.escape(t("Or try an example"))}</div>', unsafe_allow_html=True)
    columns = st.columns(len(demo.GENERATORS))
    for column, name in zip(columns, demo.GENERATORS):
        column.button(t(name.capitalize()), key=f"demo_{name}", on_click=_load_demo, args=(name,), width="stretch",
                      help=t(demo.DESCRIPTIONS[name]))


def _go_to_group() -> None:
    st.session_state.nav = next(iter(GROUPS[st.session_state.nav_group]))


def _jump() -> None:
    """Salta a la sección elegida en el buscador."""
    target = st.session_state.get("jump")
    if target in PAGES:
        st.session_state["nav"] = target
        st.session_state["nav_group"] = GROUP_OF[target]
    st.session_state["jump"] = ""


def _navigation() -> None:
    """Primero el área y debajo sus secciones. Si se abre una sección de otra área, el área se ajusta sola."""
    current = st.session_state.get("nav")
    if current in GROUP_OF:
        st.session_state["nav_group"] = GROUP_OF[current]
    elif "nav_group" not in st.session_state:
        st.session_state["nav_group"] = "Data"
    ui.label("Go to", sidebar=True)
    st.sidebar.selectbox("Go to", [""] + sorted(PAGES, key=str.lower), key="jump", on_change=_jump, label_visibility="collapsed",
                         format_func=lambda name: i18n.t("Type to find a section") if not name else i18n.t(name))
    ui.label("Area", sidebar=True)
    group = st.sidebar.radio("Area", list(GROUPS), key="nav_group", horizontal=True, label_visibility="collapsed",
                             on_change=_go_to_group, format_func=i18n.t)
    if st.session_state.get("nav") not in GROUPS[group]:
        st.session_state["nav"] = next(iter(GROUPS[group]))
    ui.label("Sections", sidebar=True)
    st.sidebar.radio("Section", list(GROUPS[group]), key="nav", label_visibility="collapsed", format_func=i18n.t)


def _language_picker() -> None:
    """Al final de la barra lateral. Se lee el valor de la ejecución anterior para traducir desde el principio."""
    names = list(i18n.LANGUAGES)
    st.session_state.setdefault("lang_choice", next(n for n, code in i18n.LANGUAGES.items() if code == config.default_language()))
    choice = st.sidebar.selectbox("Language / Idioma", names, key="lang_choice")
    st.session_state["lang"] = i18n.LANGUAGES[choice]


def main() -> None:
    state.init()
    st.session_state["lang"] = i18n.LANGUAGES.get(st.session_state.get("lang_choice") or next(
        n for n, code in i18n.LANGUAGES.items() if code == config.default_language()), "en")
    i18n.install()
    theme.inject()

    st.sidebar.markdown(f'<div class="kl-brand">KikoLens<small>{html.escape(i18n.t("Private data analysis"))}</small></div>',
                        unsafe_allow_html=True)
    source, name, options = _source_picker()
    if _load_if_new(source, name, options):
        st.rerun()

    df = st.session_state.df
    if df is None:
        _welcome()
        _language_picker()
        return

    _navigation()
    page = st.session_state.nav
    st.sidebar.markdown(
        f'<div class="kl-file">{html.escape(os.path.basename(str(st.session_state.filename)))}<br>'
        f"{len(df):,} rows · {len(df.columns)} columns</div>",
        unsafe_allow_html=True,
    )

    _language_picker()
    view = PAGES[page]
    ui.header(view.TITLE, view.SUBTITLE)
    view.render(df)


main()
