import pandas as pd
import streamlit as st

from kikolens.analyze.dictionary import build_dictionary, to_markdown
from kikolens.dashboard import components as ui
from kikolens.report.notebook import build_notebook
from kikolens.report.workbook import build_analysis_workbook
from kikolens.utils import writers as wr

TITLE = "Export"
SUBTITLE = "Take the data and the results out: files, a formatted Excel workbook or a database table."


def _name() -> str:
    import os

    return os.path.splitext(os.path.basename(str(st.session_state.filename or "data")))[0] or "data"


def _files(df: pd.DataFrame) -> None:
    fmt = st.radio("Format", [".csv", ".xlsx", ".parquet", ".json", ".ndjson", ".tsv", ".feather"], horizontal=True, key="ex_fmt",
                   format_func=lambda f: wr.FORMATS[f])
    if fmt == ".xlsx" and len(df) > wr.EXCEL_MAX_ROWS:
        ui.note(f"Excel holds at most {wr.EXCEL_MAX_ROWS:,} rows. Pick Parquet or CSV.", "high")
        return
    try:
        blob = wr.frame_to_bytes(df, fmt)
    except Exception as e:
        ui.note(str(e), "high")
        return
    st.download_button(f"Download {len(df):,} rows as {wr.FORMATS[fmt]}", blob, f"{_name()}{fmt}", key="ex_dl_file")


def _workbook(df: pd.DataFrame) -> None:
    st.caption("One Excel file with a sheet for each part of your work. Text that starts with = is stored as text, never as a formula.")
    c1, c2 = st.columns(2)
    include_data = c1.checkbox("Include the data", True, key="ex_wb_data")
    include_models = c2.checkbox("Include model metrics", True, key="ex_wb_models")
    models = {}
    if include_models:
        for key, label in (("automl_res", "Classical model"), ("lab_res", "Model lab"), ("nn_res", "Neural network")):
            if st.session_state.get(key):
                models[label] = st.session_state[key]
    rules = st.session_state.get("rules")
    rules_result = rules.validate(df) if rules is not None and rules.rules else None
    predictions = None
    cached = st.session_state.get("pd_result")
    if cached:
        predictions = cached[0]
    parts = ["data" if include_data else None, "recipe" if len(st.session_state.recipe) else None,
             "rules" if rules_result else None, f"{len(models)} model(s)" if models else None,
             "predictions" if predictions is not None else None]
    st.caption("Sheets: quality profile, statistics, issues" + "".join(f", {p}" for p in parts if p))
    try:
        blob = build_analysis_workbook(df, _name(), st.session_state.recipe, rules_result, models, predictions, include_data)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    st.download_button("Download the workbook", blob, f"{_name()}_analysis.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="ex_dl_wb")


def _database(df: pd.DataFrame) -> None:
    st.caption("Writes the current data as a table. It never overwrites a table unless you choose to.")
    uri = st.text_input("Database address", placeholder="sqlite:///results.db  or  postgresql://user:password@host/db",
                        type="password", key="ex_uri")
    c1, c2 = st.columns(2)
    table = c1.text_input("Table name", _name().replace("-", "_").replace(" ", "_").lower(), key="ex_table")
    mode = c2.radio("If the table exists", list(wr.IF_EXISTS), horizontal=True, key="ex_mode",
                    format_func=lambda m: {"fail": "Stop", "append": "Add rows", "replace": "Replace it"}[m])
    confirmed = True
    if mode == "replace":
        confirmed = st.checkbox("I understand that the existing table will be deleted", key="ex_confirm")
    if st.button("Write to the database", key="ex_db_go", type="primary", disabled=not uri or not confirmed):
        try:
            n = wr.write_database(df, uri.strip(), table.strip(), mode)
        except ValueError as e:
            ui.note(str(e), "high")
        else:
            ui.note(f"{n:,} rows written to '{table}'.", "ok")


def _dictionary(df: pd.DataFrame) -> None:
    st.caption("One entry per column: what it holds, how many values are empty, typical values, ranges and warnings. "
               "Write the descriptions yourself; the rest is calculated.")
    table = build_dictionary(df, st.session_state.get("dict_descriptions", {}))
    shown = table[["column", "description", "meaning", "empty %", "distinct", "typical values", "notes"]]
    edited = st.data_editor(shown, disabled=[c for c in shown.columns if c != "description"], hide_index=True, width="stretch", key="ex_dict")
    st.session_state.dict_descriptions = {r["column"]: r["description"] for _, r in edited.iterrows() if r["description"]}
    full = build_dictionary(df, st.session_state.dict_descriptions)
    left, right = st.columns(2)
    left.download_button("Download as Markdown", to_markdown(full, _name(), len(df)).encode("utf-8"), f"{_name()}_dictionary.md",
                         "text/markdown", key="ex_dl_md")
    right.download_button("Download as CSV", full.to_csv(index=False).encode("utf-8"), f"{_name()}_dictionary.csv", "text/csv",
                          key="ex_dl_dict_csv")


def _notebook(df: pd.DataFrame) -> None:
    st.caption("A Jupyter notebook that reloads the file, repeats every step recorded in this session and, if you wish, "
               "trains a model. Set the path to the file in its first data cell if it is not next to the notebook.")
    target = st.selectbox("Add a model for", ["(none)"] + list(df.columns), key="ex_nb_target")
    exclude = []
    if target != "(none)":
        exclude = st.multiselect("Leave out of the model", [c for c in df.columns if c != target], key="ex_nb_drop")
    text = build_notebook(str(st.session_state.filename or "data.csv"), st.session_state.recipe,
                          None if target == "(none)" else target, exclude, title=f"Analysis of {_name()}")
    st.download_button("Download the notebook", text.encode("utf-8"), f"{_name()}.ipynb", "application/x-ipynb+json", key="ex_dl_nb")


def render(df: pd.DataFrame) -> None:
    ui.stats([("Rows", f"{len(df):,}"), ("Columns", len(df.columns)), ("Steps recorded", len(st.session_state.recipe))])
    tabs = st.tabs(["Files", "Excel workbook", "Database", "Data dictionary", "Notebook"])
    with tabs[0]:
        _files(df)
    with tabs[1]:
        _workbook(df)
    with tabs[2]:
        _database(df)
    with tabs[3]:
        _dictionary(df)
    with tabs[4]:
        _notebook(df)
