import io
import os

import joblib
import pandas as pd
import streamlit as st

from kikolens.dashboard import components as ui
from kikolens.dashboard import state
from kikolens.pipeline.project import EXTENSION, load_project, save_project
from kikolens.pipeline.recipe import Recipe
from kikolens.utils.loaders import load_data

TITLE = "Project"
SUBTITLE = "Save your work, repeat it on new data and export it as code."

MODEL_SOURCES = {"automl_res": "classical", "lab_res": "lab", "nn_res": "neural"}


def _name() -> str:
    return os.path.splitext(os.path.basename(str(st.session_state.filename or "project")))[0] or "project"


def _model_blobs() -> dict:
    blobs = {}
    for key, label in MODEL_SOURCES.items():
        res = st.session_state.get(key)
        if res and "model" in res:
            buffer = io.BytesIO()
            joblib.dump({"model": res["model"], "target": res.get("target"), "task_type": res.get("task_type"),
                         "features": res.get("features") or res.get("columns")}, buffer)
            blobs[label] = buffer.getvalue()
    return blobs


def _restore(project: dict, name: str) -> None:
    original, recipe = project["original"], project["recipe"]
    state.set_dataset(original, name)
    st.session_state.datasets = project["datasets"]
    st.session_state.notes = project["notes"]
    df, log = recipe.apply(original, project["datasets"])
    st.session_state.df = df
    st.session_state.recipe = recipe
    st.session_state.clean_log = log
    st.session_state.history = [(original, "Restored project", len(recipe))] if len(recipe) else []


def _save_tab(df: pd.DataFrame) -> None:
    st.session_state.notes = st.text_area("Notes", st.session_state.notes, height=120, key="pj_notes",
                                          placeholder="What is this analysis about? What did you decide and why?")
    available = _model_blobs()
    with_models = st.checkbox("Include trained models" + (f" ({', '.join(available)})" if available else " (none yet)"),
                              False, disabled=not available, key="pj_models",
                              help="Models are stored with joblib. Only open projects with models if you trust their source.")
    if st.button("Prepare project file", key="pj_build", type="primary"):
        st.session_state.pj_blob = save_project(
            st.session_state.original_df, st.session_state.recipe, name=_name(), datasets=st.session_state.datasets,
            notes=st.session_state.notes, source=str(st.session_state.filename),
            models=available if with_models else None)
    blob = st.session_state.get("pj_blob")
    if blob:
        ui.note(f"Project ready ({len(blob) / 1e6:.1f} MB): original data, {len(st.session_state.recipe)} step(s) and notes.", "ok")
        st.download_button("Download project", blob, f"{_name()}{EXTENSION}", "application/zip", key="pj_dl")


def _open_tab() -> None:
    upload = st.file_uploader("Open a project", type=[EXTENSION.lstrip(".")], key="pj_open")
    trust = st.checkbox("Also load the models inside (only for files you trust)", False, key="pj_trust")
    if upload is not None and st.button("Open", key="pj_open_go", type="primary"):
        try:
            project = load_project(upload.getvalue(), load_models=trust)
            _restore(project, project["manifest"].get("source") or upload.name)
        except ValueError as e:
            ui.note(str(e), "high")
            return
        if project["models"]:
            st.session_state.restored_models = {k: joblib.load(io.BytesIO(v)) for k, v in project["models"].items()}
        st.rerun()


def _recipe_tab(df: pd.DataFrame) -> None:
    recipe: Recipe = st.session_state.recipe
    if not len(recipe):
        ui.note("No steps yet. Cleaning and transformations you apply are recorded here.", "low")
    else:
        ui.table(recipe.describe(), hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button("Download recipe (JSON)", recipe.to_json().encode("utf-8"), f"{_name()}_recipe.json",
                           "application/json", key="pj_rj")
        c2.download_button("Download as Python script", recipe.to_python(str(st.session_state.filename)).encode("utf-8"),
                           f"{_name()}_pipeline.py", "text/x-python", key="pj_rp")

    st.markdown("#### Repeat a recipe on another file")
    st.caption("Upload a recipe and a new file with the same columns; the cleaned result is ready to download.")
    c1, c2 = st.columns(2)
    recipe_file = c1.file_uploader("Recipe (JSON)", type=["json"], key="pj_rf")
    data_file = c2.file_uploader("New data", key="pj_df")
    if recipe_file is not None and data_file is not None and st.button("Apply recipe", key="pj_apply", type="primary"):
        try:
            loaded = Recipe.from_json(recipe_file.getvalue().decode("utf-8"))
            missing = [d for d in loaded.required_datasets if d not in st.session_state.datasets]
            if missing:
                raise ValueError(f"The recipe needs these datasets in the Combine tab first: {', '.join(missing)}.")
            result, log = loaded.apply(load_data(data_file), st.session_state.datasets)
        except (ValueError, UnicodeDecodeError) as e:
            ui.note(str(e), "high")
            return
        ui.note(f"Done: {len(result):,} rows × {len(result.columns)} columns after {len(log)} step(s).", "ok")
        ui.table(result.head(100), height=260)
        ui.download_csv(result, f"{_name()}_processed.csv", "Download result (CSV)")


def render(df: pd.DataFrame) -> None:
    ui.stats([("Dataset", os.path.basename(str(st.session_state.filename))), ("Steps recorded", len(st.session_state.recipe)),
              ("Extra datasets", len(st.session_state.datasets))])
    save_tab, open_tab, recipe_tab = st.tabs(["Save", "Open", "Recipe"])
    with save_tab:
        _save_tab(df)
    with open_tab:
        _open_tab()
    with recipe_tab:
        _recipe_tab(df)
