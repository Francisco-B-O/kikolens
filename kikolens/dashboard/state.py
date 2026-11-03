"""Estado de la sesión: dataset de trabajo, historial de limpieza y resultados de modelos."""
import streamlit as st

from kikolens.pipeline.recipe import Recipe, apply_operation

MODEL_KEYS = ("automl_res", "nn_res", "lab_res", "ts_res", "cluster_res", "chat_history", "pd_result")
MAX_HISTORY = 20


def init() -> None:
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("original_df", None)
    st.session_state.setdefault("filename", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("clean_log", [])
    st.session_state.setdefault("recipe", Recipe())
    st.session_state.setdefault("datasets", {})
    st.session_state.setdefault("notes", "")


def set_dataset(df, name: str) -> None:
    st.session_state.df = df
    st.session_state.original_df = df.copy()
    st.session_state.filename = name
    st.session_state.history = []
    st.session_state.clean_log = []
    st.session_state.recipe = Recipe()
    st.session_state.datasets = {}
    st.session_state.notes = ""
    invalidate_models()


def invalidate_models() -> None:
    for key in MODEL_KEYS:
        st.session_state.pop(key, None)


def apply_change(new_df, message: str, op: str | None = None, params: dict | None = None) -> None:
    """Sustituye el dataset guardando el anterior para poder deshacer. Si se indica `op`, el paso queda en la receta."""
    _commit(new_df, message, [(op, params or {}, message)] if op else [])


def _commit(new_df, message: str, steps) -> None:
    history = st.session_state.history
    history.append((st.session_state.df, message, len(steps)))
    del history[:-MAX_HISTORY]
    st.session_state.df = new_df
    st.session_state.clean_log.append(message)
    for op, params, text in steps:
        st.session_state.recipe.add(op, params, text)
    invalidate_models()


def apply_steps(new_df, steps, message: str) -> None:
    """Varios pasos reproducibles como una sola acción deshacible. `steps`: [(operación, parámetros, mensaje)]."""
    _commit(new_df, message, list(steps))


def run_operation(op: str, **params) -> str:
    """Ejecuta una operación sobre el dataset actual y la registra. Devuelve el mensaje; ValueError si falla."""
    new_df, message = apply_operation(st.session_state.df, op, datasets=st.session_state.datasets, **params)
    apply_change(new_df, message, op, params)
    return message


def can_undo() -> bool:
    return bool(st.session_state.history)


def undo() -> None:
    if not can_undo():
        return
    previous, _, n_steps = st.session_state.history.pop()
    st.session_state.df = previous
    if st.session_state.clean_log:
        st.session_state.clean_log.pop()
    for _ in range(n_steps):
        st.session_state.recipe.pop()
    invalidate_models()


def reset() -> None:
    if st.session_state.original_df is not None:
        st.session_state.df = st.session_state.original_df.copy()
        st.session_state.history = []
        st.session_state.clean_log = []
        st.session_state.recipe = Recipe()
        invalidate_models()
