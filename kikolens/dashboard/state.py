"""Estado de la sesión: dataset de trabajo, historial de limpieza y resultados de modelos."""
import streamlit as st

MODEL_KEYS = ("automl_res", "nn_res", "ts_res", "cluster_res", "chat_history")
MAX_HISTORY = 20


def init() -> None:
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("original_df", None)
    st.session_state.setdefault("filename", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("clean_log", [])


def set_dataset(df, name: str) -> None:
    st.session_state.df = df
    st.session_state.original_df = df.copy()
    st.session_state.filename = name
    st.session_state.history = []
    st.session_state.clean_log = []
    invalidate_models()


def invalidate_models() -> None:
    for key in MODEL_KEYS:
        st.session_state.pop(key, None)


def apply_change(new_df, message: str) -> None:
    """Sustituye el dataset guardando el anterior para poder deshacer."""
    history = st.session_state.history
    history.append((st.session_state.df, message))
    del history[:-MAX_HISTORY]
    st.session_state.df = new_df
    st.session_state.clean_log.append(message)
    invalidate_models()


def can_undo() -> bool:
    return bool(st.session_state.history)


def undo() -> None:
    if not can_undo():
        return
    previous, _ = st.session_state.history.pop()
    st.session_state.df = previous
    if st.session_state.clean_log:
        st.session_state.clean_log.pop()
    invalidate_models()


def reset() -> None:
    if st.session_state.original_df is not None:
        st.session_state.df = st.session_state.original_df.copy()
        st.session_state.history = []
        st.session_state.clean_log = []
        invalidate_models()
