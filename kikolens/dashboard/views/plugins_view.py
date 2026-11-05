import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kikolens import plugins
from kikolens.config import home_dir
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Plugins"
SUBTITLE = "Analyses and file formats added by your own Python files. They run on this computer with your permissions."


def _field(df: pd.DataFrame, analysis: plugins.Analysis, key: str, spec: dict):
    """Un control por parámetro, según su tipo. Devuelve el valor (None si se dejó sin elegir)."""
    kind, label, wid = spec["type"], spec["label"], f"pg_{analysis.name}_{key}"
    default = spec.get("default")
    if kind == "column":
        options = list(df.columns) if default is None else ["(default)"] + list(df.columns)
        value = st.selectbox(label, options, key=wid)
        return None if value == "(default)" else value
    if kind == "columns":
        return st.multiselect(label, list(df.columns), default=default or [], key=wid) or None
    if kind in ("number", "integer"):
        return st.number_input(label, value=float(default if default is not None else spec.get("min", 0.0)), key=wid, step=1.0 if kind == "integer" else None,
                               min_value=float(spec["min"]) if "min" in spec else None, max_value=float(spec["max"]) if "max" in spec else None)
    if kind == "choice":
        choices = list(spec["choices"])
        return st.selectbox(label, choices, index=choices.index(default) if default in choices else 0, key=wid)
    if kind == "bool":
        return st.checkbox(label, bool(default), key=wid)
    return st.text_input(label, "" if default is None else str(default), key=wid)


def _show(res: dict) -> None:
    ui.note(str(res["summary"]), "ok")
    for note in res.get("notes", []) or []:
        ui.note(str(note), "medium")
    for value in res.values():
        if isinstance(value, go.Figure):
            ui.chart(value, 380)
    for key, value in res.items():
        if isinstance(value, pd.DataFrame):
            if key != "table":
                st.caption(str(key))
            ui.table(value.head(500))
    scalars = [(k, v) for k, v in res.items() if k != "summary" and isinstance(v, (int, float, str, bool)) and not isinstance(v, bool)]
    if scalars:
        ui.stats([(str(k), f"{v:,.4g}" if isinstance(v, float) else str(v)) for k, v in scalars[:6]])


def render(df: pd.DataFrame) -> None:
    if not plugins.enabled():
        ui.note("Plugins are turned off (KIKOLENS_PLUGINS=0).", "low")
        return
    found = plugins.analyses()
    for message in plugins.errors():
        ui.note(message, "high", bold="Could not load")
    if not found:
        ui.note(t("No plugins found. Put a .py file with a register(registry) function in the folder below or list paths under [plugins] in kikolens.toml; see the plugins guide."), "low")
        st.code(str(home_dir() / "plugins"), language=None)
        return
    st.caption(t("Only use plugins you wrote or have read: loading one runs its code."))
    name = st.selectbox("Analysis", list(found), format_func=lambda n: f"{n}: {found[n].description}" if found[n].description else n, key="pg_name")
    analysis = found[name]
    values = {}
    columns = st.columns(2) if len(analysis.params) > 1 else [st.container()]
    for i, (key, spec) in enumerate(analysis.params.items()):
        with columns[i % len(columns)]:
            values[key] = _field(df, analysis, key, spec)
    if st.button("Run", key="pg_run", type="primary"):
        try:
            st.session_state["pg_res"] = (name, plugins.run(name, df, **{k: v for k, v in values.items() if v is not None}))
        except ValueError as e:
            st.session_state.pop("pg_res", None)
            ui.note(str(e), "high")
    cached = st.session_state.get("pg_res")
    if cached and cached[0] == name:
        _show(cached[1])
