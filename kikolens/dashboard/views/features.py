import pandas as pd
import streamlit as st

from kikolens.analyze import autofeatures as af
from kikolens.dashboard import components as ui
from kikolens.dashboard import state

TITLE = "Feature ideas"
SUBTITLE = "Let KikoLens try new columns (ratios, products, logs) and keep the ones that make a model better."


def render(df: pd.DataFrame) -> None:
    target = st.selectbox("Column to predict", df.columns, index=len(df.columns) - 1, key="af_t")
    c1, c2 = st.columns(2)
    keep = c1.slider("Most new columns to add", 1, 10, 4, key="af_n")
    gain = c2.slider("Smallest improvement worth keeping", 0.0, 0.1, 0.01, 0.005, key="af_g",
                     help="Cross-validated R2, or F1 for a category.")
    exclude = st.multiselect("Leave out columns", [c for c in df.columns if c != target], key="af_x")
    if st.button("Search", key="af_go", type="primary"):
        bar = st.progress(0.0, text="Trying ideas...")
        try:
            st.session_state.af_res = (target, af.suggest_features(
                df, target, drop=exclude, max_new=keep, min_gain=gain,
                progress=lambda f, name: bar.progress(min(f, 1.0), text=f"Trying {name}...")))
        except ValueError as e:
            st.session_state.pop("af_res", None)
            ui.note(str(e), "high")
        bar.empty()
    cached = st.session_state.get("af_res")
    if not cached or cached[0] != target:
        return
    res = cached[1]
    ui.note(res["summary"], "ok" if res["suggestions"] else "low")
    if not res["suggestions"]:
        return
    table = pd.DataFrame(res["suggestions"])[["name", "expression", "kind", "gain", "score_after"]].round(4)
    ui.table(table, hide_index=True)
    if st.button("Add these columns to my data", key="af_apply",
                 help="They become steps in the recipe, so you can undo them or repeat them on new data."):
        new, steps = af.apply_suggestions(df, res["suggestions"])
        state.apply_steps(new, steps, "Feature ideas")
        st.session_state.pop("af_res", None)
        st.rerun()
