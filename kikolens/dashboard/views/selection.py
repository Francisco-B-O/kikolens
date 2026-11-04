import pandas as pd
import streamlit as st

from kikolens.analyze import featureselect as fs
from kikolens.dashboard import components as ui
from kikolens.dashboard import state
from kikolens.dashboard.i18n import t

TITLE = "Feature selection"
SUBTITLE = "Which columns are worth keeping to predict a target? Four methods vote and a validated curve decides how many."


def render(df: pd.DataFrame) -> None:
    target = st.selectbox("Column to predict", df.columns, index=len(df.columns) - 1, key="fs_t")
    exclude = st.multiselect("Leave out columns", [c for c in df.columns if c != target], key="fs_x")
    if st.button("Score the columns", key="fs_go", type="primary"):
        bar = st.progress(0.0, text=t("Scoring..."))
        try:
            st.session_state.fs_res = (target, fs.select_features(df, target, drop=exclude, progress=lambda f, text: bar.progress(min(f, 1.0), text=text)))
        except ValueError as e:
            st.session_state.pop("fs_res", None)
            ui.note(str(e), "high")
        bar.empty()
    cached = st.session_state.get("fs_res")
    if not cached or cached[0] != target:
        return
    res = cached[1]
    ui.note(res["summary"], "ok" if res["removable"] else "accent")
    ui.stats([(f"{res['metric']} with the recommended columns", f"{res['score_recommended']:.3f}"), (f"Best {res['metric']}", f"{res['best_score']:.3f}"),
              ("Recommended", len(res["recommended"])), ("Can be left out", len(res["removable"]))])
    ui.chart(res["figure"], 360)
    ui.heat_table(res["table"].round(4), list(fs.METHODS), hide_index=True)
    st.caption(t("Each method scores the columns differently; the consensus ranks them by their average position. The curve shows the cross-validated score using the best k columns."))
    if res["removable"] and st.button("Remove the columns I can leave out", key="fs_apply",
                                      help="Adds a recipe step that drops them. You can undo it."):
        state.run_operation("drop_columns", columns=res["removable"])
        st.session_state.pop("fs_res", None)
        st.rerun()
