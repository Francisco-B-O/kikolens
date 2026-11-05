import pandas as pd
import streamlit as st

from kikolens.analyze import synthetic as sy
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Synthetic data"
SUBTITLE = "Generate a table that looks like yours (same distributions and correlations) without being real people."


def render(df: pd.DataFrame) -> None:
    a, b = st.columns(2)
    rows = a.number_input("Rows to generate", 30, 200000, min(len(df), 2000), 100, key="sy_n")
    seed = b.number_input("Random seed", 0, 10_000, 0, key="sy_s")
    if st.button("Generate", key="sy_go", type="primary"):
        try:
            with st.spinner(t("Generating...")):
                made = sy.synthesize(df, int(rows), int(seed))
                st.session_state.sy_res = (made, sy.report(df, made["data"]))
        except ValueError as e:
            st.session_state.pop("sy_res", None)
            ui.note(str(e), "high")
    cached = st.session_state.get("sy_res")
    if not cached:
        return
    made, rep = cached
    ui.note(rep["summary"], "ok" if rep["fidelity"] >= 80 and rep["exact_copies"] < 0.05 else "medium")
    ui.stats([("Fidelity", f"{rep['fidelity']:.0f}/100"), ("Exact copies of real rows", f"{rep['exact_copies']:.1%}"),
              ("Very close to a real row", f"{rep['near_copies']:.1%}" if rep["near_copies"] is not None else "n/a"),
              ("Correlation gap", f"{rep['correlation_gap']:.3f}" if rep["correlation_gap"] is not None else "n/a")])
    if made["skipped"]:
        st.caption(f"{t('Not imitated')}: {', '.join(made['skipped'])}")
    st.markdown(f"#### {t('Column by column')}")
    ui.heat_table(rep["table"].round(4), ["difference"], hide_index=True)
    st.markdown(f"#### {t('Preview')}")
    ui.table(made["data"].head(50), hide_index=True)
    st.download_button(t("Download the synthetic data (CSV)"), made["data"].to_csv(index=False).encode("utf-8"), "synthetic.csv", "text/csv", key="sy_dl")
    with st.expander(t("Is it useful? Train a model on it")):
        numeric = [c for c in df.columns if c in made["data"].columns]
        target = st.selectbox("Column to predict", numeric, index=len(numeric) - 1, key="sy_t")
        if st.button("Compare", key="sy_u"):
            try:
                st.session_state.sy_util = sy.utility(df, made["data"], target)
            except ValueError as e:
                st.session_state.pop("sy_util", None)
                ui.note(str(e), "high")
        if st.session_state.get("sy_util"):
            ui.note(st.session_state.sy_util["summary"], "accent")
    st.caption(t("Synthetic data lowers the risk but does not remove it: check the copies above and keep the original private."))
