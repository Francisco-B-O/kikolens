import pandas as pd
import streamlit as st

from kikolens.analyze import privacy as pv
from kikolens.dashboard import components as ui
from kikolens.dashboard import state

TITLE = "Privacy"
SUBTITLE = "Find personal data, mask or replace it, and check how easily a person could be recognised. All on this computer."

ACTION_LABELS = {"keep": "Keep", "mask": "Mask (keep the last 2 characters)", "hash": "Replace with a fingerprint", "drop": "Remove the column"}


def _scan(df: pd.DataFrame) -> None:
    found = pv.scan(df)
    if found.empty:
        ui.note("No column looks like personal data. Names of people or free text can still hide some, so check the preview.", "ok")
        return
    ui.note(f"{len(found)} column(s) look like personal data.", "medium",
            detail="Detected by their content (validated patterns such as card check digits) or by their name.")
    ui.table(found.round(2), hide_index=True)
    st.markdown("#### What to do with each")
    plan = {}
    for _, row in found.iterrows():
        plan[row["column"]] = st.selectbox(f"{row['column']} ({row['kind']})", list(ACTION_LABELS), index=0,
                                           format_func=lambda a: ACTION_LABELS[a], key=f"pv_a_{row['column']}")
    salt = st.text_input("Salt for fingerprints", "", key="pv_salt", type="password",
                         help="Equal values get equal fingerprints, so counts and joins still work. The salt is saved in the "
                              "recipe: do not share the recipe if the fingerprints must stay unlinkable.")
    chosen = {c: a for c, a in plan.items() if a != "keep"}
    if st.button("Apply", key="pv_go", type="primary", disabled=not chosen):
        try:
            new, steps = pv.anonymize(df, chosen, salt)
        except ValueError as e:
            ui.note(str(e), "high")
            return
        state.apply_steps(new, steps, "Privacy")
        st.rerun()


def _k(df: pd.DataFrame) -> None:
    st.caption("A person is easier to recognise when few others share the same age, postal code, sex or city. "
               "k is the size of the smallest group a person hides in.")
    guess = pv.suggest_quasi_identifiers(df)
    columns = st.multiselect("Columns that describe a person", list(df.columns), default=guess, key="pv_q")
    k = st.slider("Wanted k", 2, 20, 5, key="pv_k")
    if not columns:
        st.caption("Pick at least one column.")
        return
    try:
        res = pv.k_anonymity(df, columns, k)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    ui.note(res["summary"], "ok" if not res["share_below_k"] else "medium")
    ui.stats([("Smallest group", res["min_k"]), ("Rows in a group below k", f"{res['share_below_k']:.1%}"),
              ("Unique rows", f"{res['unique_rows']:,}"), ("Groups", f"{res['groups']:,}")])
    ui.table(res["smallest_groups"], hide_index=True)


def render(df: pd.DataFrame) -> None:
    scan, k = st.tabs(["Personal data", "How identifiable"])
    with scan:
        _scan(df)
    with k:
        _k(df)
