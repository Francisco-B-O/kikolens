import pandas as pd
import streamlit as st

from kikolens.analyze import regression as rg
from kikolens.dashboard import components as ui

TITLE = "Regression"
SUBTITLE = "What pushes a result up or down, other things equal. Coefficients, ranges and p-values, each said in a sentence."


def render(df: pd.DataFrame) -> None:
    target = st.selectbox("Column to explain", df.columns, index=len(df.columns) - 1, key="rg_t")
    two_valued = df[target].nunique(dropna=True) == 2
    numeric = pd.api.types.is_numeric_dtype(df[target]) and not pd.api.types.is_bool_dtype(df[target])
    kind = "logit" if two_valued else "ols"
    if not two_valued and not numeric:
        ui.note("This needs a numeric column (linear regression) or a column with exactly two values (logistic regression).", "low")
        return
    st.caption("Linear regression for a number." if kind == "ols" else "Logistic regression for a column with two values.")
    others = [c for c in df.columns if c != target]
    features = st.multiselect("Explain it with", others, default=others[:6], key="rg_f")
    positive = None
    if kind == "logit":
        labels = sorted(df[target].dropna().astype(str).unique())
        positive = st.selectbox("Model the chance of", labels, index=len(labels) - 1, key="rg_p")
    if st.button("Fit the model", key="rg_go", type="primary"):
        try:
            res = rg.ols(df, target, features) if kind == "ols" else rg.logit(df, target, features, positive)
            st.session_state.rg_res = ((target, kind, tuple(features), positive), res)
        except ValueError as e:
            st.session_state.pop("rg_res", None)
            ui.note(str(e), "high")
    cached = st.session_state.get("rg_res")
    if not cached or cached[0] != (target, kind, tuple(features), positive):
        return
    res = cached[1]
    m = res["model"]
    ui.note(res["summary"], "accent")
    if kind == "ols":
        ui.stats([("R2", f"{m['r2']:.3f}"), ("Adjusted R2", f"{m['adjusted r2']:.3f}"), ("Rows used", f"{m['n']:,}"),
                  ("Typical error", f"{m['residual std']:.4g}")])
    else:
        ui.stats([("AUC", f"{m['auc']:.3f}"), ("Accuracy", f"{m['accuracy']:.1%}"), ("Always guessing the commonest", f"{m['baseline accuracy']:.1%}"),
                  ("Rows used", f"{m['n']:,}")])
    if res["rows dropped for missing values"]:
        st.caption(f"{res['rows dropped for missing values']:,} rows with empty values were left out.")
    for line in res["flags"]:
        ui.note(line, "medium")
    st.markdown("#### What it says")
    for line in res["sentences"]:
        st.markdown(f"- {line}")
    ui.chart(rg.coefficient_figure(res), 120 + 30 * len(res["table"]))
    shown = res["table"].round(4)
    ui.table(shown, hide_index=True)
    ui.download_csv(res["table"], "regression.csv", "Download the table")
    if kind == "ols":
        with st.expander("Errors against fitted values"):
            ui.chart(rg.residual_figure(res), 340)
    for col, why in res["dropped"]:
        st.caption(f"'{col}' was not used: {why}.")
