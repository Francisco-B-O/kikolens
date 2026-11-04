"""Pestañas del laboratorio sobre un modelo ya entrenado: incertidumbre, qué cambiaría el resultado y equidad."""
import pandas as pd
import streamlit as st

from kikolens.analyze import conformal, counterfactual, fairness
from kikolens.dashboard import components as ui


def uncertainty(res: dict) -> None:
    st.caption("A range (or a short list of classes) that contains the truth about as often as you ask, whatever the model. "
               "Half of the test rows calibrate it and the other half check it.")
    task, pipe = res["task_type"], res["pipeline"]
    coverage = st.slider("How often should it be right", 0.7, 0.99, 0.9, 0.01, key="lab_cov")
    X, y = res["X_test"], res["y_test"]
    half = len(X) // 2
    if half < 20:
        ui.note("The test set is too small to calibrate. Run the lab with more rows or a larger test share.", "low")
        return
    try:
        model = conformal.ConformalModel(pipe, task, coverage).calibrate(X.iloc[:half], y.iloc[:half])
        check = model.evaluate(X.iloc[half:], y.iloc[half:])
    except ValueError as e:
        ui.note(str(e), "high")
        return
    items = [("Wanted", f"{coverage:.0%}"), ("Observed on unseen rows", f"{check['coverage']:.1%}")]
    if task == "regression":
        items.append(("Range width", f"{check['width']:.4g}", "total, around each prediction"))
        ui.stats(items)
        intervals = model.predict_interval(X.iloc[half:])
        ui.chart(conformal.interval_figure(intervals, y.iloc[half:]), 380)
    else:
        items += [("Average classes in a set", f"{check['average_set_size']:.2f}"),
                  ("Sets with one class", f"{check['single_answer']:.0%}")]
        ui.stats(items)
        out = model.predict_set(X.iloc[half:])
        ui.table(out.assign(actual=y.iloc[half:].to_numpy()).head(50), hide_index=False)
        st.caption("A set with several classes means the model is not sure about that row.")


def what_would_change(res: dict) -> None:
    st.caption("The smallest changes to one row that would make the model give a different answer. Useful to explain a "
               "decision and to see which levers exist.")
    pipe, X, X_test, task = res["pipeline"], res["X_train"], res["X_test"], res["task_type"]
    idx = int(st.number_input("Test row", 0, len(X_test) - 1, 0, key="lab_cf_row"))
    row = X_test.iloc[[idx]]
    fixed = st.multiselect("Things that cannot change", list(X.columns), key="lab_cf_fixed",
                           help="For example age or country.")
    kwargs = {}
    if task == "classification":
        classes = [str(c) for c in pipe.classes_]
        current = str(pipe.predict(row)[0])
        options = [c for c in classes if c != current] or classes
        kwargs["desired"] = st.selectbox(f"Now: {current}. Which result do you want?", options, key="lab_cf_want")
    else:
        now = float(pipe.predict(row)[0])
        low = st.number_input(f"Now: {now:.4g}. At least", value=now * 1.1 if now else 1.0, key="lab_cf_low")
        kwargs["target_range"] = (low, None)
    changes = st.radio("Variables allowed to change together", [1, 2], index=1, horizontal=True, key="lab_cf_n")
    if st.button("Find what would change it", key="lab_cf_go", type="primary"):
        try:
            with st.spinner("Searching..."):
                st.session_state.lab_cf = (idx, counterfactual.find(pipe, X, row, task, fixed=fixed, max_changes=changes, **kwargs))
        except ValueError as e:
            st.session_state.pop("lab_cf", None)
            ui.note(str(e), "high")
    cached = st.session_state.get("lab_cf")
    if cached and cached[0] == idx:
        out = cached[1]
        ui.note(out["summary"], "ok" if len(out["found"]) or out["already"] else "low")
        if len(out["found"]):
            ui.table(out["found"], hide_index=True)


def fair(res: dict, df: pd.DataFrame) -> None:
    if res["task_type"] != "classification":
        ui.note("The fairness check compares how a classifier treats groups. Train a model that predicts a category.", "low")
        return
    st.caption("Compares how the model treats each group. It cannot say what is fair, but it shows differences you would need to justify.")
    X, y, pipe = res["X_test"], res["y_test"], res["pipeline"]
    candidates = [c for c in df.columns if c != res["target"] and 2 <= df[c].nunique() <= 12]
    if not candidates:
        ui.note("No column with 2 to 12 groups was found to compare.", "low")
        return
    c1, c2 = st.columns(2)
    group = c1.selectbox("Compare groups of", candidates, key="lab_fg")
    classes = sorted(set(y.astype(str)))
    positive = c2.selectbox("Favourable result", classes, index=len(classes) - 1, key="lab_fp")
    try:
        aligned = df.dropna(subset=[res["target"]]).reset_index(drop=True)   # así se numeran las filas en el laboratorio
        out = fairness.audit(pipe, X, y, aligned.loc[X.index, group], positive)
    except (ValueError, KeyError) as e:
        ui.note(str(e), "high")
        return
    ui.note(out["summary"], "medium" if out["flags"] else "ok")
    for flag in out["flags"]:
        ui.note(flag, "medium")
    ui.chart(out["figure"], 360)
    ui.table(out["table"].drop(columns=["enough rows"]).round(3), hide_index=True)
    if group in X.columns:
        st.caption(f"'{group}' is also a column the model uses; leaving it out does not remove the effect if other columns carry the same information.")
