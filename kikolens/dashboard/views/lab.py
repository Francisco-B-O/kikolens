import io

import joblib
import pandas as pd
import streamlit as st

from kikolens.analyze import explain as ex
from kikolens.analyze import modellab as lab
from kikolens.analyze.predict import EXTENSION, build_bundle
from kikolens.dashboard import components as ui

TITLE = "Model lab"
SUBTITLE = "Compare many models fairly, tune the best one and check it for problems."


def _controls(df: pd.DataFrame) -> dict:
    target = st.selectbox("Column to predict", df.columns, index=len(df.columns) - 1, key="lab_target")
    task = lab.detect_task(df[target].dropna()) if df[target].notna().any() else "classification"
    default_pool = list(lab.candidates(task))
    pool = default_pool + list(lab.ADVANCED)

    left, right = st.columns(2, gap="large")
    with left:
        exclude = st.multiselect("Leave out columns", [c for c in df.columns if c != target], key="lab_exclude",
                                 help="Use this for columns that would not be known when making a real prediction.")
        models = st.multiselect(f"Models to compare ({task})", pool, default=default_pool, key=f"lab_models_{task}",
                                help="SVM, AdaBoost, Bagging and the voting and stacking ensembles are slower, so they start unticked.")
    with right:
        folds = st.slider("Cross-validation folds", 3, 10, 5, key="lab_folds",
                          help="Each model is trained and scored this many times on different slices of the training data.")
        test_size = st.slider("Share kept aside for the final test", 0.1, 0.4, 0.2, 0.05, key="lab_test")
        tune = st.checkbox("Tune the best model", True, key="lab_tune")
        n_iter = st.slider("Tuning attempts", 5, 60, 20, key="lab_iter", disabled=not tune)
        balance = st.checkbox("Balance classes", False, key="lab_balance", disabled=task != "classification",
                              help="Gives rarer classes more weight, for models that support it.")
    return dict(target=target, drop=exclude, models=models, folds=folds, test_size=test_size, tune=tune,
                n_iter=n_iter, balance=balance)


def _train(df: pd.DataFrame, params: dict) -> None:
    bar = st.progress(0.0, text="Starting...")
    try:
        res = lab.run_model_lab(df, progress=lambda f, name: bar.progress(min(f, 1.0), text=f"Evaluating {name}..."), **params)
    except ValueError as e:
        bar.empty()
        ui.note(str(e), "high")
        return
    bar.empty()
    st.session_state.lab_res = res


def _warnings(res: dict) -> None:
    for leak in res["leakage"]:
        ui.note(leak["message"], "high", bold="Possible data leak.")
    if res["balance"] and res["balance"]["imbalanced"]:
        b = res["balance"]
        ui.note(f"The rarest class ('{b['minority']}') is {b['minority_share']:.1%} of the rows. "
                "Judge the model by balanced accuracy or F1, not plain accuracy.", "medium", bold="Imbalanced classes.")
    if res["dropped"]:
        with st.expander(f"{len(res['dropped'])} column(s) were not used"):
            ui.table(pd.DataFrame(res["dropped"]), hide_index=True)


def _leaderboard(res: dict) -> None:
    ui.chart(res["plots"]["leaderboard"], 120 + 34 * len(res["leaderboard"]))
    board = res["leaderboard"].copy()
    shown = [c for c in board.columns if not c.endswith("std") and c != "total time (s)"]
    ui.table(board[shown].round(4), hide_index=True)
    if res["tuning"]:
        t = res["tuning"]
        verdict = "improved" if t["improved"] else "did not improve"
        ui.note(f"Tuning {verdict} {res['best_model']}: {t['default_cv']:.4f} to {t['tuned_cv']:.4f} "
                f"(cross-validated). Settings tried best: {t['params']}", "ok" if t["improved"] else "low")
    ui.download_csv(board, "leaderboard.csv", "Download leaderboard (CSV)")


def _diagnostics(res: dict) -> None:
    keys = [k for k in ("performance", "residuals", "residual_hist", "roc", "pr", "calibration") if k in res["plots"]]
    for i in range(0, len(keys), 2):
        cols = st.columns(2)
        for col, key in zip(cols, keys[i:i + 2]):
            with col:
                ui.chart(res["plots"][key], 360)
    st.markdown("#### Learning curve")
    ui.chart(res["plots"]["learning_curve"], 340)


def _explain(res: dict) -> None:
    ui.chart(res["plots"]["importance"], 380)
    pipe, X, task = res["pipeline"], res["X_train"], res["task_type"]

    st.markdown("#### How one feature changes the prediction")
    feature = st.selectbox("Feature", list(res["feature_importance"]), key="lab_pdp")
    try:
        ui.chart(lab.partial_dependence_figure(pipe, X, feature, task), 340)
    except Exception as e:
        ui.note(f"Could not draw this feature: {e}", "low")

    st.markdown("#### Why this row got its prediction")
    X_test = res["X_test"]
    idx = st.number_input("Test row", 0, len(X_test) - 1, 0, key="lab_row")
    row = X_test.iloc[[int(idx)]]
    effects = lab.explain_row(pipe, X, row, task).head(10)
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(x=effects["effect"][::-1], y=effects["feature"][::-1], orientation="h"))
    fig.update_layout(title="Change in prediction caused by each feature, compared with a typical value")
    ui.chart(fig, 340)
    st.caption(f"Prediction: {pipe.predict(row)[0]} · actual: {res['y_test'].iloc[int(idx)]}")

    st.markdown("#### Contribution of each feature (Shapley)")
    st.caption("Fairer than the effect above when features are related: each feature gets its average contribution over many orders.")
    if len(res["features"]) > ex.MAX_FEATURES_SHAP:
        ui.note(f"Shapley values need {ex.MAX_FEATURES_SHAP} features or fewer; leave some columns out and run the lab again.", "low")
    elif st.button("Compute Shapley values for this row", key="lab_shap_row"):
        with st.spinner("Computing..."):
            st.session_state.lab_shap = (int(idx), ex.shapley_row(pipe, X, row, task))
    shap = st.session_state.get("lab_shap")
    if shap and shap[0] == int(idx):
        ui.chart(ex.shapley_figure(shap[1]), 360)
    if len(res["features"]) <= ex.MAX_FEATURES_SHAP and st.button("Global importance from Shapley values", key="lab_shap_all"):
        with st.spinner("Computing on a sample of rows..."):
            st.session_state.lab_shap_global = ex.shapley_global(pipe, X, X_test, task)
    if st.session_state.get("lab_shap_global") is not None:
        ui.table(st.session_state.lab_shap_global.round(4), hide_index=True)


def _errors(res: dict) -> None:
    pipe, X_test, y_test, task = res["pipeline"], res["X_test"], res["y_test"], res["task_type"]
    st.markdown("#### Where the model fails")
    try:
        found = ex.error_analysis(pipe, X_test, y_test, task)
    except ValueError as e:
        ui.note(str(e), "low")
        found = None
    if found:
        ui.note(found["summary"], "medium" if len(found["segments"]) else "ok")
        if len(found["segments"]):
            ui.table(found["segments"].round(3), hide_index=True)
        with st.expander("The rows it gets most wrong"):
            ui.table(found["worst"].round(3))

    if task == "classification" and len(set(y_test)) == 2:
        st.markdown("#### Choose the decision threshold")
        classes = sorted(set(y_test.astype(str)))
        c1, c2, c3 = st.columns(3)
        positive = c1.selectbox("Positive class", classes, index=len(classes) - 1, key="lab_pos")
        fp = c2.number_input("Cost of a false alarm", 0.0, 1e6, 1.0, key="lab_cfp")
        fn = c3.number_input("Cost of a miss", 0.0, 1e6, 5.0, key="lab_cfn")
        try:
            out = ex.threshold_analysis(pipe, X_test, y_test, fp, fn, positive)
        except ValueError as e:
            ui.note(str(e), "high")
        else:
            st.markdown(out["summary"])
            ui.chart(ex.threshold_figure(out), 340)

    st.markdown("#### Compare with another model")
    names = [n for n in res["leaderboard"]["model"] if res["leaderboard"].set_index("model").loc[n, "status"] == "ok"]
    c1, c2 = st.columns(2)
    a = c1.selectbox("Model A", names, index=names.index(res["best_model"]) if res["best_model"] in names else 0, key="lab_pa")
    b = c2.selectbox("Model B", [n for n in names if n != a], key="lab_pb")
    if st.button("Compare them row by row", key="lab_pair"):
        try:
            st.session_state.lab_pair_res = ex.compare_pair(res["X_train"], res["y_train"], X_test, y_test, task, a, b)
        except ValueError as e:
            ui.note(str(e), "high")
    pair = st.session_state.get("lab_pair_res")
    if pair and set(pair["scores"]) == {a, b}:
        st.markdown(pair["verdict"])
        ui.table(pair["disagreements"].round(3))


def _predict(res: dict, df: pd.DataFrame) -> None:
    pipe, X = res["pipeline"], res["X_train"]
    cols = st.columns(3)
    values = {}
    for i, col in enumerate(X.columns):
        target = cols[i % 3]
        s = X[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            values[col] = target.number_input(col, value=float(s.median()), key=f"lab_in_{col}")
        else:
            options = sorted(s.dropna().astype(str).unique())[:200]
            values[col] = target.selectbox(col, options, key=f"lab_in_{col}")
    if st.button("Predict", key="lab_predict", type="primary"):
        frame = pd.DataFrame([values], columns=X.columns).astype({c: X[c].dtype for c in X.columns
                                                                    if pd.api.types.is_numeric_dtype(X[c])}, errors="ignore")
        out = pipe.predict(frame)[0]
        st.metric("Prediction", f"{out:.4f}" if res["task_type"] == "regression" else str(out))
        if res["task_type"] == "classification" and hasattr(pipe, "predict_proba"):
            probs = pd.DataFrame({"class": pipe.classes_, "probability": pipe.predict_proba(frame)[0]})
            ui.table(probs.sort_values("probability", ascending=False).round(4), hide_index=True)


def render(df: pd.DataFrame) -> None:
    params = _controls(df)
    if st.button("Run the lab", key="lab_run", type="primary"):
        if not params["models"]:
            ui.note("Select at least one model.", "medium")
        else:
            _train(df, params)

    res = st.session_state.get("lab_res")
    if not res or res["target"] != params["target"]:
        return

    st.divider()
    cv = res["leaderboard"].set_index("model").loc[res["best_model"], res["primary_metric"]]
    items = [("Best model", res["best_model"]), (f"{res['primary_metric']} (cross-validated)", f"{cv:.4f}")]
    items += [(f"Test {k}", f"{v:.4f}") for k, v in list(res["metrics"].items())[:3]]
    ui.stats(items)
    st.markdown(res["narrative"])
    _warnings(res)

    buffer = io.BytesIO()
    joblib.dump(build_bundle(res, df, name=f"{res['best_model']} for {res['target']}", recipe=st.session_state.recipe), buffer)
    st.download_button("Download model package", buffer.getvalue(), f"{res['target']}_lab{EXTENSION}", key="lab_export",
                       help="Includes the expected columns and ranges. Use it in the Predict page or with: kikolens score")

    tabs = st.tabs(["Leaderboard", "Diagnostics", "Explain", "Errors", "Try it"])
    with tabs[0]:
        _leaderboard(res)
    with tabs[1]:
        _diagnostics(res)
    with tabs[2]:
        _explain(res)
    with tabs[3]:
        _errors(res)
    with tabs[4]:
        _predict(res, df)
