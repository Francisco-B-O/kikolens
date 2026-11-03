import pandas as pd
import streamlit as st

from kikolens.analyze.neural import ACTIVATIONS, PRESETS, SOLVERS, parse_hidden_layers, run_neural_network
from kikolens.dashboard import components as ui
from kikolens.dashboard.views import _modeling as modeling

TITLE = "Neural network"
SUBTITLE = "Train a multilayer perceptron and watch how it learns."


def render(df: pd.DataFrame) -> None:
    target = st.selectbox("Column to predict", df.columns, index=len(df.columns) - 1, key="nn_target")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Architecture")
        preset = st.selectbox("Preset", list(PRESETS) + ["Custom"], index=1, key="nn_preset")
        if preset == "Custom":
            layers_text = st.text_input("Hidden layers", "64, 32", help="Neurons per layer, separated by commas.", key="nn_layers")
        else:
            layers_text = ", ".join(map(str, PRESETS[preset]))
            st.caption(f"Hidden layers: {layers_text}")
        activation = st.selectbox("Activation", ACTIVATIONS, key="nn_act",
                                  help="relu is a good default; tanh and logistic are smoother; identity is linear.")
    with right:
        st.markdown("#### Training")
        solver = st.selectbox("Optimizer", SOLVERS, key="nn_solver", help="adam suits most data; lbfgs is fast on small data.")
        learning_rate = st.select_slider("Learning rate", [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], 0.001, key="nn_lr",
                                         disabled=solver == "lbfgs")
        alpha = st.select_slider("L2 regularization", [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0], 0.0001, key="nn_alpha")
        max_iter = st.slider("Maximum epochs", 20, 1000, 300, 10, key="nn_epochs")
        early = st.checkbox("Stop early when validation stops improving", True, key="nn_early", disabled=solver == "lbfgs")
        test_size = st.slider("Share of data held out for testing", 0.1, 0.4, 0.2, 0.05, key="nn_test")

    if st.button("Train network", key="nn_train", type="primary"):
        try:
            parse_hidden_layers(layers_text)
            with st.spinner("Training..."):
                st.session_state.nn_res = run_neural_network(
                    df, target, hidden_layers=layers_text, activation=activation, solver=solver,
                    learning_rate=learning_rate, alpha=alpha, max_iter=max_iter,
                    early_stopping=early, test_size=test_size)
        except ValueError as e:
            ui.note(str(e), "high")

    res = st.session_state.get("nn_res")
    if not res or res.get("target") != target:
        return

    st.divider()
    modeling.metrics_row(res, [("Parameters", f"{res['n_parameters']:,}"), ("Epochs", res["n_iter"])])
    st.markdown(res["narrative"])

    base = st.session_state.get("automl_res")
    if base and base.get("target") == target and base["task_type"] == res["task_type"]:
        st.markdown("#### Compared with the classical model")
        keys = sorted(set(base["metrics"]) & set(res["metrics"]))
        if keys:
            ui.table(pd.DataFrame({"Classical model": [base["metrics"][k] for k in keys],
                                   "Neural network": [res["metrics"][k] for k in keys]}, index=keys).round(4))
        else:
            st.caption("The two runs report different metrics, so they cannot be compared directly.")

    a, b = st.columns(2)
    with a:
        ui.chart(res["plots"]["architecture"], 360)
    with b:
        ui.chart(res["plots"]["training"], 360)
    c, d = st.columns(2)
    with c:
        ui.chart(res["plots"]["performance"], 360)
    with d:
        ui.chart(res["plots"]["importance"], 360)

    modeling.export_button(res, "nn_export")
    st.divider()
    modeling.explain_and_simulate(res, df, "nn")
