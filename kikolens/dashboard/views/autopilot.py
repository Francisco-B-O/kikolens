import pandas as pd
import streamlit as st

from kikolens.analyze import forecasting as fx
from kikolens.analyze.autopilot import column_roles, next_steps, run_autopilot, suggest_target
from kikolens.dashboard import components as ui
from kikolens.dashboard import state

TITLE = "Autopilot"
SUBTITLE = "A first full analysis in one click: clean, find patterns, model a column and forecast dates."


def _run(df: pd.DataFrame, target, horizon: int, thorough: bool) -> None:
    bar = st.progress(0.0, text="Starting...")
    try:
        res = run_autopilot(df, target, forecast_horizon=horizon, fast=not thorough,
                            progress=lambda f, text: bar.progress(min(f, 1.0), text=f"{text}..."))
    except ValueError as e:
        bar.empty()
        ui.note(str(e), "high")
        return
    bar.empty()
    st.session_state.pilot_res = res


def render(df: pd.DataFrame) -> None:
    guess = suggest_target(df, column_roles(df))
    options = ["(let it choose)"] + list(df.columns)
    left, right = st.columns(2, gap="large")
    with left:
        choice = st.selectbox("Column to predict", options, key="pilot_target",
                              help=f"Left on automatic it would pick: {guess}." if guess else "No suitable column was found.")
        thorough = st.checkbox("Thorough (every model, tuned)", False, key="pilot_thorough",
                               help="Slower, and usually a little more accurate.")
    with right:
        horizon = st.slider("Periods to forecast", 3, 60, 12, key="pilot_horizon")
    if st.button("Analyze everything", key="pilot_go", type="primary"):
        _run(df, None if choice == options[0] else choice, horizon, thorough)

    res = st.session_state.get("pilot_res")
    if not res:
        return

    ui.note(res["narrative"], "accent")
    ui.stats([("Quality before", f"{res['quality_before']['score']:.0f}"),
              ("Quality after", f"{res['quality_after']['score']:.0f}"),
              ("Findings", len(res["findings"])),
              ("Best model", res["model"]["best_model"] if res["model"] else "none")])
    for line in res["problems"]:
        ui.note(line, "medium")

    st.markdown("#### What it did")
    for line in res["steps"]:
        st.markdown(f"- {line}")
    if len(res["recipe"]):
        if st.button("Apply the cleaning to my data", key="pilot_apply",
                     help="Adds these steps to the recipe, so you can undo them or replay them on new data."):
            steps = [(s.op, s.params, s.message) for s in res["recipe"].steps]
            state.apply_steps(res["clean"], steps, "Autopilot cleaning")
            st.rerun()

    if res["findings"]:
        st.markdown("#### Findings")
        for finding in res["findings"][:8]:
            ui.note(finding["message"], finding["severity"] if finding["severity"] in ("high", "medium", "low") else "")

    if res["model"]:
        model = res["model"]
        st.markdown(f"#### Predicting {model['target']}")
        ui.chart(model["plots"]["leaderboard"], 120 + 34 * len(model["leaderboard"]))
        st.caption("Open the Model lab for tuning, explanations and where the model fails.")

    if res["forecast"]:
        fc = res["forecast"]
        st.markdown(f"#### Forecast of {fc['value']}")
        ui.chart(fx.forecast_figure(fc["series"], fc["result"]), 400)
        st.caption(fc["result"]["summary"])

    st.markdown("#### Suggested next steps")
    for line in next_steps(res):
        st.markdown(f"- {line}")
