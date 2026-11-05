import pandas as pd
import streamlit as st

from kikolens.analyze import simulation as sim
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Simulation"
SUBTITLE = "What could happen when inputs are uncertain, and what is the best choice within your limits."

EXAMPLE_INPUTS = "price = normal(10, 2)\nvolume = triangular(80, 100, 140)\ncost = fixed(800)"


def _lines(text: str) -> list:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _monte_carlo(df: pd.DataFrame) -> None:
    st.caption(t("Each input gets a range of plausible values; the formula is evaluated thousands of times to show the range of outcomes."))
    numeric = list(df.select_dtypes("number").columns)
    from_data = st.multiselect("Inputs taken from the data (their values are resampled and their correlation is kept)", numeric, key="mc_cols")
    typed = st.text_area("More inputs, one per line: name = distribution(numbers)", EXAMPLE_INPUTS if not from_data else "", key="mc_inputs",
                         help="normal(mean, sd), uniform(low, high), triangular(low, mode, high), lognormal(mean, sd), beta(a, b), poisson(mean), fixed(value)")
    formula = st.text_input("Formula for the result", "price * volume - cost" if not from_data else " * ".join(from_data[:2]), key="mc_formula")
    c1, c2, c3 = st.columns(3)
    n = c1.select_slider("Simulations", [1000, 5000, 10000, 50000, 100000], 10000, key="mc_n")
    use_threshold = c2.checkbox("Chance of going above a value", key="mc_use")
    threshold = c3.number_input("Value", value=0.0, key="mc_thr", disabled=not use_threshold)
    correlation = st.text_input("Correlations between typed inputs (optional), like price,volume = -0.5; separate with semicolons", key="mc_corr")
    if st.button("Simulate", key="mc_go", type="primary"):
        try:
            inputs, correlations = {}, {}
            if from_data:
                inputs, found = sim.inputs_from_data(df, from_data)
                if found is not None:
                    correlations = {(a, b): float(found.loc[a, b]) for a in found.index for b in found.columns if a < b}
            for line in _lines(typed):
                name, spec = sim.parse_input(line)
                inputs[name] = spec
            for part in filter(None, (p.strip() for p in correlation.split(";"))):
                pair, _, rho = part.partition("=")
                a, _, b = pair.partition(",")
                correlations[(a.strip(), b.strip())] = float(rho)
            res = sim.monte_carlo(inputs, formula, n, correlations or None, threshold if use_threshold else None)
            st.session_state["mc_res"] = res
        except ValueError as e:
            st.session_state.pop("mc_res", None)
            ui.note(str(e), "high")
    res = st.session_state.get("mc_res")
    if res is None:
        return
    ui.note(res["summary"], "ok")
    p = res["percentiles"]
    items = [("Average", f"{res['mean']:,.4g}"), ("Median", f"{p['p50']:,.4g}"), ("5th percentile", f"{p['p5']:,.4g}"), ("95th percentile", f"{p['p95']:,.4g}")]
    if "p_above" in res:
        items.append(("Chance above the value", f"{res['p_above']:.0%}"))
    ui.stats(items)
    for note in res["notes"]:
        ui.note(note, "medium")
    ui.chart(res["figure"], 380)
    a, b = st.columns(2)
    with a:
        ui.chart(res["tornado_figure"], 300)
    with b:
        ui.chart(res["convergence_figure"], 300)
    ui.download_csv(res["samples"], "simulations.csv", "Download the simulations (CSV)")


def _program() -> None:
    st.caption(t("Choose how much of each thing to do to get the best total, within limits (budget, capacity, minimums)."))
    c1, c2 = st.columns([1, 3])
    maximize = c1.radio("Goal", ["Maximize", "Minimize"], key="lp_goal") == "Maximize"
    objective = c2.text_input("Objective (a sum of numbers times variables)", "3*x + 5*y", key="lp_obj")
    constraints = st.text_area("Limits, one per line", "x <= 4\n2*y <= 12\n3*x + 2*y <= 18", key="lp_cons")
    integer = st.text_input("Variables that must be whole numbers (optional, comma-separated)", key="lp_int")
    if st.button("Solve", key="lp_go", type="primary"):
        try:
            res = sim.linear_program(sim.parse_linear(objective), [sim.parse_linear_constraint(line) for line in _lines(constraints)], None, maximize,
                                     [v.strip() for v in integer.split(",") if v.strip()])
            st.session_state["lp_res"] = res
        except ValueError as e:
            st.session_state.pop("lp_res", None)
            ui.note(str(e), "high")
    res = st.session_state.get("lp_res")
    if res is None:
        return
    ui.note(res["summary"], "ok")
    ui.chart(res["figure"], 320)
    ui.table(res["constraints"].round(4))
    if res["constraints"]["worth of one more unit"].notna().any():
        st.caption(t("The worth of one more unit is how much the objective changes if that limit is relaxed by one unit; limits that are not reached are worth nothing."))


def _formula() -> None:
    st.caption(t("For a formula that is not linear (diminishing returns, products), the search tries several starting points."))
    expression = st.text_input("Formula to maximize or minimize", "x * y", key="of_expr")
    bounds = st.text_area("Variables and their bounds, one per line: name = minimum:maximum", "x = 0:100\ny = 0:100", key="of_bounds")
    cons = st.text_area("Restrictions (optional), one per line", "x + y <= 20", key="of_cons")
    goal = st.radio("Goal", ["Maximize", "Minimize"], key="of_goal", horizontal=True)
    if st.button("Optimize", key="of_go", type="primary"):
        try:
            res = sim.optimize_expression(expression, dict(sim.parse_variable(line) for line in _lines(bounds)), _lines(cons), goal == "Maximize")
            st.session_state["of_res"] = res
        except ValueError as e:
            st.session_state.pop("of_res", None)
            ui.note(str(e), "high")
    res = st.session_state.get("of_res")
    if res is None:
        return
    ui.note(res["summary"], "ok")
    for note in res["notes"]:
        ui.note(note, "medium")
    ui.table(res["solution"].rename("value").to_frame().round(6))


def render(df: pd.DataFrame) -> None:
    tabs = st.tabs(["Monte Carlo", "Linear program", "Optimize a formula"])
    with tabs[0]:
        _monte_carlo(df)
    with tabs[1]:
        _program()
    with tabs[2]:
        _formula()
