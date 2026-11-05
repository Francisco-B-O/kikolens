import pandas as pd
import streamlit as st

from kikolens.analyze import causality as cz
from kikolens.dashboard import components as ui

TITLE = "Causal effect"
SUBTITLE = "Did a treatment really change the outcome, or do the groups just differ? Adjusts for confounders and says how far to trust the answer."


def _notes(res: dict) -> None:
    for note in res.get("notes", []):
        ui.note(note, "medium")


def _binary_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if df[c].nunique(dropna=True) == 2]


def _run(key: str, signature, fn) -> dict | None:
    """Ejecuta al pulsar el botón y recuerda el resultado mientras no cambien las opciones."""
    if st.button("Estimate", key=f"{key}_go", type="primary"):
        try:
            st.session_state[key] = (signature, fn())
        except ValueError as e:
            st.session_state.pop(key, None)
            ui.note(str(e), "high")
    cached = st.session_state.get(key)
    return cached[1] if cached and cached[0] == signature else None


def _effect(df: pd.DataFrame) -> None:
    flags, numeric = _binary_columns(df), list(df.select_dtypes("number").columns)
    if not flags or not numeric:
        ui.note("This needs a treatment column with two values (treated or not) and a numeric outcome.", "low")
        return
    c1, c2, c3 = st.columns(3)
    treatment = c1.selectbox("Treatment", flags, key="cz_t")
    outcome = c2.selectbox("Outcome", [c for c in numeric if c != treatment], key="cz_o")
    method = c3.selectbox("Main method", list(cz.METHODS), key="cz_m")
    others = [c for c in df.columns if c not in (treatment, outcome)]
    covariates = st.multiselect("Adjust for (everything that may decide who is treated)", others, default=others[:8], key="cz_c")
    res = _run("cz_res", (treatment, outcome, method, tuple(covariates)), lambda: cz.estimate_effect(df, treatment, outcome, covariates or None, method=method))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    low, high = res["interval"]
    ui.stats([("Effect", f"{res['effect']:+.4g}"), ("95% range", f"{low:+.3g} to {high:+.3g}"), ("Raw difference", f"{res['naive']:+.4g}"),
              ("Rows outside overlap", f"{res['outside_overlap']:.0%}")])
    _notes(res)
    ui.chart(res["figure"], 360)
    ui.table(res["table"].round(4))
    a, b = st.columns(2)
    with a:
        ui.chart(res["overlap_figure"], 320)
    with b:
        ui.chart(res["balance_figure"], 320)


def _uplift(df: pd.DataFrame) -> None:
    flags, numeric = _binary_columns(df), list(df.select_dtypes("number").columns)
    if not flags or not numeric:
        ui.note("This needs a treatment column with two values and a numeric outcome.", "low")
        return
    c1, c2 = st.columns(2)
    treatment = c1.selectbox("Treatment", flags, key="cu_t")
    outcome = c2.selectbox("Outcome", [c for c in numeric if c != treatment], key="cu_o")
    others = [c for c in df.columns if c not in (treatment, outcome)]
    covariates = st.multiselect("Characteristics that may change the effect", others, default=others[:8], key="cu_c")
    res = _run("cu_res", (treatment, outcome, tuple(covariates)), lambda: cz.uplift(df, treatment, outcome, covariates or None))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    ui.chart(res["figure"], 380)
    ui.table(res["deciles"].round(4))
    st.caption("What most changes the effect")
    ui.table(res["drivers"].rename("importance").round(4).to_frame())


def _threshold(df: pd.DataFrame) -> None:
    numeric = list(df.select_dtypes("number").columns)
    if len(numeric) < 2:
        ui.note("This needs two numeric columns: the one with the threshold and the outcome.", "low")
        return
    c1, c2, c3 = st.columns(3)
    running = c1.selectbox("Column with the threshold", numeric, key="cd_r")
    outcome = c2.selectbox("Outcome", [c for c in numeric if c != running], key="cd_o")
    cutoff = c3.number_input("Threshold", value=float(df[running].median()), key=f"cd_c_{running}")
    res = _run("cd_res", (running, outcome, cutoff), lambda: cz.regression_discontinuity(df, running, outcome, cutoff))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    ui.chart(res["figure"], 400)
    ui.table(res["table"].round(4))


def _synthetic(df: pd.DataFrame) -> None:
    numeric = list(df.select_dtypes("number").columns)
    if len(df.columns) < 3 or not numeric:
        ui.note("This needs a unit column (city, store...), a time column and a numeric outcome.", "low")
        return
    c1, c2, c3 = st.columns(3)
    unit = c1.selectbox("Unit", [c for c in df.columns if df[c].nunique() <= 200 and c not in numeric] or list(df.columns), key="cs_u")
    time = c2.selectbox("Time", [c for c in df.columns if c != unit], key="cs_t")
    outcome = c3.selectbox("Outcome", numeric, key="cs_o")
    units = sorted(df[unit].dropna().unique().tolist(), key=str)
    times = sorted(df[time].dropna().unique().tolist(), key=str)
    if len(units) < 4 or len(times) < 8:
        ui.note("This needs at least four units and eight periods.", "low")
        return
    d1, d2 = st.columns(2)
    treated = d1.selectbox("Treated unit", units, key="cs_x")
    when = d2.selectbox("First period after the intervention", times, index=int(len(times) * 0.7), key="cs_w")
    res = _run("cs_res", (unit, time, outcome, str(treated), str(when)), lambda: cz.synthetic_control(df, unit, time, outcome, treated, when))
    if res is None:
        return
    ui.note(res["summary"], "ok" if res["p_value"] < 0.1 else "low")
    _notes(res)
    ui.chart(res["figure"], 380)
    ui.chart(res["gap_figure"], 340)
    ui.table(res["weights"].rename("weight").round(4).to_frame())


def _instrument(df: pd.DataFrame) -> None:
    numeric = list(df.select_dtypes("number").columns)
    if len(numeric) < 3:
        ui.note("This needs a numeric outcome, a treatment and at least one instrument.", "low")
        return
    c1, c2 = st.columns(2)
    outcome = c1.selectbox("Outcome", numeric, key="ci_o")
    endogenous = c2.selectbox("Treatment (the one that may be confounded)", [c for c in numeric if c != outcome], key="ci_e")
    rest = [c for c in numeric if c not in (outcome, endogenous)]
    instruments = st.multiselect("Instrument(s): move the treatment but touch the outcome only through it", rest, key="ci_i")
    controls = st.multiselect("Controls", [c for c in rest if c not in instruments], key="ci_c")
    if not instruments:
        ui.note("Choose at least one instrument.", "low")
        return
    res = _run("ci_res", (outcome, endogenous, tuple(instruments), tuple(controls)), lambda: cz.iv_2sls(df, outcome, endogenous, instruments, controls))
    if res is None:
        return
    ui.note(res["summary"], "ok" if res["first_stage_f"] >= 10 else "low")
    ui.stats([("Effect", f"{res['effect']:+.4g}"), ("Ordinary regression", f"{res['ols']:+.4g}"), ("First-stage F", f"{res['first_stage_f']:.1f}"), ("p-value", f"{res['p_value']:.3f}")])
    _notes(res)


def render(df: pd.DataFrame) -> None:
    tabs = st.tabs(["Effect of a treatment", "Who responds most", "Threshold", "Synthetic control", "Instrument"])
    for tab, fn in zip(tabs, (_effect, _uplift, _threshold, _synthetic, _instrument)):
        with tab:
            fn(df)
