import pandas as pd
import streamlit as st

from kikolens.analyze import concepts as cp
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Concept lab"
SUBTITLE = "Move the controls and watch why statistics works the way it does. These simulations do not use your data."

EXTRA_FIGURES = ("population_figure", "curves_figure")


def _control(concept: str, key: str, spec: dict, container):
    wid = f"cl_{concept}_{key}"
    label = spec["label"]
    if spec["type"] == "choice":
        return container.selectbox(label, spec["choices"], index=spec["choices"].index(spec["default"]), key=wid,
                                   format_func=lambda c: t(cp.POPULATIONS[c][0]) if c in cp.POPULATIONS else c)
    low, high, default = spec["min"], spec["max"], spec["default"]
    if spec["type"] == "int":
        if high - low <= 100:
            return int(container.slider(label, int(low), int(high), int(default), key=wid))
        return int(container.number_input(label, int(low), int(high), int(default), key=wid))
    if high - low <= 4:
        return float(container.slider(label, float(low), float(high), float(default), step=0.01, key=wid))
    return float(container.number_input(label, float(low), float(high), float(default), key=wid))


def render(df: pd.DataFrame) -> None:
    names = list(cp.CONCEPTS)
    concept = st.selectbox("Concept", names, format_func=lambda n: t(cp.CONCEPTS[n]["title"]), key="cl_concept")
    spec = cp.CONCEPTS[concept]
    st.markdown(f"**{t(spec['question'])}**")
    params = spec["params"]
    columns = st.columns(min(3, len(params)))
    values = {key: _control(concept, key, p, columns[i % len(columns)]) for i, (key, p) in enumerate(params.items())}
    seed_key = f"cl_seed_{concept}"
    st.session_state.setdefault(seed_key, 0)
    if st.button("Draw new random data", key=f"cl_new_{concept}", help="The same settings with another random sample: the result changes a little, the lesson does not."):
        st.session_state[seed_key] += 1
    seed = st.session_state[seed_key]
    signature = (concept, tuple(sorted(values.items())), seed)
    cached = st.session_state.get("cl_result")
    if cached and cached[0] == signature:
        res = cached[1]
    else:
        try:
            res = cp.run(concept, seed, **values)
        except ValueError as e:
            ui.note(str(e), "high")
            return
        st.session_state["cl_result"] = (signature, res)
    ui.note(res["summary"], "ok")
    ui.chart(res["figure"], 420)
    for extra in EXTRA_FIGURES:
        if extra in res:
            ui.chart(res[extra], 340)
    if res.get("table") is not None:
        with st.expander(t("The numbers")):
            ui.table(res["table"].round(4))
    for note in res["notes"]:
        ui.note(note, "medium")
    st.markdown("**" + t("What to take from this") + "**")
    st.write(res["lesson"])
    st.markdown("**" + t("Try this") + "**")
    st.write(res["try"])
