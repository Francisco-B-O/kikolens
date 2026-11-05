import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from kikolens.analyze import bayes
from kikolens.analyze import experiment as exq
from kikolens.analyze import stattests as sx
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Experiments"
SUBTITLE = "Statistical tests explained in plain language, with effect sizes and A/B testing."


def _verdict(result: dict) -> None:
    ui.note(result["conclusion"], "ok" if result["significant"] else "low")


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as e:
        ui.note(str(e), "high")
        return None


def _groups(df: pd.DataFrame) -> None:
    nums = df.select_dtypes("number").columns.tolist()
    cats = [c for c in df.columns if c not in nums and df[c].nunique() <= sx.MAX_GROUPS] + \
           [c for c in nums if 2 <= df[c].nunique() <= 8]
    if not nums or not cats:
        ui.note("This needs a numeric column and a column that splits the rows into groups.", "low")
        return
    c1, c2, c3 = st.columns(3)
    value = c1.selectbox("Compare this number", nums, key="ex_gv")
    group = c2.selectbox("Between the groups of", cats, key="ex_gg")
    alpha = c3.select_slider("Significance level", [0.01, 0.05, 0.1], 0.05, key="ex_ga")
    result = _run(sx.compare_groups, df, value, group, alpha)
    if not result:
        return
    _verdict(result)
    ui.stats([("Test", result["test"]), ("p-value", "< 0.0001" if result["p_value"] < 1e-4 else f"{result['p_value']:.4f}"),
              (result["effect_name"], f"{result['effect']:.2f}", result["effect_size"])])
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for name, sample in result["samples"].items():
            fig.add_trace(go.Box(y=sample, name=name, boxmean=True))
        fig.update_layout(title=f"{value} by {group}", showlegend=False)
        ui.chart(fig, 340)
    with right:
        ui.table(result["summary"].round(4), hide_index=True)
        if result["post_hoc"] is not None:
            st.markdown("**Which groups differ**")
            ui.table(result["post_hoc"].round(5), hide_index=True)


def _association(df: pd.DataFrame) -> None:
    cats = [c for c in df.columns if df[c].nunique() <= sx.MAX_CATEGORIES and not pd.api.types.is_float_dtype(df[c])]
    if len(cats) < 2:
        ui.note("This needs two columns with a small set of values.", "low")
        return
    c1, c2 = st.columns(2)
    a = c1.selectbox("First column", cats, key="ex_aa")
    b = c2.selectbox("Second column", [c for c in cats if c != a], key="ex_ab")
    result = _run(sx.association, df, a, b)
    if not result:
        return
    _verdict(result)
    left, right = st.columns(2)
    with left:
        shares = result["table"].div(result["table"].sum(axis=1), axis=0)
        fig = px.imshow(shares, text_auto=".0%", aspect="auto", labels=dict(x=b, y=a, color="Share"),
                        title=f"Share of each {b} within {a}")
        ui.chart(fig, 360)
    with right:
        res = result["residuals"]
        fig = px.imshow(res, text_auto=".1f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-4, zmax=4,
                        labels=dict(x=b, y=a, color="Residual"), title="Where it differs from chance")
        ui.chart(fig, 360)
        st.caption("Values beyond about +2 or -2 are the combinations that appear much more or less than chance.")


def _correlation(df: pd.DataFrame) -> None:
    nums = df.select_dtypes("number").columns.tolist()
    if len(nums) < 2:
        ui.note("This needs two numeric columns.", "low")
        return
    c1, c2 = st.columns(2)
    x = c1.selectbox("First number", nums, key="ex_cx")
    y = c2.selectbox("Second number", [c for c in nums if c != x], key="ex_cy")
    result = _run(sx.correlation_test, df, x, y)
    if not result:
        return
    _verdict(result)
    ui.stats([("Pearson r", f"{result['pearson_r']:.3f}", f"95% CI {result['ci'][0]:.2f} to {result['ci'][1]:.2f}"),
              ("Spearman rho", f"{result['spearman_rho']:.3f}"), ("Rows used", f"{result['n']:,}")])
    data = result["data"]
    fig = px.scatter(data.sample(min(len(data), 3000), random_state=0), x=x, y=y, opacity=0.6, trendline="ols") \
        if _has_statsmodels() else px.scatter(data.sample(min(len(data), 3000), random_state=0), x=x, y=y, opacity=0.6)
    ui.chart(fig, 380)


def _has_statsmodels() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except ImportError:
        return False


def _ab(df: pd.DataFrame) -> None:
    two = [c for c in df.columns if df[c].nunique() == 2]
    outcomes = [c for c in df.columns if c in df.select_dtypes("number").columns or df[c].nunique() == 2]
    if not two or not outcomes:
        ui.note("An A/B test needs a column with exactly two groups and an outcome column.", "low")
        return
    c1, c2, c3 = st.columns(3)
    group = c1.selectbox("Group column (two values)", two, key="ex_bg")
    options = sorted(df[group].dropna().astype(str).unique())
    control = c2.selectbox("Control group", options, key="ex_bc")
    outcome = c3.selectbox("Outcome", [c for c in outcomes if c != group], key="ex_bo")
    success = None
    if df[outcome].nunique() == 2:
        success = st.selectbox("Count as success", sorted(df[outcome].dropna().astype(str).unique()),
                               index=1, key="ex_bs")
    result = _run(sx.ab_test, df, group, outcome, control, success)
    if result:
        _verdict(result)
        ui.stats([(f"{result['control']} (control)", f"{result['values'][0]:.4g}", f"{result['n'][0]:,} rows"),
                  (f"{result['treatment']}", f"{result['values'][1]:.4g}", f"{result['n'][1]:,} rows"),
                  ("Relative lift", f"{result['relative_lift']:+.1%}"),
                  ("p-value", "< 0.0001" if result["p_value"] < 1e-4 else f"{result['p_value']:.4f}")])
        fig = go.Figure(go.Bar(x=[f"{result['control']} (control)", result["treatment"]], y=list(result["values"])))
        fig.update_layout(title="Result by group", showlegend=False)
        ui.chart(fig, 300)

    st.markdown("#### How many rows will I need?")
    st.caption("Plan an experiment before running it.")
    kind = st.radio("Measuring", ["A rate (conversion, share)", "An average"], horizontal=True, key="ex_pk")
    p1, p2, p3 = st.columns(3)
    try:
        if kind.startswith("A rate"):
            base = p1.number_input("Current rate", 0.001, 0.999, 0.10, 0.01, key="ex_pb", format="%.3f")
            change = p2.number_input("Smallest change worth detecting (absolute)", -0.5, 0.5, 0.02, 0.005, key="ex_pc", format="%.3f")
            power = p3.select_slider("Power", [0.7, 0.8, 0.9, 0.95], 0.8, key="ex_pp")
            need = sx.sample_size_proportions(base, change, power=power)
        else:
            std = p1.number_input("Standard deviation", 0.0001, 1e9, 10.0, key="ex_ps")
            change = p2.number_input("Smallest change worth detecting", -1e9, 1e9, 2.0, key="ex_pm")
            power = p3.select_slider("Power", [0.7, 0.8, 0.9, 0.95], 0.8, key="ex_pp2")
            need = sx.sample_size_means(std, change, power=power)
        st.metric("Rows needed per group", f"{need:,}", help="At the 5% significance level.")
    except ValueError as e:
        ui.note(str(e), "high")


def _bayes(df: pd.DataFrame) -> None:
    st.caption("Instead of a p-value: how likely is it that the new version is better, and what would a wrong choice cost?")
    two = [c for c in df.columns if df[c].nunique() == 2]
    outcomes = [c for c in df.select_dtypes("number").columns] + [c for c in two if c not in df.select_dtypes("number").columns]
    if not two or not outcomes:
        ui.note("This needs a column with exactly two groups and an outcome column.", "low")
        return
    c1, c2, c3 = st.columns(3)
    group = c1.selectbox("Group column (two values)", two, key="ex_yg")
    options = sorted(df[group].dropna().astype(str).unique())
    control = c2.selectbox("Control group", options, key="ex_yc")
    outcome = c3.selectbox("Outcome", [c for c in outcomes if c != group], key="ex_yo")
    success = None
    if df[outcome].nunique() == 2:
        success = st.selectbox("Count as success", sorted(df[outcome].dropna().astype(str).unique()), index=1, key="ex_ys")
    result = _run(bayes.bayesian_ab, df, group, outcome, control, success=success)
    if result:
        ui.note(result["summary"], "ok" if max(result["probability_b_better"], 1 - result["probability_b_better"]) >= 0.95 else "low")
        ui.stats([("Chance the variant is better", f"{result['probability_b_better']:.1%}"),
                  ("Likely difference", f"{result['difference']:+.4g}"),
                  ("Likely relative lift", f"{result['lift']:+.1%}"),
                  ("Expected loss if wrong", f"{result['expected_loss_b']:.4g}")])


def _screen(df: pd.DataFrame) -> None:
    target = st.selectbox("Find what relates to", df.columns, index=len(df.columns) - 1, key="ex_st")
    if st.button("Screen all columns", key="ex_sgo", type="primary"):
        try:
            st.session_state.ex_screen = (target, sx.screen_associations(df, target))
        except ValueError as e:
            ui.note(str(e), "high")
            return
    cached = st.session_state.get("ex_screen")
    if not cached or cached[0] != target:
        return
    table = cached[1]
    related = table[table["related"]]
    ui.note(f"{len(related)} of {len(table)} columns are related to '{target}' after correcting for testing many columns at once."
            if len(related) else f"No column is reliably related to '{target}' once many tests are accounted for.",
            "ok" if len(related) else "low")
    top = table.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["strength"], y=top["column"], orientation="h",
                           marker_color=["#2e7d4f" if r else "#9a9a9a" for r in top["related"]]))
    fig.update_layout(title="Strength of the relationship (0 to 1)", xaxis_title="Effect size")
    ui.chart(fig, 120 + 28 * len(top))
    ui.table(table.round(5), hide_index=True)
    ui.download_csv(table, "associations.csv", "Download results")


def _trustworthy(df: pd.DataFrame) -> None:
    st.caption(t("Checks the experiment is sound before reading the result: the split of users, the effect with less noise, whether it holds over time and where it differs."))
    numeric = list(df.select_dtypes("number").columns)
    groups = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 10]
    if not groups or not numeric:
        ui.note("This needs a column with the group of each row (2 to 10 values) and a numeric outcome (use 0 and 1 for conversions).", "low")
        return
    c1, c2, c3 = st.columns(3)
    group = c1.selectbox("Group", groups, key="xq_group")
    control = c2.selectbox("Control group", sorted(df[group].dropna().astype(str).unique()), key="xq_control")
    outcome = c3.selectbox("Outcome", [c for c in numeric if c != group], key="xq_outcome")
    d1, d2, d3 = st.columns(3)
    pre = d1.selectbox("Same measure before the experiment (optional)", ["(none)"] + [c for c in numeric if c not in (group, outcome)], key="xq_pre")
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    date = d2.selectbox("Date (optional)", ["(none)"] + dates, key="xq_date")
    segs = d3.multiselect("Look for different effects by", [c for c in df.columns if c not in (group, outcome) and 2 <= df[c].nunique() <= 30 and not pd.api.types.is_float_dtype(df[c])], key="xq_segs")
    pre = None if pre.startswith("(") else pre
    date = None if date.startswith("(") else date
    signature = (group, control, outcome, pre, date, tuple(segs))
    if st.button("Analyse the experiment", key="xq_go", type="primary"):
        try:
            st.session_state["xq_res"] = (signature, exq.analyze_experiment(df, group, outcome, control, pre, date, segs or None))
        except ValueError as e:
            st.session_state.pop("xq_res", None)
            ui.note(str(e), "high")
    cached = st.session_state.get("xq_res")
    if not cached or cached[0] != signature:
        return
    res = cached[1]
    ui.note(res["verdict"], "high" if res["srm"]["broken"] else "ok")
    ui.chart(res["srm"]["figure"], 260)
    ui.table(res["table"].round(4))
    if len(res["table"]) and "figure" in res:
        ui.chart(res["figure"], 260)
    for note in res["notes"]:
        ui.note(note, "medium")
    if "cuped" in res:
        st.subheader(t("Less noise with the earlier measure"))
        ui.note(res["cuped"]["summary"], "accent")
        ui.table(res["cuped"]["table"].round(4))
    if "over_time" in res:
        st.subheader(t("Over time"))
        ui.chart(res["over_time"]["figure"], 320)
    if "segments" in res:
        st.subheader(t("By segment"))
        ui.table(res["segments"]["table"].round(4))


def render(df: pd.DataFrame) -> None:
    tabs = st.tabs(["Compare groups", "Two categories", "Correlation", "A/B test", "A/B (Bayesian)", "What relates to...", "Is the test sound?"])
    for tab, body in zip(tabs, (_groups, _association, _correlation, _ab, _bayes, _screen, _trustworthy)):
        with tab:
            body(df)
