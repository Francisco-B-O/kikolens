"""Pruebas de humo del dashboard con el arnés headless de Streamlit."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import kikolens.dashboard.app as dashboard_app
from kikolens.pipeline.project import load_project

APP = str(Path(dashboard_app.__file__))
PAGES = ["Overview", "Autopilot", "Explore", "Quality and cleaning", "Transform", "Patterns", "Statistics", "Ask", "Models",
         "Model lab", "Neural network", "Predict", "Time series", "Segments", "Report", "Project",
         "Panel", "Export", "Monitoring", "Text", "Experiments", "Anomalies", "Dimensions", "Compare data",
         "Survival", "Customers", "Impact", "Feature ideas", "Why it changed", "Regression", "Model health", "Privacy", "Match records", "Distribution fit",
         "Home", "Data story", "Chart gallery", "Analyst", "Sequential tests", "Segment explorer", "Recommendations", "Feature selection",
         "Forecast with drivers", "Run history", "Synthetic data", "Causal effect", "Simulation", "Marketing mix", "Classical series", "Model governance", "Network", "Geography", "Process control", "Inventory", "Survey", "Concept lab"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    n = 160
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "x": rng.normal(size=n),
        "z": rng.normal(size=n),
        "group": rng.choice(["a", "b", "c"], n),
        "price": [f"{v:,.0f}" for v in rng.integers(100, 5000, n)],
    })
    df["y"] = 3 * df["x"] + 20 + np.arange(n) * 0.05
    df.loc[::17, "z"] = np.nan
    return df


def open_page(data, page, **state):
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["df"] = data
    at.session_state["original_df"] = data.copy()
    at.session_state["filename"] = "sample.csv"
    at.session_state["nav"] = page
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]


def test_welcome_screen_without_data():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert_clean(at)
    assert any("KikoLens" in m.value for m in at.markdown)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(data, page):
    assert_clean(open_page(data, page))


def test_models_train_button(data):
    at = open_page(data, "Models")
    at.selectbox(key="aml_target").set_value("y")
    at.button(key="aml_train").click().run()
    assert_clean(at)
    assert "automl_res" in at.session_state


def test_autopilot_runs_and_can_apply_its_cleaning(data):
    at = open_page(data, "Autopilot")
    at.button(key="pilot_go").click().run()
    assert_clean(at)
    res = at.session_state["pilot_res"]
    assert res["target"] and len(res["recipe"]) >= 1
    at.button(key="pilot_apply").click().run()
    assert_clean(at)
    assert len(at.session_state["recipe"]) == len(res["recipe"]) and "pilot_res" not in at.session_state


def test_neural_train_button(data):
    at = open_page(data, "Neural network")
    at.selectbox(key="nn_target").set_value("y")
    at.slider(key="nn_epochs").set_value(40)
    at.button(key="nn_train").click().run()
    assert_clean(at)
    assert at.session_state["nn_res"]["task_type"] == "regression"


def test_neural_rejects_bad_layers(data):
    at = open_page(data, "Neural network")
    at.selectbox(key="nn_preset").set_value("Custom")
    at.run()
    at.text_input(key="nn_layers").set_value("abc")
    at.button(key="nn_train").click().run()
    assert_clean(at)
    assert "nn_res" not in at.session_state


def test_time_series_analysis(data):
    at = open_page(data, "Time series")
    at.selectbox(key="ts_value").set_value("y")
    at.button(key="ts_go").click().run()
    assert_clean(at)
    assert at.session_state["ts_res"]["fc"]["forecast"].shape[0] == 12


def test_auto_clean_then_undo_and_reset(data):
    at = open_page(data, "Quality and cleaning")
    at.button(key="q_auto").click().run()
    assert_clean(at)
    assert pd.api.types.is_numeric_dtype(at.session_state["df"]["price"])
    assert len(at.session_state["history"]) == 1
    at.button(key="q_undo").click().run()
    assert not pd.api.types.is_numeric_dtype(at.session_state["df"]["price"])
    at.button(key="q_auto").click().run()
    at.button(key="q_reset").click().run()
    assert_clean(at)
    assert at.session_state["history"] == []


def test_manual_sql_and_blocked_sql(data):
    at = open_page(data, "Ask")
    at.button(key="sql_run").click().run()
    assert_clean(at)
    at.text_area(key="sql_manual").set_value("DROP TABLE df")
    at.button(key="sql_run").click().run()
    assert_clean(at)
    assert any("Security" in m.value for m in at.markdown)


def test_report_builds(data):
    at = open_page(data, "Report")
    at.button(key="rp_go").click().run()
    assert_clean(at)
    assert "Data quality" in at.session_state["rp_html"]


def test_segments_run(data):
    at = open_page(data, "Segments")
    at.button(key="cl_go").click().run()
    assert_clean(at)
    assert "cluster_res" in at.session_state


@pytest.mark.parametrize("page,prefix,train,target_key", [
    ("Models", "aml", "aml_train", "aml_target"),
    ("Neural network", "nn", "nn_train", "nn_target"),
])
def test_simulate_and_explain_after_training(data, page, prefix, train, target_key):
    at = open_page(data, page)
    at.selectbox(key=target_key).set_value("y")
    at.button(key=train).click().run()
    at.button(key=f"{prefix}_simulate").click().run()
    assert_clean(at)
    assert any(m.label == "Predicted outcome" for m in at.metric)
    at.button(key=f"{prefix}_explain").click().run()
    assert_clean(at)


def test_simulator_with_categorical_target_and_features(data):
    at = open_page(data.assign(label=np.where(data["x"] > 0, "up", "down")), "Neural network")
    at.selectbox(key="nn_target").set_value("label")
    at.slider(key="nn_epochs").set_value(40)
    at.button(key="nn_train").click().run()
    assert at.session_state["nn_res"]["task_type"] == "classification"
    at.button(key="nn_simulate").click().run()
    assert_clean(at)


def test_explore_every_chart_type(data):
    for kind in ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin", "Density heatmap"]:
        at = open_page(data, "Explore")
        at.selectbox(key="ex_kind").set_value(kind)
        at.run()
        assert_clean(at)


def test_style_chart_never_leaves_title_undefined():
    import plotly.express as px

    from kikolens.dashboard.theme import style_chart

    fig = style_chart(px.line(pd.DataFrame({"a": [1, 2, 3], "b": [3, 1, 2]}), x="a", y="b"))
    assert fig.layout.title.text == ""


def test_model_lab_runs_end_to_end(data):
    at = open_page(data, "Model lab")
    at.selectbox(key="lab_target").set_value("y")
    at.run()
    at.multiselect(key="lab_models_regression").set_value(["Baseline (mean)", "Ridge", "Decision Tree"])
    at.slider(key="lab_folds").set_value(3)
    at.checkbox(key="lab_tune").set_value(False)
    at.button(key="lab_run").click().run()
    assert_clean(at)
    assert at.session_state["lab_res"]["task_type"] == "regression"
    at.button(key="lab_predict").click().run()
    assert_clean(at)
    assert any(m.label == "Prediction" for m in at.metric)
    at.button(key="lab_shap_row").click().run()
    assert_clean(at)
    assert at.session_state["lab_shap"][1]["table"].shape[0] >= 2
    at.button(key="lab_pair").click().run()
    assert_clean(at)
    assert "lab_pair_res" in at.session_state


def test_model_lab_reports_bad_target(data):
    at = open_page(data, "Model lab")
    at.selectbox(key="lab_target").set_value("date")
    at.run()
    at.button(key="lab_run").click().run()
    assert_clean(at)
    assert "lab_res" not in at.session_state


def test_transform_records_steps_and_undo_removes_them(data):
    at = open_page(data, "Transform")
    assert_clean(at)
    at.text_input(key="tf_a_name").set_value("double")
    at.text_input(key="tf_a_expr").set_value("x * 2")
    at.button(key="tf_a_go").click().run()
    assert_clean(at)
    assert "double" in at.session_state["df"].columns
    assert [s.op for s in at.session_state["recipe"].steps] == ["add_column"]
    at.button(key="tf_undo").click().run()
    assert "double" not in at.session_state["df"].columns
    assert len(at.session_state["recipe"]) == 0


def test_transform_shows_error_for_bad_formula(data):
    at = open_page(data, "Transform")
    at.text_input(key="tf_a_name").set_value("bad")
    at.text_input(key="tf_a_expr").set_value("__import__('os')")
    at.button(key="tf_a_go").click().run()
    assert_clean(at)
    assert "bad" not in at.session_state["df"].columns
    assert any("allowed" in m.value for m in at.markdown)


def test_auto_clean_is_one_undo_but_many_recipe_steps(data):
    at = open_page(data, "Quality and cleaning")
    at.button(key="q_auto").click().run()
    assert len(at.session_state["recipe"]) >= 1
    at.button(key="q_undo").click().run()
    assert len(at.session_state["recipe"]) == 0


def test_project_page_prepares_file(data):
    at = open_page(data, "Project")
    assert_clean(at)
    at.button(key="pj_build").click().run()
    assert_clean(at)
    assert load_project(at.session_state["pj_blob"])["manifest"]["rows"] == len(data)


def test_predict_without_a_model_explains_what_to_do(data):
    at = open_page(data, "Predict")
    assert_clean(at)
    assert any("Train a model first" in m.value for m in at.markdown)


def test_predict_uses_a_model_trained_in_the_session(data, tmp_path, monkeypatch):
    monkeypatch.setenv("KIKOLENS_HOME", str(tmp_path))
    from kikolens.analyze.automl import run_automl_analysis
    frame = data.drop(columns=["date"])
    at = open_page(frame, "Predict", automl_res=run_automl_analysis(frame, "y"))
    assert_clean(at)
    at.button(key="pd_go").click().run()
    assert_clean(at)
    out = at.session_state["pd_result"][0]
    assert "predicted_y" in out.columns and len(out) == len(frame)
    at.button(key="pd_one").click().run()
    assert_clean(at)
    at.button(key="pd_save").click().run()
    assert_clean(at)
    assert list((tmp_path / "models").glob("*.klmodel"))


def test_anomalies_find_and_remove(data):
    at = open_page(data, "Anomalies")
    assert_clean(at)
    at.button(key="an_go").click().run()
    assert_clean(at)
    flagged = at.session_state["an_res"]["n_flagged"]
    assert flagged >= 1
    at.button(key="an_drop").click().run()
    assert_clean(at)
    assert len(at.session_state["df"]) == len(data) - flagged


def test_experiments_tabs_render_without_errors(data):
    at = open_page(data, "Experiments")
    assert_clean(at)
    at.button(key="ex_sgo").click().run()
    assert_clean(at)
    assert at.session_state["ex_screen"][1]["column"].tolist()


def test_dimensions_and_compare_render(data):
    assert_clean(open_page(data, "Dimensions"))
    at = open_page(data, "Compare data")
    assert_clean(at)
    assert any("PSI" in c.value for c in at.caption)


def test_sidebar_groups_follow_the_open_section(data):
    at = open_page(data, "Anomalies")
    assert at.session_state["nav_group"] == "Analysis"
    at = open_page(data, "Project")
    assert at.session_state["nav_group"] == "Share"


def test_panel_suggest_add_and_export(data):
    at = open_page(data, "Panel")
    assert_clean(at)
    at.button(key="pn_suggest").click().run()
    assert_clean(at)
    assert len(at.session_state["panel"].tiles) >= 4
    at.button(key="pn_add").click().run()
    assert_clean(at)


def test_quality_smart_fill_and_extra_checks(data):
    at = open_page(data, "Quality and cleaning")
    assert_clean(at)
    at.selectbox(key="q_op").set_value("Smart fill (KNN, iterative, by group)").run()
    at.button(key="q_apply_sf").click().run()
    assert_clean(at)
    assert at.session_state["df"]["z"].isna().sum() == 0
    assert [s.op for s in at.session_state["recipe"].steps] == ["impute_advanced"]
    at.button(key="q_rl_suggest").click().run()
    assert_clean(at)
    assert len(at.session_state["rules"].rules) > 0


def test_time_series_compare_models_and_groups(data):
    at = open_page(data, "Time series")
    at.selectbox(key="ts_value").set_value("y")
    at.button(key="ts_go").click().run()
    assert_clean(at)
    at.button(key="ts_compare").click().run()
    assert_clean(at)
    assert at.session_state["ts_models"][1]["best"]
    at.button(key="ts_by_group").click().run()
    assert_clean(at)


def test_pasting_a_table_loads_it():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox(key="source_mode").set_value("Paste").run()
    at.text_area(key="paste_text").set_value("a,b\n1,2\n3,4\n5,6").run()
    assert_clean(at)
    assert at.session_state["df"].shape == (3, 2)


def test_export_page_offers_every_route(data):
    at = open_page(data, "Export")
    assert_clean(at)
    assert [t.label for t in at.tabs] == ["Files", "Excel workbook", "Database", "Data dictionary", "Notebook"]


@pytest.fixture(scope="module")
def behaviour():
    rng = np.random.default_rng(8)
    n = 300
    df = pd.DataFrame({"user": rng.integers(0, 60, n), "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 150, n), "D"),
                       "amount": rng.uniform(5, 90, n), "plan": rng.choice(["free", "pro"], n)})
    df["tenure"] = rng.exponential(np.where(df["plan"] == "pro", 20, 10))
    df["churned"] = rng.integers(0, 2, n)
    return df


def test_survival_page_draws_curves(behaviour):
    at = open_page(behaviour, "Survival")
    at.selectbox(key="sv_d").set_value("tenure")
    at.selectbox(key="sv_e").set_value("churned")
    at.run()
    at.selectbox(key="sv_g").set_value("plan")
    at.button(key="sv_go").click().run()
    assert_clean(at)
    assert set(at.session_state["sv_res"][1]["curves"]) == {"free", "pro"}


def test_customers_tabs_run(behaviour):
    at = open_page(behaviour, "Customers")
    at.selectbox(key="cu_cu").set_value("user")
    at.button(key="cu_cohort_go").click().run()
    assert_clean(at)
    assert "cu_cohort" in at.session_state
    at.selectbox(key="cu_rc").set_value("user")
    at.button(key="cu_rfm_go").click().run()
    assert_clean(at)
    assert "cu_rfm" in at.session_state


def test_impact_estimates_with_a_control(behaviour):
    daily = behaviour.groupby("date").agg(sales=("amount", "sum"), visits=("user", "size")).reset_index()
    at = open_page(daily, "Impact")
    at.selectbox(key="im_v").set_value("sales")
    at.multiselect(key="im_c").set_value(["visits"])
    at.button(key="im_go").click().run()
    assert_clean(at)


def test_feature_ideas_page_runs():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.uniform(100, 1000, 400), "b": rng.uniform(10, 100, 400)})
    df["y"] = 3 * df["a"] / df["b"] + rng.normal(0, 0.1, 400)
    at = open_page(df, "Feature ideas")
    at.button(key="af_go").click().run()
    assert_clean(at)
    assert at.session_state["af_res"][1]["suggestions"]
    at.button(key="af_apply").click().run()
    assert_clean(at)
    assert any(s.op == "add_column" for s in at.session_state["recipe"].steps)


def test_lab_uncertainty_counterfactual_and_fairness_tabs():
    rng = np.random.default_rng(2)
    n = 500
    df = pd.DataFrame({"income": rng.normal(50, 15, n), "debt": rng.normal(20, 8, n), "region": rng.choice(["n", "s"], n)})
    df["approved"] = np.where(0.08 * df["income"] - 0.1 * df["debt"] + rng.normal(0, 1, n) > 2, "yes", "no")
    at = open_page(df, "Model lab")
    at.selectbox(key="lab_target").set_value("approved")
    at.run()
    at.multiselect(key="lab_models_classification").set_value(["Baseline (majority)", "Logistic Regression"])
    at.slider(key="lab_folds").set_value(3)
    at.checkbox(key="lab_tune").set_value(False)
    at.button(key="lab_run").click().run()
    assert_clean(at)
    at.button(key="lab_cf_go").click().run()
    assert_clean(at)
    assert "lab_cf" in at.session_state


def test_why_it_changed_explains_a_drop(behaviour):
    at = open_page(behaviour, "Why it changed")
    at.selectbox(key="dr_m").set_value("amount")
    at.button(key="dr_go").click().run()
    assert_clean(at)
    assert at.session_state["dr_res"]["tables"]


def test_regression_page_fits_and_explains(data):
    at = open_page(data, "Regression")
    at.selectbox(key="rg_t").set_value("y")
    at.run()
    at.multiselect(key="rg_f").set_value(["x", "z", "date"])
    at.button(key="rg_go").click().run()
    assert_clean(at)
    res = at.session_state["rg_res"][1]
    assert res["kind"] == "ols" and res["model"]["r2"] > 0.9


def test_model_health_scores_newer_data(data):
    from kikolens.analyze.modellab import run_model_lab
    from kikolens.analyze.predict import build_bundle
    res = run_model_lab(data.drop(columns=["date", "price"]), "y", models=["Ridge"], folds=3, tune=False)
    bundle = build_bundle(res, data.drop(columns=["date", "price"]), name="m")
    at = open_page(data, "Model health", restored_models={"m": bundle})
    assert_clean(at)
    at.button(key="mh_go").click().run()
    assert_clean(at)
    assert at.session_state["mh_res"]["table"]["rows"].sum() == len(data)


def test_anomalies_has_a_tab_for_unusual_periods(data):
    at = open_page(data, "Anomalies")
    assert_clean(at)
    assert [t.label for t in at.tabs][:2] == ["Unusual rows", "Unusual periods"]


def test_privacy_page_masks_personal_columns_and_records_the_steps():
    df = pd.DataFrame({"email": [f"u{i}@example.com" for i in range(60)], "age": range(20, 80), "city": ["A", "B"] * 30})
    at = open_page(df, "Privacy")
    assert_clean(at)
    at.selectbox(key="pv_a_email").set_value("hash")
    at.run()                                        # el botón se habilita al elegir una acción
    at.button(key="pv_go").click().run()
    assert_clean(at)
    assert not at.session_state["df"]["email"].str.contains("@").any()
    assert [s.op for s in at.session_state["recipe"].steps] == ["hash_column"]


def test_export_dictionary_and_notebook_tabs_render(data):
    at = open_page(data, "Export")
    assert_clean(at)
    assert [t.label for t in at.tabs][-2:] == ["Data dictionary", "Notebook"]


def test_section_finder_jumps_to_any_section(data):
    at = open_page(data, "Overview")
    at.selectbox(key="jump").set_value("Survival").run()
    assert_clean(at)
    assert at.session_state["nav"] == "Survival" and at.session_state["nav_group"] == "Analysis"
    assert at.session_state["jump"] == ""
    at.selectbox(key="jump").set_value("Model lab").run()
    assert at.session_state["nav_group"] == "Models"


def test_distribution_page_fits_a_column(data):
    at = open_page(data, "Distribution fit")
    at.selectbox(key="di_c").set_value("x")
    at.run()
    assert_clean(at)
    assert any("fits best" in m.value for m in at.markdown)


def test_new_analysis_pages_work_on_suitable_data(data):
    # selección de variables
    at = open_page(data, "Feature selection")
    at.selectbox(key="fs_t").set_value("y")
    at.run()
    at.button(key="fs_go").click().run()
    assert_clean(at)
    assert "fs_res" in at.session_state


def test_sequential_recommendations_synthetic_and_history_pages():
    rng = np.random.default_rng(0)
    n = 1200
    arm = np.tile(["control", "new"], n // 2)
    ab = pd.DataFrame({"arm": arm, "won": np.where(arm == "control", rng.random(n) < 0.1, rng.random(n) < 0.16).astype(int)})
    at = open_page(ab, "Sequential tests")
    assert_clean(at)
    assert any("Always-valid" in str(m.label) or "Decision" in str(m.label) for m in at.metric) or any("kl-stat" in m.value for m in at.markdown)

    rows = [(f"u{u}", f"i{(u % 3) * 10 + int(rng.integers(0, 10))}") for u in range(150) for _ in range(6)]
    at = open_page(pd.DataFrame(rows, columns=["user", "item"]).drop_duplicates(), "Recommendations")
    assert_clean(at)

    people = pd.DataFrame({"age": rng.integers(20, 70, 300), "income": rng.normal(2000, 400, 300), "city": rng.choice(["A", "B"], 300)})
    at = open_page(people, "Synthetic data")
    at.button(key="sy_go").click().run()
    assert_clean(at)
    assert len(at.session_state["sy_res"][0]["data"]) == 300

    at = open_page(people, "Run history")
    assert_clean(at)


def test_causal_effect_page_estimates_and_remembers(data):
    rng = np.random.default_rng(3)
    n = 500
    age = rng.normal(size=n)
    treated = (rng.random(n) < 1 / (1 + np.exp(-age))).astype(int)
    df = pd.DataFrame({"treated": treated, "outcome": 2 * treated + 3 * age + rng.normal(size=n), "age": age})
    at = open_page(df, "Causal effect")
    assert_clean(at)
    at.button(key="cz_res_go").click().run()
    assert_clean(at)
    assert "cz_res" in at.session_state and "Treatment 'treated'" in at.session_state["cz_res"][1]["summary"]


def test_panel_cross_filter(data):
    at = open_page(data, "Panel")
    at.button(key="pn_suggest").click().run()
    assert_clean(at)
    chart = next(t for t in at.session_state["panel"].tiles if t["type"] == "chart" and t["kind"] == "bar")
    column = chart["x"]
    value = str(data[column].iloc[0])
    at.session_state["pn_cross"] = {chart["id"]: [{"column": column, "op": "in", "value": [value]}]}
    at.run()
    assert_clean(at)
    shown = int((data[column].astype(str) == value).sum())
    assert any(f"Showing {shown:,} of {len(data):,} rows" in str(c.value) for c in at.caption)
    at.button(key="pn_clear_cross").click().run()
    assert_clean(at)
    assert at.session_state["pn_cross"] == {}


def test_classical_series_page_runs_every_tab():
    from test_classic import monthly
    y, _ = monthly()
    df = pd.DataFrame({"month": y.index, "sales": y.values})
    at = open_page(df, "Classical series")
    assert_clean(at)
    for key, state in (("cs_stat_go", "cs_stat"), ("cs_acf_go", "cs_acf"), ("cs_stl_go", "cs_stl"), ("cs_arima_go", "cs_arima")):
        at.button(key=key).click().run()
        assert_clean(at)
        assert state in at.session_state, state
    assert at.session_state["cs_arima"][1]["order"][1] in (0, 1) and at.session_state["cs_stl"][1]["period"] in range(2, 400)


def test_text_page_new_tabs_run():
    from test_textmining import corpus
    texts, _ = corpus(150)
    df = pd.DataFrame({"review": texts, "when": pd.date_range("2024-01-01", periods=150, freq="D")})
    at = open_page(df, "Text")
    assert_clean(at)
    assert [t.label for t in at.tabs][6:] == ["Topics", "Duplicates and similar", "Patterns", "Phrases", "Readability"]
    at.button(key="tx_topics_go").click().run()
    assert_clean(at)
    assert at.session_state["tx_topics"][1]["k"] in range(2, 9)
    at.button(key="tx_dups_go").click().run()
    at.button(key="tx_patterns_go").click().run()
    assert_clean(at)
    assert "tx_dups" in at.session_state and "tx_patterns" in at.session_state
    at.selectbox(key="tx_tdate").set_value("when").run()
    at.button(key="tx_topics_go").click().run()
    assert_clean(at)
    assert "over_time" in at.session_state["tx_topics"][1]


def test_network_page_runs_every_tab():
    from test_network import planted
    edges, _ = planted(20, 3)
    at = open_page(edges, "Network")
    assert_clean(at)
    at.button(key="nw_res_go").click().run()
    assert_clean(at)
    assert at.session_state["nw_res"][1]["nodes"] >= 55
    for key in ("nw_path_go", "nw_links_go", "nw_rob_go"):
        at.button(key=key).click().run()
        assert_clean(at)
    assert all(k in at.session_state for k in ("nw_path", "nw_links", "nw_rob"))


def test_geography_page_runs_every_tab():
    from test_geo import two_cities
    d = two_cities(300)
    at = open_page(d, "Geography")
    assert_clean(at)
    at.button(key="geo_dist_go").click().run()
    at.button(key="geo_zones_go").click().run()
    at.button(key="geo_spatial_go").click().run()
    at.button(key="geo_grid_go").click().run()
    assert_clean(at)
    assert at.session_state["geo_zones"][1]["zones"] == 2 and at.session_state["geo_spatial"][1]["moran"] != 0
    assert at.session_state["geo_dist"][1]["median_km"] >= 0 and at.session_state["geo_grid"][1]["cells"] > 1


def test_experiments_page_checks_that_the_test_is_sound():
    from test_experiment import trial
    d = trial(2500)
    at = open_page(d, "Experiments")
    assert_clean(at)
    assert [t.label for t in at.tabs][-1] == "Is the test sound?"
    at.selectbox(key="xq_group").set_value("g").run()
    at.selectbox(key="xq_control").set_value("control").run()
    at.selectbox(key="xq_outcome").set_value("y").run()
    at.selectbox(key="xq_pre").set_value("pre").run()
    at.button(key="xq_go").click().run()
    assert_clean(at)
    res = at.session_state["xq_res"][1]
    assert res["verdict"].startswith("new beats control") and "cuped" in res


def test_survey_page_runs_every_tab():
    from test_survey import likert_items
    d = likert_items(400)
    d["overall"] = np.clip(np.round(d.mean(axis=1) + np.random.default_rng(1).normal(0, 0.4, 400)), 1, 5)
    d["recommend"] = np.random.default_rng(2).integers(0, 11, 400)
    d["region"] = np.random.default_rng(3).choice(["n", "s"], 400)
    at = open_page(d, "Survey")
    assert_clean(at)
    for key in ("sv_likert_go", "sr_res_go", "sn_res_go", "sc_res_go", "sk_res_go", "sg_res_go", "sq_res_go"):
        at.button(key=key).click().run()
        assert_clean(at)
    for state in ("sv_likert", "sr_res", "sn_res", "sc_res", "sk_res", "sg_res", "sq_res"):
        assert state in at.session_state, state
    assert at.session_state["sr_res"][1]["alpha"] > 0.5


def test_concept_lab_works_with_data_and_without():
    at = open_page(pd.DataFrame({"a": [1, 2, 3]}), "Concept lab")
    assert_clean(at)
    at.selectbox(key="cl_concept").set_value("power").run()
    assert_clean(at)
    assert at.session_state["cl_result"][1]["needed_per_group"] in (63, 64, 65)
    at.slider(key="cl_power_effect").set_value(1.0).run()
    assert_clean(at)
    assert at.session_state["cl_result"][1]["needed_per_group"] < 20
    again = AppTest.from_file(APP, default_timeout=60).run()
    again.button(key="open_lab").click().run()
    assert_clean(again)
    assert again.session_state["cl_result"][1]["concept"] == "clt"
    again.button(key="cl_new_clt").click().run()
    assert_clean(again)
    assert again.session_state["cl_seed_clt"] == 1
    again.button(key="leave_lab").click().run()
    assert not again.session_state["learn_only"]
