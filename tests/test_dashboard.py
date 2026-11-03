"""Pruebas de humo del dashboard con el arnés headless de Streamlit."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import kikolens.dashboard.app as dashboard_app
from kikolens.pipeline.project import load_project

APP = str(Path(dashboard_app.__file__))
PAGES = ["Overview", "Explore", "Quality and cleaning", "Transform", "Patterns", "Statistics", "Ask", "Models",
         "Model lab", "Neural network", "Predict", "Time series", "Segments", "Report", "Project",
         "Panel", "Monitoring", "Text", "Experiments", "Anomalies", "Dimensions", "Compare data"]


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
