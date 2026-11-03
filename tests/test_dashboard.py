"""Pruebas de humo del dashboard con el arnés headless de Streamlit."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import kikolens.dashboard.app as dashboard_app

APP = str(Path(dashboard_app.__file__))
PAGES = ["Overview", "Explore", "Quality and cleaning", "Patterns", "Statistics", "Ask", "Models",
         "Neural network", "Time series", "Segments", "Report"]


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
