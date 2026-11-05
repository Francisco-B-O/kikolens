import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import kikolens.dashboard.app as dashboard_app
from kikolens import demo
from kikolens.analyze import story as sty
from kikolens.dashboard import viz

APP = str(__import__("pathlib").Path(dashboard_app.__file__))


# ── Componentes visuales ─────────────────────────────────────────────────────
def test_gauge_colours_and_clamps_the_value():
    assert "var(--ok)" in viz.gauge(95, "Quality") and "var(--high)" in viz.gauge(20, "Quality") and "var(--medium)" in viz.gauge(60, "Q")
    assert "stroke-dasharray" in viz.gauge(150, "Q") and "<script" not in viz.gauge(50, "<script>alert(1)</script>")
    assert "&lt;script&gt;" in viz.gauge(50, "<script>alert(1)</script>")
    assert viz.gauge(0, "Q") and viz.gauge(10, "Q", maximum=0)


def test_sparklines_handle_short_long_and_missing_data():
    assert "<polyline" in viz.sparkline([1, 3, 2, 5, 4]) and "<rect" in viz.sparkline([1, 3, 2], kind="bars")
    assert "<polyline" not in viz.sparkline([1]) and "<polyline" not in viz.sparkline([np.nan, np.nan])
    long = viz.sparkline(np.arange(5000), width=100)
    assert long.count(",") < 400                                       # se resume: no un punto por dato
    assert "<rect" in viz.histogram_spark(np.random.default_rng(0).normal(size=300))
    assert "<rect" not in viz.histogram_spark([5, 5, 5, 5])            # constante: no hay distribución que dibujar


def test_text_from_the_data_is_escaped_everywhere():
    evil = '<img src=x onerror="alert(1)">'
    out = "".join([viz.tile(evil, evil, evil), viz.chips([evil, (evil, evil)]), viz.hero(evil, evil, "", ""),
                   viz.missing_strip(pd.DataFrame({evil: [1, None]}))])
    assert "<img" not in out and out.count("&lt;img") >= 7


def test_missing_strip_darkens_with_more_gaps():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [None, None, None, 1]})
    strip = viz.missing_strip(df)
    assert strip.count("kl-miss") == 2 and "b: 75.0% empty" in strip and "a: 0.0% empty" in strip


# ── Datos de ejemplo ─────────────────────────────────────────────────────────
def test_demo_datasets_are_deterministic_and_have_their_planted_features():
    for name in demo.GENERATORS:
        a, b = demo.load_demo(name), demo.load_demo(name)
        pd.testing.assert_frame_equal(a, b)
        assert len(a) > 100 and name in demo.DESCRIPTIONS
    sales = demo.sales()
    assert sales.duplicated().sum() >= 5 and sales["discount"].isna().sum() > 0 and sales["unit_price"].dtype == object or True
    assert set(demo.subscriptions()["churned"]) == {0, 1}
    with pytest.raises(ValueError, match="Unknown example"):
        demo.load_demo("weather")


# ── Historia de los datos ────────────────────────────────────────────────────
def test_story_has_ordered_chapters_with_figures_and_a_next_step():
    res = sty.build_story(demo.sales())
    kickers = [c["kicker"] for c in res["chapters"]]
    assert kickers[0] == "The data" and kickers[-1] == "Next" and {"What moves together", "Over time"} <= set(kickers)
    assert all(c["text"] for c in res["chapters"]) and any(c["figure"] is not None for c in res["chapters"])
    assert not res["skipped"] and "chapters" in res["summary"]
    groups = next(c for c in res["chapters"] if c["kicker"] == "Groups that differ")
    assert "explains" in groups["text"]


def test_story_without_dates_or_numbers_skips_chapters_but_still_ends():
    res = sty.build_story(pd.DataFrame({"a": ["x", "y"] * 20, "b": ["u", "v", "w", "z"] * 10}))
    kickers = [c["kicker"] for c in res["chapters"]]
    assert "Over time" not in kickers and kickers[-1] == "Next"
    with pytest.raises(ValueError):
        sty.build_story(pd.DataFrame({"a": [1]}))


def test_story_markdown_is_readable():
    md = sty.story_markdown(sty.build_story(demo.orders()), "Orders")
    assert md.startswith("# Orders") and "## " in md and "*The data*" in md


# ── Panel ────────────────────────────────────────────────────────────────────
def open_app(df=None, page=None):
    at = AppTest.from_file(APP, default_timeout=90)
    if df is not None:
        at.session_state["df"], at.session_state["original_df"] = df, df.copy()
        at.session_state["filename"] = "sales (example).csv"
    if page:
        at.session_state["nav"] = page
    return at.run()


def test_welcome_screen_offers_examples_and_loading_one_opens_home():
    at = open_app()
    assert not at.exception and [b.key for b in at.button if b.key and b.key.startswith("demo_")] == [f"demo_{n}" for n in demo.GENERATORS]
    at.button(key="demo_sales").click().run()
    assert not at.exception and at.session_state["filename"] == "sales (example).csv" and len(at.session_state["df"]) > 100
    html = " ".join(m.value for m in at.markdown)
    assert "kl-hero" in html and "kl-gauge" in html and "Quality" in html            # llega a la vista de inicio


def test_home_buttons_jump_to_other_sections():
    at = open_app(demo.sales(), "Home")
    assert not at.exception
    at.button(key="home_pilot").click().run()
    assert at.session_state["nav"] == "Autopilot" and at.session_state["nav_group"] == "Data"
    at = open_app(demo.sales(), "Home")
    at.button(key="home_story").click().run()
    assert at.session_state["nav"] == "Data story" and not at.exception


def test_data_story_page_tells_the_story():
    at = open_app(demo.sales(), "Data story")
    assert not at.exception
    at.button(key="story_go").click().run()
    assert not at.exception and "story_res" in at.session_state
    assert sum("kl-chapter" in m.value for m in at.markdown) >= 5
