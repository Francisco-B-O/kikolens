import pandas as pd
import pytest

from kikolens.learn import glossary as gl
from kikolens.learn.glossary_data import TERMS
from kikolens.learn.guides_data import GUIDES


def test_every_term_is_complete_in_both_languages_and_keys_are_unique():
    keys = [row[0] for row in TERMS]
    assert len(keys) == len(set(keys)) and len(keys) >= 60
    for key, en, es, def_en, def_es, watch_en, watch_es, see in TERMS:
        for text in (en, es, def_en, def_es, watch_en, watch_es):
            assert isinstance(text, str) and text.strip(), key
        assert 30 <= len(def_en) <= 330 and 30 <= len(def_es) <= 380 and 15 <= len(watch_en) <= 220 and 15 <= len(watch_es) <= 260, key
        assert def_en != def_es and watch_en != watch_es or key in ("arima", "bootstrap"), key                       # la traducción no es una copia
        assert key not in see and all(k in keys for k in see), key
    names_en = [row[1].lower() for row in TERMS]
    assert len(names_en) == len(set(names_en))


def test_guides_cover_real_sections_and_only_use_known_terms():
    from kikolens.dashboard import app
    keys = {row[0] for row in TERMS}
    assert set(GUIDES) <= set(app.PAGES), set(GUIDES) - set(app.PAGES)                  # si se renombra una sección, esta prueba avisa
    assert len(GUIDES) >= 20
    for page, (reads_en, reads_es, watch_en, watch_es, terms) in GUIDES.items():
        assert all(len(t.strip()) > 40 for t in (reads_en, reads_es, watch_en, watch_es)), page
        assert terms and set(terms) <= keys and len(set(terms)) == len(terms), page


def test_lookups_work_by_key_and_by_name_in_both_languages():
    en, es = gl.get("p_value"), gl.get("p-valor", "es")
    assert en["key"] == es["key"] == "p_value" and en["term"] == "p-value" and es["term"] == "p-valor" and en["definition"] != es["definition"]
    assert gl.get("P-VALUE")["key"] == "p_value" and gl.get("confidence interval", "es")["key"] == "confidence_interval" and gl.get("intervalo de confianza", "es")["key"] == "confidence_interval"
    assert {x["key"] for x in en["see_also"]} >= {"null_hypothesis", "alpha"} and all(x["term"] for x in es["see_also"])
    with pytest.raises(ValueError, match="Did you mean: p-value"):
        gl.get("p-valu")
    with pytest.raises(ValueError, match="not in the glossary"):
        gl.get("zzzzzz")
    with pytest.raises(ValueError, match="language"):
        gl.get("p_value", "fr")


def test_search_ranks_names_before_definitions_and_handles_empty_queries():
    found = gl.search("power")
    assert found["key"].iloc[0] == "power" and isinstance(found, pd.DataFrame) and list(found.columns) == ["term", "key", "definition"]
    by_definition = gl.search("bell")
    assert len(by_definition) >= 2 and "normal_distribution" in set(by_definition["key"])
    assert len(gl.search("")) == len(TERMS) and gl.search("xyzxyz").empty
    assert gl.search("sobreajuste", "es")["key"].iloc[0] == "overfitting" and gl.glossary()["term"].str.lower().is_monotonic_increasing


def test_terms_in_finds_whole_words_only():
    text = "The p-value and the confidence interval of an A/B test, with a baseline."
    assert {"p_value", "confidence_interval", "ab_test", "baseline"} <= set(gl.terms_in(text))
    assert "power" not in gl.terms_in("empower the powerful")
    assert "intervalo" not in " ".join(gl.terms_in("el intervalo de confianza", "es")) and "confidence_interval" in gl.terms_in("el intervalo de confianza", "es")


def test_guide_returns_text_and_term_definitions_or_none():
    g = gl.guide("Experiments")
    assert g["reads"] and g["watch_out"] and {t["key"] for t in g["terms"]} >= {"p_value", "power"} and all(t["definition"] for t in g["terms"])
    assert gl.guide("Experiments", "es")["reads"] != g["reads"] and gl.guide("Home") is None and gl.guide("Nope") is None
    assert gl.pages_with_guide() == list(GUIDES)


def test_dashboard_shows_the_guide_where_there_is_one_and_the_glossary_page_searches():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from kikolens.dashboard import app as dashboard_app

    def open_page(page, language="English"):
        at = AppTest.from_file(str(Path(dashboard_app.__file__)), default_timeout=60)
        df = pd.DataFrame({"a": range(40), "b": range(40, 80), "c": list("xy") * 20})
        at.session_state["df"], at.session_state["original_df"], at.session_state["filename"], at.session_state["nav"] = df, df.copy(), "x.csv", page
        at.session_state["lang_choice"] = language
        return at.run()

    with_guide = open_page("Experiments")
    assert not with_guide.exception and any(e.label == "How to read this page" for e in with_guide.expander)
    without = open_page("Home")
    assert not without.exception and not any(e.label == "How to read this page" for e in without.expander)
    spanish = open_page("Experiments", "Español")
    assert not spanish.exception and any(e.label == "Cómo leer esta página" for e in spanish.expander)
    page = open_page("Glossary")
    assert not page.exception and len(page.expander) >= 70
    page.text_input(key="gl_query").set_value("overfitting").run()
    assert not page.exception and len(page.expander) >= 1 and any("overfitting" in e.label for e in page.expander)
    page.text_input(key="gl_query").set_value("qqqqqq").run()
    assert not page.exception and not page.expander


def test_python_api_and_terminal():
    import kikolens as kl
    from typer.testing import CliRunner

    from kikolens.cli.main import app
    assert len(kl.glossary()) == len(TERMS) and kl.glossary("p_value")["term"] == "p-value" and kl.glossary("p-valor", "es")["term"] == "p-valor"
    assert isinstance(kl.glossary("interval"), pd.DataFrame) and len(kl.glossary("interval")) >= 2
    with pytest.raises(ValueError):
        kl.glossary("qqqqqq")
    runner = CliRunner()
    out = runner.invoke(app, ["glossary", "power"])
    text = " ".join(out.output.split())
    assert out.exit_code == 0 and "statistical power" in text and "Watch out" in text and "See also" in text
    assert "Terms matching" in " ".join(runner.invoke(app, ["glossary", "interval"]).output.split())
    assert runner.invoke(app, ["glossary", "--lang", "es", "p-valor"]).exit_code == 0 and runner.invoke(app, ["glossary", "qqqqqq"]).exit_code == 1
    assert runner.invoke(app, ["glossary"]).exit_code == 0
