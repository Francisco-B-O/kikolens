import ast
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import kikolens.dashboard.app as dashboard_app
from kikolens.dashboard import i18n
from kikolens.dashboard.es import ES

VIEWS = Path(dashboard_app.__file__).parent
APP = str(Path(dashboard_app.__file__))
WIDGETS = {"button", "selectbox", "multiselect", "checkbox", "slider", "select_slider", "radio", "text_input", "text_area",
           "number_input", "file_uploader", "download_button", "expander", "caption", "markdown", "metric", "date_input",
           "subheader", "header", "info", "warning", "success", "error", "spinner", "tabs"}


def _strings(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    elif isinstance(node, (ast.List, ast.Tuple)):
        for element in node.elts:
            yield from _strings(element)


def ui_strings() -> dict:
    """Los textos fijos que el panel muestra: títulos, subtítulos y la etiqueta de cada widget (primer argumento)."""
    found = {}
    for path in sorted(VIEWS.rglob("*.py")):
        if path.name in ("es.py", "i18n.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            texts = []
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("TITLE", "SUBTITLE"):
                texts = list(_strings(node.value))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
                owner = node.func.value
                owner_name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")
                if node.func.attr in WIDGETS or (owner_name == "ui" and node.func.attr in ("note", "label")):
                    texts = list(_strings(node.args[0]))
            for text in texts:
                if text.strip() and any(c.isalpha() for c in text) and not text.startswith("<"):
                    found[text] = path.name
    return found


def test_lookup_prefixes_bold_and_fallback():
    assert i18n.t("Model lab", "es") == "Laboratorio de modelos"
    assert i18n.t("#### Findings", "es") == "#### Hallazgos"
    assert i18n.t("**Which groups differ**", "es") == "**Qué grupos difieren**"
    assert i18n.t("Not a known string", "es") == "Not a known string"
    assert i18n.t("Model lab", "en") == "Model lab" and i18n.t(5, "es") == 5


def test_every_section_title_and_subtitle_is_translated():
    import kikolens.dashboard.views as views_pkg
    missing = []
    for path in sorted(Path(views_pkg.__file__).parent.glob("*.py")):
        if path.name.startswith("_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and node.targets[0].id in ("TITLE", "SUBTITLE") and node.value.value not in ES:
                missing.append(f"{path.name}: {node.value.value}")
    assert not missing, missing


def test_most_widget_labels_are_translated():
    strings = ui_strings()
    missing = sorted(s for s in strings if s not in ES and not any(s.startswith(p) and s[len(p):] in ES for p in i18n.PREFIXES))
    share = 1 - len(missing) / len(strings)
    assert share >= 0.8, f"{share:.0%} translated; missing e.g. {missing[:15]}"


def test_no_translation_is_orphaned_or_empty():
    strings = set(ui_strings()) | {m for m in ES if m.startswith(("#### ", "**"))}
    package = VIEWS.parent
    extra = [package / "demo.py", package / "analyze" / "story.py", package / "analyze" / "visuals.py"]     # textos que el panel muestra desde fuera de las vistas
    source = "".join(p.read_text(encoding="utf-8") for p in list(VIEWS.rglob("*.py")) + extra if p.name != "es.py")
    orphans = [k for k in ES if k not in strings and k not in source.replace("\\'", "'")]
    assert len(orphans) <= 8, orphans      # algunos textos se construyen por partes (name.capitalize(), cadenas largas)
    assert all(v.strip() for v in ES.values())


def spanish(page="Overview", language="Español"):
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0) * 2})
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["df"] = df
    at.session_state["original_df"] = df.copy()
    at.session_state["filename"] = "x.csv"
    at.session_state["nav"] = page
    at.session_state["lang_choice"] = language
    return at.run()


def test_dashboard_renders_in_spanish_and_restores_english():
    at = spanish("Model lab")
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Laboratorio de modelos" in html and "Compara muchos modelos" in html
    assert any(b.label == "Ejecutar el laboratorio" for b in at.button)
    assert [r.label for r in at.sidebar.radio][:1] == ["Área"] or any("Área" in str(m.value) for m in at.sidebar.markdown)
    at = spanish("Model lab", "English")      # otra ejecución: el idioma anterior no debe quedar pegado
    assert not at.exception
    assert any(b.label == "Run the lab" for b in at.button)
    assert i18n._original and st.button is i18n._original[("module", "button")]


@pytest.mark.parametrize("page", ["Autopilot", "Customers", "Survival", "Panel", "Monitoring", "Report"])
def test_spanish_pages_do_not_crash(page):
    assert not spanish(page).exception
