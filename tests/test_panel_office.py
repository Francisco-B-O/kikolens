import io

import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import panel as pn

pytest.importorskip("pptx")
pytest.importorskip("docx")

from kikolens.report import office  # noqa: E402


@pytest.fixture(scope="module")
def sales():
    rng = np.random.default_rng(4)
    n = 400
    df = pd.DataFrame({"region": rng.choice(["North", "South", "East", "West"], n), "channel": rng.choice(["web", "shop"], n),
                       "revenue": rng.gamma(3, 100, n), "units": rng.integers(1, 20, n), "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 200, n), "D")})
    df["cost"] = df["revenue"] * rng.uniform(0.3, 0.7, n)
    return df


def full_panel():
    p = pn.Panel("Sales review")
    p.add({"type": "metric", "title": "Rows", "agg": "count"})
    p.add({"type": "metric", "title": "Revenue", "agg": "sum", "column": "revenue"})
    p.add({"type": "chart", "kind": "bar", "title": "Revenue by region", "x": "region", "y": "revenue", "agg": "sum"})
    p.add({"type": "chart", "kind": "pie", "title": "Rows by channel", "x": "channel"})
    p.add({"type": "chart", "kind": "line", "title": "Revenue over time", "x": "date", "y": "revenue", "agg": "sum"})
    p.add({"type": "chart", "kind": "histogram", "title": "Revenue distribution", "x": "revenue"})
    p.add({"type": "chart", "kind": "scatter", "title": "Cost against revenue", "x": "revenue", "y": "cost"})
    p.add({"type": "chart", "kind": "box", "title": "Revenue spread", "x": "region", "y": "revenue"})
    p.add({"type": "table", "title": "Top rows", "n": 5, "sort": "revenue"})
    p.add({"type": "chart", "kind": "bar", "title": "Broken", "x": "nope", "y": "revenue"})
    return p


# ── Filtros cruzados ─────────────────────────────────────────────────────────
def test_selecting_bars_slices_filters_the_other_tiles_but_not_the_source(sales):
    p = full_panel()
    bar = next(t for t in p.tiles if t["title"] == "Revenue by region")
    flt = pn.selection_filters(bar, {"points": [{"y": "North", "x": 100.0}, {"y": "West", "x": 50.0}]}, sales)
    assert flt == [{"column": "region", "op": "in", "value": ["North", "West"]}]
    pie = next(t for t in p.tiles if t.get("kind") == "pie")
    assert pn.selection_filters(pie, {"points": [{"label": "web"}]}, sales) == [{"column": "channel", "op": "in", "value": ["web"]}]
    data, results = p.evaluate(sales, {bar["id"]: flt})
    assert set(data["region"]) == {"North", "West"}
    by_title = {t["title"]: (v, e) for t, v, e in results}
    assert by_title["Revenue"][0] == pytest.approx(data["revenue"].sum()) and by_title["Rows"][0] == len(data)
    source = by_title["Revenue by region"][0]
    assert len(source.data[0].y) == 4                                   # el gráfico origen sigue mostrando todos los grupos
    pie_fig = by_title["Rows by channel"][0]
    assert sum(pie_fig.data[0].values) == len(data)


def test_boxes_on_histograms_and_scatters_and_date_ranges_become_range_filters(sales):
    p = full_panel()
    tiles = {t["title"]: t for t in p.tiles}
    hist = pn.selection_filters(tiles["Revenue distribution"], {"box": [{"x": [400, 100]}]}, sales)
    assert hist == [{"column": "revenue", "op": ">=", "value": 100}, {"column": "revenue", "op": "<=", "value": 400}]
    sc = pn.selection_filters(tiles["Cost against revenue"], {"box": [{"x": [0, 200], "y": [10, 90]}]}, sales)
    assert [f["column"] for f in sc] == ["revenue", "revenue", "cost", "cost"]
    line = pn.selection_filters(tiles["Revenue over time"], {"points": [{"x": "2024-03-01"}, {"x": "2024-02-01"}]}, sales)
    assert line == [{"column": "date", "op": ">=", "value": "2024-02-01T00:00:00"}, {"column": "date", "op": "<=", "value": "2024-03-01T00:00:00"}]
    data, _ = p.evaluate(sales, {tiles["Revenue over time"]["id"]: line, tiles["Revenue distribution"]["id"]: hist})
    assert data["date"].between("2024-02-01", "2024-03-01").all() and data["revenue"].between(100, 400).all() and 0 < len(data) < len(sales)
    assert pn.describe_filters(hist) == "revenue from 100 to 400" and "region is North" in pn.describe_filters([{"column": "region", "op": "in", "value": ["North"]}])


def test_empty_or_unusable_selections_filter_nothing_and_an_empty_result_is_explained(sales):
    p = full_panel()
    bar = p.tiles[2]
    assert pn.selection_filters(bar, None, sales) == [] and pn.selection_filters(bar, {"points": []}, sales) == []
    assert pn.selection_filters(p.tiles[0], {"points": [{"y": "x"}]}, sales) == []                 # una métrica no se puede marcar
    assert pn.selection_filters({**bar, "x": "nope"}, {"points": [{"y": "a"}]}, sales) == []
    nothing = {"t1": [{"column": "region", "op": "in", "value": ["Atlantis"]}]}
    data, results = p.evaluate(sales, nothing)
    assert len(data) == 0 and all(e == "No rows match the current selection." for _, _, e in results)
    same, _ = p.evaluate(sales, {})
    assert len(same) == len(sales)


# ── PowerPoint ───────────────────────────────────────────────────────────────
def test_pptx_has_native_charts_key_figures_tables_and_the_failures(sales):
    from pptx import Presentation
    blob = office.panel_pptx(sales, full_panel())
    prs = Presentation(io.BytesIO(blob))
    slides = list(prs.slides)
    texts = [" ".join(sh.text_frame.text for sh in s.shapes if sh.has_text_frame) for s in slides]
    assert "Sales review" in texts[0] and "400 of 400 rows" in texts[0]
    assert any("Key figures" in t and "Revenue" in t for t in texts)
    charts = [sh.chart for s in slides for sh in s.shapes if sh.has_chart]
    assert len(charts) == 5                                                    # barras, sectores, línea, histograma y dispersión; la caja va como tabla
    assert any(len(list(c.plots[0].categories)) == 4 for c in charts)
    assert sum(1 for s in slides for sh in s.shapes if sh.has_table) == 2      # la caja y la tabla
    assert "Could not be calculated" in texts[-1] and "Broken" in texts[-1] and "nope" in texts[-1]


def test_pptx_and_docx_record_the_filters_and_the_chart_selection(sales):
    from docx import Document
    from pptx import Presentation
    p = full_panel()
    p.filters = [{"column": "channel", "op": "==", "value": "web"}]
    cross = {p.tiles[2]["id"]: [{"column": "region", "op": "in", "value": ["North"]}]}
    first = Presentation(io.BytesIO(office.panel_pptx(sales, p, cross))).slides[0]
    cover = " ".join(sh.text_frame.text for sh in first.shapes if sh.has_text_frame)
    assert "channel == web" in cover and "region is North" in cover
    rows = int(((sales["channel"] == "web") & (sales["region"] == "North")).sum())
    assert f"{rows:,} of 400 rows" in cover
    body = "\n".join(par.text for par in Document(io.BytesIO(office.panel_docx(sales, p, cross))).paragraphs)
    assert "region is North" in body and f"{rows:,} of 400 rows" in body


# ── Word ─────────────────────────────────────────────────────────────────────
def test_docx_has_headings_images_tables_and_the_failures(sales):
    from docx import Document
    doc = Document(io.BytesIO(office.panel_docx(sales, full_panel())))
    headings = [par.text for par in doc.paragraphs if par.style.name.startswith("Heading") or par.style.name == "Title"]
    assert headings[0] == "Sales review" and {"Key figures", "Revenue by region", "Top rows", "Could not be calculated"} <= set(headings)
    assert len(doc.inline_shapes) == 5 and len(doc.tables) == 3                # cifras, caja y tabla
    assert doc.tables[0].cell(1, 0).text == "Revenue" and "Broken: " in "\n".join(par.text for par in doc.paragraphs)


def test_every_chart_kind_gives_drawable_data(sales):
    p = full_panel()
    for tile in p.tiles:
        if tile["type"] != "chart" or tile["title"] == "Broken":
            continue
        spec = office.chart_data(pn.build_figure(sales, tile), tile)
        assert spec["shape"] in ("categories", "points", "table")
        if spec["shape"] == "categories":
            assert len(spec["categories"]) == len(spec["values"]) > 0
            assert office._png(spec)[:8] == b"\x89PNG\r\n\x1a\n"
    assert office._png({"kind": "scatter", "x": [1, 2], "y": [2, 3], "x_label": "a", "y_label": "b"})[:4] == b"\x89PNG"


def test_a_panel_with_no_tiles_still_exports_a_cover(sales):
    from docx import Document
    from pptx import Presentation
    empty = pn.Panel("Empty")
    assert len(Presentation(io.BytesIO(office.panel_pptx(sales, empty))).slides) == 1
    assert Document(io.BytesIO(office.panel_docx(sales, empty))).paragraphs[0].text == "Empty"


def test_missing_libraries_give_an_actionable_message(monkeypatch):
    import builtins
    real = builtins.__import__

    def refuse(name, *a, **k):
        if name.startswith("pptx") or name.startswith("docx"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ImportError, match=r"kikolens\[office\]"):
        office.panel_pptx(pd.DataFrame({"a": [1]}), pn.Panel("x"))
    with pytest.raises(ImportError, match="python-docx"):
        office.panel_docx(pd.DataFrame({"a": [1]}), pn.Panel("x"))


def test_terminal_and_python_api_write_every_format(sales, tmp_path):
    import kikolens as kl
    from typer.testing import CliRunner
    from kikolens.cli.main import app
    csv = tmp_path / "s.csv"
    sales.to_csv(csv, index=False)
    runner = CliRunner()
    for fmt in ("pptx", "docx", "html"):
        out = runner.invoke(app, ["panel", str(csv), "--format", fmt, "--output", str(tmp_path / f"p.{fmt}"), "--where", "region in North,West"])
        assert out.exit_code == 0, out.output
        assert (tmp_path / f"p.{fmt}").stat().st_size > 1000
    from pptx import Presentation
    cover = " ".join(sh.text_frame.text for sh in Presentation(str(tmp_path / "p.pptx")).slides[0].shapes if sh.has_text_frame)
    assert f"{int(sales['region'].isin(['North', 'West']).sum()):,} of 400 rows" in cover
    assert runner.invoke(app, ["panel", str(csv), "--format", "pdf"]).exit_code == 1
    assert runner.invoke(app, ["panel", str(csv), "--where", "garbage"]).exit_code == 1
    built = kl.panel(sales, tmp_path / "api.pptx", where=["channel == web"])
    assert built.exists() and kl.panel(sales).tiles
    saved = kl.panel(sales, definition=full_panel().to_json())
    assert saved.title == "Sales review"
    with pytest.raises(ValueError, match="must end in"):
        kl.panel(sales, tmp_path / "x.pdf")


def test_panel_filters_can_be_read_from_text():
    assert pn.parse_filter("region in North, West") == {"column": "region", "op": "in", "value": ["North", "West"]}
    assert pn.parse_filter("revenue>=100") == {"column": "revenue", "op": ">=", "value": "100"}
    assert pn.parse_filter("note is empty") == {"column": "note", "op": "is empty", "value": None}
    with pytest.raises(ValueError, match="Cannot read"):
        pn.parse_filter("garbage")
