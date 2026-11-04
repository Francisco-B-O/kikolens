import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import panel as pn
from kikolens.cli.main import app
from kikolens.pipeline.project import load_project, save_project
from kikolens.pipeline.recipe import Recipe

runner = CliRunner()


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "region": rng.choice(list("ABC"), n),
                         "kind": rng.choice(["x", "y"], n), "units": rng.integers(1, 50, n),
                         "revenue": rng.normal(1000, 100, n)})


def test_metrics(df):
    assert pn.compute_metric(df, {"agg": "count"}) == 300
    assert pn.compute_metric(df, {"agg": "sum", "column": "units"}) == df["units"].sum()
    assert pn.compute_metric(df, {"agg": "nunique", "column": "region"}) == 3
    with pytest.raises(ValueError, match="numeric"):
        pn.compute_metric(df, {"agg": "mean", "column": "region"})
    with pytest.raises(ValueError, match="not found"):
        pn.compute_metric(df, {"agg": "sum", "column": "ghost"})
    with pytest.raises(ValueError, match="Aggregation"):
        pn.compute_metric(df, {"agg": "mode", "column": "units"})


@pytest.mark.parametrize("tile", [
    {"kind": "bar", "x": "region", "y": "revenue", "agg": "mean"}, {"kind": "bar", "x": "region"},
    {"kind": "line", "x": "date", "y": "revenue", "agg": "sum"}, {"kind": "area", "x": "date", "y": "units"},
    {"kind": "pie", "x": "kind"}, {"kind": "histogram", "x": "revenue", "color": "region"},
    {"kind": "scatter", "x": "units", "y": "revenue", "color": "kind"}, {"kind": "box", "x": "region", "y": "revenue"}])
def test_every_chart_kind_builds(df, tile):
    assert len(pn.build_figure(df, tile).data) >= 1


def test_chart_errors_are_friendly(df):
    cases = [({"kind": "histogram", "x": "region"}, "numeric"), ({"kind": "scatter", "x": "units"}, "vertical"),
             ({"kind": "box", "x": "region", "y": "kind"}, "numeric"), ({"kind": "donut", "x": "region"}, "Chart kind"),
             ({"kind": "bar", "x": "ghost"}, "not found"),
             ({"kind": "bar", "x": "region", "y": "kind", "agg": "sum"}, "numeric")]
    for tile, text in cases:
        with pytest.raises(ValueError, match=text):
            pn.build_figure(df, tile)


def test_dates_are_grouped_into_periods(df):
    assert len(pn._grouped(df, "date", "revenue", "sum", 10)) < len(df)


def test_table_tile(df):
    out = pn.build_table(df, {"columns": ["region", "units"], "sort": "units", "ascending": False, "n": 5})
    assert out.shape == (5, 2) and out["units"].is_monotonic_decreasing
    grouped = pn.build_table(df, {"group_by": "region", "value": "units", "agg": "sum", "n": 10, "sort": "units"})
    assert grouped["units"].sum() == df["units"].sum()
    with pytest.raises(ValueError, match="Cannot sort"):
        pn.build_table(df, {"columns": ["region"], "sort": "units"})


def test_filters_apply_and_errors_do_not_stop_other_tiles(df):
    panel = pn.Panel("t", filters=[{"column": "region", "op": "==", "value": "A"}])
    panel.add({"type": "metric", "agg": "count"})
    panel.add({"type": "chart", "kind": "bar", "x": "ghost"})
    data, results = panel.evaluate(df)
    assert len(data) == (df["region"] == "A").sum() and results[0][1] == len(data)
    assert results[1][2] and "not found" in results[1][2]


def test_panel_editing_and_limits():
    panel = pn.Panel()
    a = panel.add({"type": "metric"})
    b = panel.add({"type": "table"})
    panel.move(b, -1)
    assert [t["id"] for t in panel.tiles] == [b, a]
    panel.remove(a)
    assert len(panel.tiles) == 1
    with pytest.raises(ValueError, match="Tile type"):
        panel.add({"type": "gauge"})
    for _ in range(pn.MAX_TILES - 1):
        panel.add({"type": "metric"})
    with pytest.raises(ValueError, match="at most"):
        panel.add({"type": "metric"})


def test_json_roundtrip_and_validation(df):
    panel = pn.suggest_panel(df, "Mine")
    again = pn.Panel.from_json(panel.to_json())
    assert again.title == "Mine" and [t["id"] for t in again.tiles] == [t["id"] for t in panel.tiles]
    assert not any(e for _, _, e in again.evaluate(df)[1])
    with pytest.raises(ValueError, match="valid JSON"):
        pn.Panel.from_json("{")
    with pytest.raises(ValueError, match="not a KikoLens panel"):
        pn.Panel.from_json("[]")
    with pytest.raises(ValueError, match="newer"):
        pn.Panel.from_json(json.dumps({"kikolens_panel": 9, "tiles": []}))


def test_suggested_panel_covers_the_data(df):
    kinds = {t.get("kind") or t["type"] for t in pn.suggest_panel(df).tiles}
    assert {"metric", "line", "bar", "pie", "histogram", "table"} <= kinds


def test_html_export_escapes_and_is_self_contained(df):
    panel = pn.Panel("<script>alert(1)</script>", filters=[{"column": "region", "op": "==", "value": "A"}])
    panel.add({"type": "metric", "title": "<b>x</b>", "agg": "count"})
    panel.add({"type": "chart", "kind": "bar", "x": "ghost", "title": "broken"})
    page = pn.panel_html(df, panel, offline=False)
    assert "<script>alert(1)</script>" not in page and "&lt;b&gt;x&lt;/b&gt;" in page
    assert "not found" in page and "Filters" in page
    assert str(df["revenue"].iloc[0]) not in page   # los datos no se incluyen


def test_format_metric():
    assert pn.format_metric(None) == "n/a" and pn.format_metric(12345) == "12,345"
    assert pn.format_metric(0.123456) == "0.1235" and pn.format_metric(1234567.8) == "1,234,568"


def test_project_keeps_the_panel(df):
    panel = pn.suggest_panel(df)
    blob = save_project(df, Recipe(), panel=panel.to_json())
    assert pn.Panel.from_json(load_project(blob)["panel"]).title == panel.title
    assert load_project(save_project(df, Recipe()))["panel"] is None


def test_cli_panel(tmp_path, df):
    source = tmp_path / "d.csv"
    df.to_csv(source, index=False)
    out = tmp_path / "p.html"
    result = runner.invoke(app, ["panel", str(source), "--output", str(out), "--save", str(tmp_path / "p.json")])
    assert result.exit_code == 0 and out.exists() and (tmp_path / "p.json").exists()
    assert runner.invoke(app, ["panel", str(source), str(tmp_path / "p.json"), "--output", str(out)]).exit_code == 0
    assert runner.invoke(app, ["panel", str(source), str(tmp_path / "missing.json")]).exit_code == 1


def test_partial_first_and_last_periods_are_not_summed_as_if_complete():
    idx = pd.date_range("2024-01-05", "2024-08-20")            # viernes a martes: semanas incompletas en los dos extremos
    df = pd.DataFrame({"date": idx, "v": 10.0})
    out = pn._grouped(df, "date", "v", "sum", 10)
    assert (out["v"] == 70).all() and len(out) >= 8              # solo semanas completas
    means = pn._grouped(df, "date", "v", "mean", 10)
    assert len(means) == len(out) + 2                            # una media no se distorsiona: se conservan los extremos
