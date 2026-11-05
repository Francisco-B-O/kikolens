import io
import sqlite3

import numpy as np
import pandas as pd
import pytest
import requests
from openpyxl import load_workbook
from typer.testing import CliRunner

from kikolens.cli.main import app
from kikolens.pipeline.recipe import Recipe
from kikolens.report.workbook import build_analysis_workbook
from kikolens.utils import loaders as ld
from kikolens.utils import writers as wr

runner = CliRunner()


@pytest.fixture
def df():
    return pd.DataFrame({"when": pd.date_range("2024-01-01", periods=6), "name": ["a", "b", "=SUM(A1:A2)", "+cmd", "@x", None],
                         "price": [1.5, 2.25, np.nan, 4.0, 5.5, 6.0], "n": [1, 2, 3, 4, 5, 6]})


# ── Formatos ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".xlsx", ".json", ".ndjson", ".parquet", ".feather"])
def test_every_format_roundtrips(tmp_path, df, suffix):
    path = wr.write_frame(df, tmp_path / "out" / f"d{suffix}")
    back = ld.load_data(str(path)) if suffix != ".feather" else pd.read_feather(path)
    assert len(back) == len(df) and list(back.columns) == list(df.columns)
    assert back["n"].tolist() == df["n"].tolist()


def test_unknown_format_and_bytes_helper(tmp_path, df):
    with pytest.raises(ValueError, match="Output must end"):
        wr.write_frame(df, tmp_path / "x.docx")
    assert wr.frame_to_bytes(df, ".csv").startswith(b"when,name")
    assert wr.frame_to_bytes(df, ".xlsx")[:2] == b"PK"


# ── Excel con formato ────────────────────────────────────────────────────────
def test_workbook_formatting_and_formula_safety(df):
    blob = wr.build_workbook({"Data": df, "Second": df.head(2)}, {"Source": "test"}, title="T")
    wb = load_workbook(io.BytesIO(blob))
    assert wb.sheetnames == ["About", "Data", "Second"]
    ws = wb["Data"]
    assert ws.freeze_panes == "A2" and ws.auto_filter.ref == "A1:D7" and ws["A1"].font.bold
    assert ws["B4"].value == "=SUM(A1:A2)" and ws["B4"].data_type == "s"        # texto, no fórmula
    assert ws["B5"].data_type == "s" and ws["B6"].data_type == "s"
    assert ws["C2"].number_format.startswith("#,##0") and ws["A2"].number_format == "yyyy-mm-dd"
    assert ws["C4"].value is None                                                # NaN → celda vacía
    assert wb["About"]["A1"].value == "T" and wb["About"]["B4"].value == "test"


def test_workbook_edge_cases():
    with pytest.raises(ValueError, match="nothing to export"):
        wr.build_workbook({})
    odd = pd.DataFrame({"mixed": [1, "a", {"k": 1}, None], "tz": pd.date_range("2024-01-01", periods=4, tz="UTC")})
    wb = load_workbook(io.BytesIO(wr.build_workbook({"a/b:c*?[x]" + "y" * 40: odd, "dup": odd})))
    assert all(len(n) <= 31 and not set("[]:*?/\\") & set(n) for n in wb.sheetnames)
    assert wb[wb.sheetnames[0]]["A4"].value == "{'k': 1}"


def test_analysis_workbook_has_a_sheet_per_part(df):
    recipe = Recipe()
    recipe.add("drop_duplicates", {}, "Removed 0 duplicated rows.")
    models = {"Lab": {"target": "price", "metrics": {"R2": 0.9, "label": "x"}}}
    rules = {"table": pd.DataFrame({"rule": ["n must be unique"], "violations": [0], "pass %": [100.0], "error": [""], "id": ["x"]})}
    wb = load_workbook(io.BytesIO(build_analysis_workbook(df, "demo", recipe, rules, models, df.head(2))))
    assert {"About", "Data", "Predictions", "Column profile", "Statistics", "Quality issues", "Recipe", "Rules",
            "Model metrics"} <= set(wb.sheetnames)
    assert [c.value for c in wb["Model metrics"][2]] == ["Lab", "price", "R2", 0.9]
    assert "id" not in [c.value for c in wb["Rules"][1]]
    light = load_workbook(io.BytesIO(build_analysis_workbook(df, "demo", include_data=False)))
    assert "Data" not in light.sheetnames


# ── Bases de datos ───────────────────────────────────────────────────────────
def test_database_write_modes(tmp_path, df):
    uri = f"sqlite:///{(tmp_path / 'r.db').as_posix()}"
    assert wr.write_database(df, uri, "results") == 6
    with pytest.raises(ValueError, match="already exists"):
        wr.write_database(df, uri, "results")                                   # por defecto no pisa
    assert wr.write_database(df.head(2), uri, "results", "append") == 2
    with sqlite3.connect(tmp_path / "r.db") as con:
        assert con.execute("select count(*) from results").fetchone()[0] == 8
    wr.write_database(df.head(3), uri, "results", "replace")
    with sqlite3.connect(tmp_path / "r.db") as con:
        assert con.execute("select count(*) from results").fetchone()[0] == 3


def test_database_write_validation(tmp_path, df):
    uri = f"sqlite:///{(tmp_path / 'r.db').as_posix()}"
    for bad in ("1abc", "a-b", "x; drop table y", "", "a" * 70):
        with pytest.raises(ValueError, match="table name"):
            wr.write_database(df, uri, bad)
    with pytest.raises(ValueError, match="if_exists"):
        wr.write_database(df, uri, "t", "drop")
    with pytest.raises(ValueError, match="no data"):
        wr.write_database(df.head(0), uri, "t")
    with pytest.raises(ValueError, match="unique"):
        wr.write_database(pd.DataFrame([[1, 2]], columns=["a", "a"]), uri, "t")
    with pytest.raises(ValueError, match="Could not write"):
        wr.write_database(df, "sqlite:////nonexistent_folder/xyz/r.db", "t")


def test_database_read_tables_and_queries(tmp_path, df):
    uri = f"sqlite:///{(tmp_path / 'r.db').as_posix()}"
    wr.write_database(df, uri, "alpha")
    wr.write_database(df.head(2), uri, "beta")
    assert ld.list_tables(uri) == ["alpha", "beta"]
    assert len(ld.load_data(uri)) == 6 and len(ld.load_data(uri, table="beta")) == 2
    assert len(ld.load_data(uri, query="SELECT n FROM alpha WHERE n > 4")) == 2
    with pytest.raises(ValueError, match="not found"):
        ld.load_data(uri, table="ghost")
    with pytest.raises(ValueError, match="Security"):
        ld.load_data(uri, query="DROP TABLE alpha")


# ── Cargas nuevas ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,columns", [("a\tb\n1\t2\n3\t4", ["a", "b"]), ("a,b\n1,2\n3,4", ["a", "b"]),
                                           ("a;b\n1;2\n3;4", ["a", "b"]), ("a|b\n1|2", ["a", "b"])])
def test_pasted_tables_detect_the_separator(text, columns):
    assert list(ld.load_text(text).columns) == columns


def test_pasted_text_errors():
    with pytest.raises(ValueError, match="nothing to read"):
        ld.load_text("  \n ")
    with pytest.raises(ValueError, match="No rows"):
        ld.load_text("a,b")


def test_excel_sheets_can_be_listed_and_chosen(tmp_path):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as xl:
        pd.DataFrame({"a": [1, 2]}).to_excel(xl, sheet_name="first", index=False)
        pd.DataFrame({"b": [3, 4, 5]}).to_excel(xl, sheet_name="second", index=False)
    assert ld.list_sheets(str(path)) == ["first", "second"]
    assert list(ld.load_data(str(path)).columns) == ["a"]
    assert list(ld.load_data(str(path), sheet="second").columns) == ["b"]
    with open(path, "rb") as handle:
        assert len(ld.load_data(handle, sheet="second")) == 3 and ld.list_sheets(handle) == ["first", "second"]
    with pytest.raises(ValueError, match="Could not read"):
        ld.list_sheets(str(tmp_path / "missing.xlsx"))


def test_google_sheets_links_become_csv_downloads(monkeypatch):
    url = "https://docs.google.com/spreadsheets/d/ABC123_-x/edit?usp=sharing#gid=77"
    assert ld.google_sheets_csv_url(url) == "https://docs.google.com/spreadsheets/d/ABC123_-x/export?format=csv&gid=77"
    assert ld.google_sheets_csv_url("https://docs.google.com/spreadsheets/d/ABC/edit").endswith("export?format=csv")
    assert ld.google_sheets_csv_url("https://example.com/a.csv") is None
    seen = {}

    class Reply:
        headers = {"content-type": "text/csv"}
        content = b"a,b\n1,2\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda u, timeout: seen.update(url=u) or Reply())
    assert ld.load_data(url).shape == (1, 2) and seen["url"].endswith("gid=77")
    Reply.headers = {"content-type": "text/html; charset=utf-8"}
    with pytest.raises(ValueError, match="anyone with the link"):
        ld.load_data(url)


def test_cloud_storage_needs_its_extra_package(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "fsspec", None)
    with pytest.raises(ValueError, match="pip install fsspec s3fs"):
        ld.load_data("s3://bucket/data.csv")
    with pytest.raises(ValueError, match="Unsupported"):
        ld.load_data("s3://bucket/data.txt")


def test_cloud_storage_reads_through_fsspec(monkeypatch, tmp_path):
    fsspec = pytest.importorskip("fsspec")
    path = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
    monkeypatch.setattr(fsspec, "open", lambda url, mode: open(path, "rb"))
    assert len(ld.load_data("gs://bucket/d.csv")) == 2


# ── CLI ──────────────────────────────────────────────────────────────────────
def test_cli_export_and_to_db(tmp_path, df):
    source = tmp_path / "d.csv"
    df.to_csv(source, index=False)
    assert runner.invoke(app, ["export", str(source), str(tmp_path / "o.parquet")]).exit_code == 0
    assert runner.invoke(app, ["export", str(source), str(tmp_path / "o.json")]).exit_code == 0
    wb = runner.invoke(app, ["export", str(source), str(tmp_path / "a.xlsx"), "--workbook"])
    assert wb.exit_code == 0 and "Column profile" in load_workbook(tmp_path / "a.xlsx").sheetnames
    assert runner.invoke(app, ["export", str(source), str(tmp_path / "a.csv"), "--workbook"]).exit_code == 1
    assert runner.invoke(app, ["export", str(source), str(tmp_path / "a.docx")]).exit_code == 1
    recipe = Recipe()
    recipe.add("filter_rows", {"column": "n", "op": ">", "value": "3"}, "")
    (tmp_path / "r.json").write_text(recipe.to_json())
    runner.invoke(app, ["export", str(source), str(tmp_path / "f.csv"), "--recipe", str(tmp_path / "r.json")])
    assert len(pd.read_csv(tmp_path / "f.csv")) == 3

    uri = f"sqlite:///{(tmp_path / 'x.db').as_posix()}"
    assert runner.invoke(app, ["to-db", str(source), uri, "results"]).exit_code == 0
    assert runner.invoke(app, ["to-db", str(source), uri, "results"]).exit_code == 1
    assert runner.invoke(app, ["to-db", str(source), uri, "results", "--if-exists", "replace"]).exit_code == 0
    assert runner.invoke(app, ["to-db", str(source), uri, "bad name"]).exit_code == 1


def test_existing_commands_can_now_write_any_format(tmp_path, df):
    source = tmp_path / "d.csv"
    df.to_csv(source, index=False)
    assert runner.invoke(app, ["clean", str(source), "--output", str(tmp_path / "c.xlsx")]).exit_code == 0
    assert (tmp_path / "c.xlsx").exists()
