import io
import pandas as pd
import pytest
from unittest.mock import MagicMock
from kikolens.utils.loaders import load_data


def test_load_csv_from_streamlit_upload():
    csv_content = b"name,age\nAlice,30\nBob,25\n"
    mock_file = MagicMock()
    mock_file.name = "data.csv"
    mock_file.read.return_value = csv_content
    # Simular el comportamiento de pd.read_csv con un objeto file-like
    mock_file.__iter__ = lambda self: iter(csv_content.decode().splitlines(keepends=True))
    # Usar io.BytesIO para que pandas pueda leerlo
    real_file = io.BytesIO(csv_content)
    real_file.name = "data.csv"
    result = load_data(real_file)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["name", "age"]
    assert len(result) == 2


def test_load_json_from_streamlit_upload():
    json_content = b'[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
    real_file = io.BytesIO(json_content)
    real_file.name = "data.json"
    result = load_data(real_file)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_load_csv_from_local_path(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("a,b\n1,2\n3,4\n")
    result = load_data(str(csv_file))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == ["a", "b"]


def test_load_json_local_file(tmp_path):
    json_file = tmp_path / "test.json"
    json_file.write_text('[{"x": 1}, {"x": 2}]')
    result = load_data(str(json_file))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_unsupported_format_raises():
    mock_file = MagicMock()
    mock_file.name = "data.txt"
    with pytest.raises(ValueError, match="Unsupported"):
        load_data(mock_file)


def test_unsupported_local_path_raises(tmp_path):
    bad_file = tmp_path / "data.xml"
    bad_file.write_text("<root/>")
    with pytest.raises(ValueError, match="Unsupported"):
        load_data(str(bad_file))


def test_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "d.parquet"
    df.to_parquet(path)
    pd.testing.assert_frame_equal(load_data(str(path)), df)


def test_tsv(tmp_path):
    path = tmp_path / "d.tsv"
    path.write_text("a\tb\n1\t2\n")
    assert load_data(str(path)).shape == (1, 2)


def test_uploaded_ndjson_and_parquet(tmp_path):
    import io

    class Upload(io.BytesIO):
        def __init__(self, data, name):
            super().__init__(data)
            self.name = name

    nd = Upload(b'{"a": 1}\n{"a": 2}\n', "x.ndjson")
    assert load_data(nd)["a"].tolist() == [1, 2]

    buf = io.BytesIO()
    pd.DataFrame({"a": [1]}).to_parquet(buf)
    assert len(load_data(Upload(buf.getvalue(), "x.parquet"))) == 1


def test_url_download(monkeypatch):
    import requests

    class Resp:
        content = b"a,b\n1,2\n3,4\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda url, timeout: Resp())
    assert load_data("https://example.com/data.csv").shape == (2, 2)


def test_url_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported"):
        load_data("https://example.com/page.html")


def test_sqlite_table_with_odd_name(tmp_path):
    import sqlite3

    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE "my table" (a INTEGER)')
    con.execute('INSERT INTO "my table" VALUES (1)')
    con.commit()
    con.close()
    assert len(load_data(f"sqlite:///{db.as_posix()}")) == 1
