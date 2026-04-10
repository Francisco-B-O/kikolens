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
