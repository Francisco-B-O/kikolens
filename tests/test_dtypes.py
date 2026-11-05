import pandas as pd

from kikolens.analyze.discovery import run_discovery_engine
from kikolens.analyze.quality import looks_numeric_text, strip_whitespace
from kikolens.utils.dtypes import categorical_columns, is_text, text_columns


def frames():
    base = pd.DataFrame({"t": ["a", "b", "a"], "n": [1, 2, 3]})
    return [base, base.astype({"t": "string"}), base.astype({"t": "category"})]


def test_text_detection_across_dtypes():
    object_df, string_df, category_df = frames()
    assert is_text(object_df["t"]) and is_text(string_df["t"])
    assert not is_text(category_df["t"]) and not is_text(object_df["n"])
    assert text_columns(string_df) == ["t"]
    assert all(categorical_columns(f) == ["t"] for f in frames())


def test_numeric_text_in_string_dtype():
    assert looks_numeric_text(pd.Series(["1", "2,500", "45%"], dtype="string"))


def test_strip_whitespace_in_string_dtype():
    out, _ = strip_whitespace(pd.DataFrame({"t": [" a ", "b "]}, dtype="string"))
    assert out["t"].tolist() == ["a", "b"]


def test_discovery_sees_string_columns():
    df = pd.DataFrame({"t": ["x"] * 95 + ["y"] * 5}, dtype="string")
    assert any("Low Variance" in i["type"] for i in run_discovery_engine(df))
