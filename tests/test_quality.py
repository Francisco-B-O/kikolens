import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import quality as q


@pytest.fixture
def messy():
    return pd.DataFrame({
        "id": [1, 2, 3, 3, 5, 6],
        "price": ["1,200", "300", "45", "45", "10", None],
        "when": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-05", "2024-01-06"],
        "const": ["x"] * 6,
        "name": ["  a", "b ", "c", "c", "d", "e"],
    })


def test_detects_text_types():
    assert q.looks_numeric_text(pd.Series(["1", "2,500", "45%", None]))
    assert not q.looks_numeric_text(pd.Series(["a", "b"]))
    assert q.looks_datetime_text(pd.Series(["2024-01-01", "2024-02-01"]))
    assert not q.looks_datetime_text(pd.Series(["10", "20"]))
    assert q.looks_boolean_text(pd.Series(["yes", "no", "yes"]))
    assert not q.looks_boolean_text(pd.Series([1, 0]))


def test_profile_flags_problems(messy):
    p = q.profile_quality(messy)
    kinds = {i["kind"] for i in p["issues"]}
    assert {"missing", "duplicates", "constant", "type"} <= kinds
    assert p["duplicate_rows"] == 1
    assert "const" in p["constant_columns"]
    assert set(p["mistyped_columns"]) == {"price", "when"}
    assert 0 <= p["score"] < 100


def test_clean_dataset_scores_100():
    df = pd.DataFrame({"a": np.arange(20.0), "b": list("abcdefghij") * 2})
    df["b"] = df["b"] + df["a"].astype(str)
    assert q.profile_quality(df)["score"] == 100.0


def test_empty_dataframe_does_not_crash():
    p = q.profile_quality(pd.DataFrame())
    assert p["rows"] == 0 and p["score"] == 100.0


def test_outlier_count():
    s = pd.Series([1, 2, 3, 2, 1, 2, 3, 100])
    assert q.count_outliers_iqr(s) == 1
    assert q.count_outliers_iqr(pd.Series(["a", "b"])) == 0


def test_operations_do_not_mutate_original(messy):
    snapshot = messy.copy()
    q.apply_operation(messy, "drop_duplicates")
    q.apply_operation(messy, "convert_type", column="price", target="numeric")
    q.apply_operation(messy, "strip_whitespace")
    pd.testing.assert_frame_equal(messy, snapshot)


def test_convert_numeric_and_datetime(messy):
    out, msg = q.apply_operation(messy, "convert_type", column="price", target="numeric")
    assert out["price"].iloc[0] == 1200 and pd.isna(out["price"].iloc[-1])
    out, _ = q.apply_operation(messy, "convert_type", column="when", target="datetime")
    assert pd.api.types.is_datetime64_any_dtype(out["when"])


def test_convert_reports_lost_values():
    df = pd.DataFrame({"a": ["1", "x", "3"]})
    _, msg = q.convert_type(df, "a", "numeric")
    assert "1 value" in msg


def test_fill_missing_strategies():
    df = pd.DataFrame({"n": [1.0, None, 3.0], "t": ["a", None, "a"]})
    assert q.fill_missing(df, "n", "mean")[0]["n"].iloc[1] == 2.0
    assert q.fill_missing(df, "n", "zero")[0]["n"].iloc[1] == 0
    assert q.fill_missing(df, "t", "mode")[0]["t"].iloc[1] == "a"
    assert q.fill_missing(df, "t", "constant", value="?")[0]["t"].iloc[1] == "?"
    with pytest.raises(ValueError, match="not valid"):
        q.fill_missing(df, "t", "mean")
    with pytest.raises(ValueError, match="required"):
        q.fill_missing(df, "t", "constant")


def test_clip_outliers_and_errors():
    df = pd.DataFrame({"a": [1, 2, 3, 2, 1, 2, 3, 100.0]})
    out, _ = q.clip_outliers(df, "a")
    assert out["a"].max() < 100
    with pytest.raises(ValueError):
        q.clip_outliers(pd.DataFrame({"a": ["x"]}), "a")


def test_standardize_names_handles_collisions():
    df = pd.DataFrame([[1, 2, 3]], columns=["First Name", "first-name", "  "])
    out, _ = q.standardize_column_names(df)
    assert list(out.columns) == ["first_name", "first_name_2", "column"]


def test_unknown_operation():
    with pytest.raises(ValueError, match="Unknown"):
        q.apply_operation(pd.DataFrame(), "explode")


def test_auto_clean_fixes_everything_safe(messy):
    out, log = q.auto_clean(messy)
    assert len(out) == 5
    assert "const" not in out.columns
    assert pd.api.types.is_numeric_dtype(out["price"])
    assert pd.api.types.is_datetime64_any_dtype(out["when"])
    assert out["name"].iloc[0] == "a"
    assert q.profile_quality(out)["score"] > q.profile_quality(messy)["score"]
    assert len(log) >= 4
