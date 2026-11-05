import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import transform as t


@pytest.fixture
def sales():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "region": list("ABABABABAB"),
        "unit price": [10.0, 20, 30, 40, 50, 60, 70, 80, 90, np.nan],
        "units": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "code": ["a-1", "b-2", "c-3", "d-4", "e-5", "f-6", "g-7", "h-8", "i-9", None],
    })


# ── Expresiones ──────────────────────────────────────────────────────────────
def test_formula_with_columns_and_backticks(sales):
    out, msg = t.add_column(sales, "total", "`unit price` * units + 1")
    assert out["total"].iloc[1] == 41
    assert "total" in msg and "total" not in sales.columns


def test_formula_functions_and_division_by_zero(sales):
    out, _ = t.add_column(sales, "r", "log1p(units) / (units - 1)")
    assert np.isnan(out["r"].iloc[0])  # división por cero → NaN, no infinito
    assert out["r"].iloc[1] == pytest.approx(np.log(3))


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')", "units.__class__", "open('x')", "[1, 2]", "units if 1 else 2",
    "lambda: 1", "(1).real", "'abc'", "True",
])
def test_formula_rejects_anything_but_arithmetic(sales, expr):
    with pytest.raises(ValueError):
        t.add_column(sales, "x", expr)


def test_formula_errors_are_friendly(sales):
    with pytest.raises(ValueError, match="Unknown column"):
        t.add_column(sales, "x", "nope + 1")
    with pytest.raises(ValueError, match="not valid"):
        t.add_column(sales, "x", "units +")
    with pytest.raises(ValueError, match="empty"):
        t.add_column(sales, "x", "  ")
    with pytest.raises(ValueError, match="longer"):
        t.add_column(sales, "x", "1+" * 200 + "1")
    with pytest.raises(ValueError, match="not found"):
        t.add_column(sales, "x", "`missing` + 1")


def test_formula_comparison_gives_flag(sales):
    out, _ = t.add_column(sales, "big", "units > 5")
    assert out["big"].sum() == 5


# ── Filas ────────────────────────────────────────────────────────────────────
def test_filter_numeric_text_and_dates(sales):
    assert len(t.filter_rows(sales, "units", ">=", "8")[0]) == 3
    assert len(t.filter_rows(sales, "region", "==", "A")[0]) == 5
    assert len(t.filter_rows(sales, "region", "in", "A, B")[0]) == 10
    assert len(t.filter_rows(sales, "code", "contains", "-1")[0]) == 1
    assert len(t.filter_rows(sales, "date", ">", "2024-01-08")[0]) == 2
    assert len(t.filter_rows(sales, "unit price", "is empty")[0]) == 1
    assert len(t.filter_rows(sales, "unit price", "is not empty")[0]) == 9


def test_filter_not_equal_keeps_missing_rows(sales):
    out, _ = t.filter_rows(sales, "code", "!=", "a-1")
    assert len(out) == 9


def test_filter_validation(sales):
    with pytest.raises(ValueError, match="not a number"):
        t.filter_rows(sales, "units", ">", "abc")
    with pytest.raises(ValueError, match="text"):
        t.filter_rows(sales, "region", ">", "A")
    with pytest.raises(ValueError, match="required"):
        t.filter_rows(sales, "units", ">", "")
    with pytest.raises(ValueError, match="Unknown operator"):
        t.filter_rows(sales, "units", "~", "1")


def test_sort_and_sample(sales):
    out, _ = t.sort_rows(sales, "units", ascending=False)
    assert out["units"].iloc[0] == 10
    assert len(t.sample_rows(sales, n=4)[0]) == 4
    assert len(t.sample_rows(sales, fraction=0.5)[0]) == 5
    assert len(t.sample_rows(sales, n=999)[0]) == 10
    with pytest.raises(ValueError):
        t.sample_rows(sales)
    with pytest.raises(ValueError):
        t.sample_rows(sales, fraction=2)


# ── Columnas ─────────────────────────────────────────────────────────────────
def test_concat_and_split(sales):
    out, _ = t.concat_columns(sales, ["region", "code"], "key", "|")
    assert out["key"].iloc[0] == "A|a-1"
    assert pd.isna(out["key"].iloc[9])
    out, _ = t.split_column(sales, "code", "-", ["letter", "num"])
    assert out["letter"].iloc[2] == "c" and out["num"].iloc[2] == "3"
    with pytest.raises(ValueError):
        t.concat_columns(sales, ["region"], "k")


def test_date_parts(sales):
    out, msg = t.extract_date_parts(sales, "date", ["year", "month", "weekday", "is_weekend", "days_since", "week"])
    assert out["date_year"].iloc[0] == 2024
    assert out["date_is_weekend"].iloc[5] == 1.0  # 2024-01-06 fue sábado
    assert out["date_days_since"].iloc[3] == 3
    with pytest.raises(ValueError):
        t.extract_date_parts(sales, "date", ["century"])
    with pytest.raises(ValueError, match="does not contain dates"):
        t.extract_date_parts(sales, "region", ["year"])


def test_bins_scale_log_rank_replace(sales):
    out, _ = t.bin_column(sales, "units", bins=2, method="uniform")
    assert out["units_bin"].nunique() == 2
    out, _ = t.scale_column(sales, "units", "standard")
    assert out["units"].mean() == pytest.approx(0, abs=1e-9)
    out, _ = t.scale_column(sales, "units", "minmax")
    assert out["units"].min() == 0 and out["units"].max() == 1
    with pytest.raises(ValueError, match="no spread"):
        t.scale_column(pd.DataFrame({"a": [1, 1, 1]}), "a")
    out, _ = t.log_transform(sales, "units")
    assert out["units"].iloc[0] == pytest.approx(np.log(2))
    with pytest.raises(ValueError, match="negative"):
        t.log_transform(pd.DataFrame({"a": [-1.0, 2.0]}), "a")
    assert t.rank_column(sales, "units")[0]["units_rank"].max() == 1.0
    out, _ = t.replace_values(sales, "region", {"A": "Alpha"})
    assert (out["region"] == "Alpha").sum() == 5
    out, _ = t.replace_values(sales, "units", {"1": 100})
    assert out["units"].iloc[0] == 100


def test_one_hot(sales):
    out, _ = t.one_hot_encode(sales, "region")
    assert {"region_A", "region_B"} <= set(out.columns) and "region" not in out.columns
    with pytest.raises(ValueError, match="categories"):
        t.one_hot_encode(sales, "units", max_categories=3)


# ── Secuencias ───────────────────────────────────────────────────────────────
def test_lags_and_rolling_keep_original_row_order(sales):
    shuffled = sales.sample(frac=1, random_state=1).reset_index(drop=True)
    out, _ = t.lag_features(shuffled, "units", [1], order_by="date")
    # el lag de una fila es el valor de la fecha anterior, y las filas conservan su orden
    assert (out["date"] == shuffled["date"]).all()
    by_date = out.set_index("date")
    assert by_date.loc[pd.Timestamp("2024-01-05"), "units_lag1"] == 4
    out, _ = t.rolling_feature(sales, "units", 3, "mean", order_by="date")
    assert out["units_roll3_mean"].iloc[4] == pytest.approx(4.0)


def test_lags_by_group(sales):
    out, _ = t.lag_features(sales, "units", [1], group_by="region")
    assert pd.isna(out["units_lag1"].iloc[0]) and pd.isna(out["units_lag1"].iloc[1])
    assert out["units_lag1"].iloc[2] == 1
    out, _ = t.rolling_feature(sales, "units", 2, "sum", group_by="region")
    assert len(out) == 10
    with pytest.raises(ValueError):
        t.lag_features(sales, "units", [0])
    with pytest.raises(ValueError):
        t.rolling_feature(sales, "units", 1)


# ── Forma ────────────────────────────────────────────────────────────────────
def test_group_pivot_melt(sales):
    out, _ = t.group_aggregate(sales, "region", {"units": ["sum", "mean"]})
    assert out.loc[out["region"] == "A", "units_sum"].iloc[0] == 25
    with pytest.raises(ValueError, match="Unknown aggregation"):
        t.group_aggregate(sales, "region", {"units": "variance"})
    out, _ = t.pivot_table(sales, "date", "region", "units")
    assert {"units_A", "units_B"} <= set(out.columns)
    out, _ = t.melt_columns(sales, ["region"], ["units", "unit price"])
    assert len(out) == 20 and set(out["variable"]) == {"units", "unit price"}


def test_rename_validates_collisions(sales):
    out, _ = t.rename_columns(sales, {"units": "qty"})
    assert "qty" in out.columns
    with pytest.raises(ValueError, match="duplicate"):
        t.rename_columns(sales, {"units": "region"})


def test_merge_and_append(sales):
    lookup = pd.DataFrame({"region": ["A", "B"], "manager": ["Ana", "Ben"]})
    out, _ = t.merge_datasets(sales, lookup, on="region", how="left", other="managers")
    assert out["manager"].notna().all() and len(out) == 10
    with pytest.raises(ValueError, match="both datasets"):
        t.merge_datasets(sales, lookup, on="units")
    out, _ = t.append_dataset(sales, sales.head(2))
    assert len(out) == 12


def test_operations_do_not_modify_input(sales):
    snapshot = sales.copy(deep=True)
    for fn, args in [(t.add_column, ("z", "units * 2")), (t.scale_column, ("units",)), (t.log_transform, ("units",)),
                     (t.replace_values, ("region", {"A": "x"})), (t.bin_column, ("units",))]:
        fn(sales, *args)
    pd.testing.assert_frame_equal(sales, snapshot)
