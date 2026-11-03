import pandas as pd
import pytest
from kikolens.sql.runner import run_sql_query


def test_basic_select(simple_df):
    result = run_sql_query(simple_df, "SELECT * FROM df")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(simple_df)


def test_select_with_filter(simple_df):
    result = run_sql_query(simple_df, "SELECT * FROM df WHERE age > 28")
    assert isinstance(result, pd.DataFrame)
    assert all(result["age"] > 28)


def test_select_aggregation(simple_df):
    result = run_sql_query(simple_df, "SELECT SUM(sales) AS total FROM df")
    assert isinstance(result, pd.DataFrame)
    assert "total" in result.columns
    assert result["total"].iloc[0] == pytest.approx(5300.0)


def test_select_order_by(simple_df):
    result = run_sql_query(simple_df, "SELECT name, sales FROM df ORDER BY sales DESC")
    assert result.iloc[0]["name"] == "Diana"


def test_insert_blocked(simple_df):
    with pytest.raises(ValueError, match="read-only"):
        run_sql_query(simple_df, "INSERT INTO df VALUES ('Eve', 40, 999)")


def test_update_blocked(simple_df):
    with pytest.raises(ValueError, match="read-only"):
        run_sql_query(simple_df, "UPDATE df SET age = 99 WHERE name = 'Alice'")


def test_delete_blocked(simple_df):
    with pytest.raises(ValueError, match="read-only"):
        run_sql_query(simple_df, "DELETE FROM df WHERE age < 30")


def test_drop_blocked(simple_df):
    with pytest.raises(ValueError, match="read-only"):
        run_sql_query(simple_df, "DROP TABLE df")


def test_with_clause(simple_df):
    sql = "WITH cte AS (SELECT * FROM df WHERE age > 25) SELECT * FROM cte"
    result = run_sql_query(simple_df, sql)
    assert isinstance(result, pd.DataFrame)


def test_csv_file(tmp_path):
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv(csv_file, index=False)
    result = run_sql_query(str(csv_file), "SELECT * FROM df")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


@pytest.mark.parametrize("query", [
    "SELECT 1; DROP TABLE df",
    "-- hi\nDROP TABLE df",
    "/* x */ DELETE FROM df",
    "WITH c AS (SELECT 1) INSERT INTO df SELECT * FROM c",
    "SELECT * FROM read_csv('/etc/passwd')",
    "COPY df TO 'out.csv'",
    "",
])
def test_dangerous_queries_blocked(simple_df, query):
    with pytest.raises(ValueError, match="Security"):
        run_sql_query(simple_df, query)


def test_keywords_inside_strings_and_functions_allowed(simple_df):
    result = run_sql_query(simple_df, "SELECT replace(name, 'A', 'B') AS n, 'drop; table' AS s FROM df")
    assert len(result) == len(simple_df)
    assert (result["s"] == "drop; table").all()


def test_trailing_semicolon_allowed(simple_df):
    assert len(run_sql_query(simple_df, "SELECT * FROM df;")) == len(simple_df)


def test_parquet_and_json_files(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_parquet(tmp_path / "t.parquet")
    df.to_json(tmp_path / "t.json")
    assert len(run_sql_query(str(tmp_path / "t.parquet"), "SELECT * FROM df")) == 3
    assert len(run_sql_query(str(tmp_path / "t.json"), "SELECT * FROM df")) == 3


def test_unsupported_file(tmp_path):
    (tmp_path / "x.txt").write_text("hi")
    with pytest.raises(ValueError, match="Unsupported"):
        run_sql_query(str(tmp_path / "x.txt"), "SELECT * FROM df")
