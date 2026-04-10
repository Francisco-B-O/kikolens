import pandas as pd
import numpy as np
import pytest
from kikolens.analyze.stats import basic_statistics, correlation_matrix


def test_basic_statistics_returns_dataframe(simple_df):
    result = basic_statistics(simple_df)
    assert isinstance(result, pd.DataFrame)


def test_basic_statistics_has_numeric_columns(simple_df):
    result = basic_statistics(simple_df)
    # Debe incluir columnas numéricas (age, sales)
    assert "age" in result.columns or "age" in result.index


def test_basic_statistics_with_missing_values():
    df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0], "b": [10.0, None, 30.0, 40.0]})
    result = basic_statistics(df)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_correlation_matrix_only_numeric(simple_df):
    corr = correlation_matrix(simple_df)
    # Solo columnas numéricas (age, sales)
    assert isinstance(corr, pd.DataFrame)
    for col in corr.columns:
        assert col in ["age", "sales"]


def test_correlation_matrix_empty_on_no_numerics():
    df = pd.DataFrame({"name": ["a", "b"], "city": ["x", "y"]})
    corr = correlation_matrix(df)
    assert corr.empty


def test_correlation_matrix_diagonal_is_one(regression_df):
    corr = correlation_matrix(regression_df)
    for col in corr.columns:
        assert abs(corr.loc[col, col] - 1.0) < 1e-9
