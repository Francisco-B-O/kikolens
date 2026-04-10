import pandas as pd
import numpy as np
import pytest
from kikolens.analyze.clustering import run_clustering_analysis


def test_returns_expected_keys(regression_df):
    result = run_clustering_analysis(regression_df, n_clusters=3)
    assert "fig" in result
    assert "profiles" in result
    assert "n_clusters" in result
    assert "df_with_clusters" in result


def test_correct_number_of_clusters(regression_df):
    result = run_clustering_analysis(regression_df, n_clusters=4)
    assert result["n_clusters"] == 4
    assert result["df_with_clusters"]["Kiko_Cluster"].nunique() == 4


def test_profiles_shape(regression_df):
    result = run_clustering_analysis(regression_df, n_clusters=3)
    assert result["profiles"].shape[0] == 3


def test_no_numeric_columns_raises():
    df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "city": ["NYC", "LA", "NYC"]})
    with pytest.raises(ValueError, match="No numeric columns"):
        run_clustering_analysis(df, n_clusters=2)


def test_handles_missing_values():
    df = pd.DataFrame({
        "a": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0],
        "b": [10.0, None, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    })
    result = run_clustering_analysis(df, n_clusters=2)
    assert isinstance(result["profiles"], pd.DataFrame)
