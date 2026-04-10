import pandas as pd
import numpy as np
import pytest
from kikolens.analyze.discovery import run_discovery_engine


def test_returns_list(simple_df):
    result = run_discovery_engine(simple_df)
    assert isinstance(result, list)


def test_detects_high_correlation(correlated_df):
    result = run_discovery_engine(correlated_df)
    types = [i["type"] for i in result]
    assert any("Correlation" in t for t in types)


def test_detects_unique_id():
    df = pd.DataFrame({
        "id":    range(100),
        "value": np.random.rand(100),
    })
    result = run_discovery_engine(df)
    types = [i["type"] for i in result]
    assert any("Unique" in t for t in types)


def test_detects_low_variance_categorical():
    df = pd.DataFrame({
        "category": ["A"] * 95 + ["B"] * 5,  # 95% dominado por "A"
        "value":    np.random.rand(100),
    })
    result = run_discovery_engine(df)
    types = [i["type"] for i in result]
    assert any("Low Variance" in t for t in types)


def test_detects_outliers():
    values = list(np.random.normal(50, 5, 100)) + [1000.0]  # outlier extremo
    df = pd.DataFrame({"val": values})
    result = run_discovery_engine(df)
    types = [i["type"] for i in result]
    assert any("Anomaly" in t for t in types)


def test_insight_structure():
    df = pd.DataFrame({"a": range(10), "b": range(10)})
    result = run_discovery_engine(df)
    for insight in result:
        assert "type" in insight
        assert "severity" in insight
        assert "message" in insight
        assert "detail" in insight


def test_handles_large_df():
    """Debe muestrear sin error con datasets grandes."""
    np.random.seed(0)
    df = pd.DataFrame(np.random.rand(10000, 5), columns=list("abcde"))
    result = run_discovery_engine(df)
    assert isinstance(result, list)
