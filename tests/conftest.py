import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def regression_df():
    """DataFrame con target numérico continuo (>20 valores únicos → regresión)."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age":    np.random.randint(20, 60, n).astype(float),
        "income": np.random.randint(20000, 100000, n).astype(float),
        "score":  np.random.rand(n) * 100,
        "target": np.random.rand(n) * 50000 + 30000,
    })


@pytest.fixture
def classification_df():
    """DataFrame con target categórico binario (≤20 valores únicos → clasificación)."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age":    np.random.randint(20, 60, n).astype(float),
        "income": np.random.randint(20000, 100000, n).astype(float),
        "score":  np.random.rand(n) * 100,
        "label":  np.random.choice(["yes", "no"], n),
    })


@pytest.fixture
def correlated_df():
    """DataFrame con dos columnas altamente correlacionadas (>0.85)."""
    np.random.seed(0)
    x = np.random.rand(300)
    return pd.DataFrame({
        "x":     x,
        "y":     x * 2 + np.random.rand(300) * 0.01,  # correlación ~0.99
        "noise": np.random.rand(300),
    })


@pytest.fixture
def simple_df():
    """DataFrame simple para tests de SQL y stats."""
    return pd.DataFrame({
        "name":  ["Alice", "Bob", "Charlie", "Diana"],
        "age":   [30, 25, 35, 28],
        "sales": [1000.0, 1500.0, 800.0, 2000.0],
    })
