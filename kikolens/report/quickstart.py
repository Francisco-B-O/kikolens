"""Cuaderno de bienvenida: crea datos de ejemplo y recorre la API de Python. Se ejecuta tal cual, sin archivos."""
import json

from kikolens.report.notebook import _code, _md


def quickstart_notebook() -> str:
    cells = [
        _md("""
            # KikoLens in a notebook

            Everything the dashboard does is also available from Python. Each call returns a result that shows itself nicely here
            and keeps all the details (`result.table`, `result.figure`, `result.summary`...). Nothing leaves this computer.
            """),
        _code("""
            import numpy as np
            import pandas as pd
            import kikolens as kl

            rng = np.random.default_rng(0)
            n = 360
            df = pd.DataFrame({
                "date": pd.date_range("2024-01-01", periods=n),
                "region": rng.choice(["North", "South", "East", "West"], n),
                "channel": rng.choice(["Online", "Store"], n, p=[0.6, 0.4]),
                "units": rng.integers(20, 160, n).astype(float),
            })
            df["revenue"] = (df["units"] * 14 + 3 * np.arange(n) / 10 + 200 * np.sin(2 * np.pi * np.arange(n) / 7)
                             + rng.normal(0, 60, n)).round(2)
            df.loc[rng.choice(n, 15, replace=False), "units"] = np.nan   # a few gaps, as in real data
            # With your own file: df = kl.load("my_data.csv")
            df.head()
            """),
        _md("## How good are the data?"),
        _code("kl.profile(df)"),
        _md("## A first full analysis in one call"),
        _code("""
            pilot = kl.autopilot(df)
            pilot
            """),
        _md("""
            ## Compare models and see how they do

            The lab keeps rows aside, cross-validates every model and tests the best one once.
            """),
        _code("""
            lab = kl.lab(df, "revenue", models=["Ridge", "Random Forest"], tune=False, drop=["date"])
            lab
            """),
        _code("lab.plot()"),
        _md("## What explains revenue?"),
        _code('kl.regress(df, "revenue", ["units", "region", "channel"])'),
        _md("## Forecast"),
        _code("""
            fc = kl.forecast(df, "date", "revenue", horizon=14)
            fc
            """),
        _code("fc.plot()"),
        _md("""
            ## Ask the data with SQL

            The table is called `df`; only reading is allowed.
            """),
        _code('kl.query(df, "select region, round(sum(revenue)) as revenue from df group by region order by 2 desc")'),
        _md("""
            ## In your own notebooks

            `%load_ext kikolens` adds `%kl_profile df`, `%kl_autopilot df`, `%kl_privacy df` and `%%kl_sql df`.
            Run `kikolens start` for the full dashboard.
            """),
    ]
    notebook = {"nbformat": 4, "nbformat_minor": 4,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "cells": cells}
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
