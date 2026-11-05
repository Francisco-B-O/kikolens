"""Medidas de rendimiento: cuánto tardan las operaciones principales con un conjunto de datos sintético de N filas, para detectar
regresiones y saber qué esperar con datos grandes. `kikolens benchmark --rows 20000`."""
import platform
import time

import numpy as np
import pandas as pd


def make_data(rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=rows, freq="h"),
        "region": rng.choice(["North", "South", "East", "West"], rows),
        "channel": rng.choice(["Online", "Store"], rows),
        "units": rng.integers(1, 200, rows).astype(float),
        "price": rng.uniform(5, 50, rows),
        "discount": rng.choice([0, 0.05, 0.1, 0.2], rows),
    })
    df["revenue"] = df["units"] * df["price"] * (1 - df["discount"]) + rng.normal(0, 25, rows)
    df.loc[rng.random(rows) < 0.03, "price"] = np.nan
    return df


def _operations(df: pd.DataFrame) -> dict:
    from kikolens.analyze import drift, forecasting, timeseries
    from kikolens.analyze.anomalies import detect_anomalies
    from kikolens.analyze.modellab import run_model_lab
    from kikolens.analyze.quality import auto_clean, profile_quality
    from kikolens.analyze.story import build_story
    from kikolens.analyze.synthetic import synthesize
    from kikolens.sql.runner import run_sql_query

    half = len(df) // 2
    daily = timeseries.prepare_series(df, "date", "revenue", freq="D", agg="sum")
    return {
        "profile quality": lambda: profile_quality(df),
        "automatic cleaning": lambda: auto_clean(df),
        "SQL aggregation": lambda: run_sql_query(df, "select region, channel, sum(revenue) from df group by 1, 2"),
        "drift between halves": lambda: drift.compare_datasets(df.iloc[:half], df.iloc[half:]),
        "anomaly detection": lambda: detect_anomalies(df.drop(columns="date")),
        "daily forecast (5 models)": lambda: forecasting.forecast_models(daily, 14),
        "model lab (3 models, no tuning)": lambda: run_model_lab(df.drop(columns="date"), "revenue", models=["Ridge", "Random Forest", "Hist Gradient Boosting"],
                                                                 folds=3, tune=False),
        "data story": lambda: build_story(df),
        "synthetic data": lambda: synthesize(df.drop(columns="date"), min(len(df), 5000)),
    }


def run_benchmark(rows: int = 20000, repeat: int = 1, only=None) -> dict:
    """Tiempo (el mejor de `repeat`) de cada operación. `only` limita las operaciones por una parte de su nombre."""
    if rows < 200:
        raise ValueError("Use at least 200 rows.")
    if repeat < 1 or repeat > 10:
        raise ValueError("repeat must be between 1 and 10.")
    df = make_data(rows)
    operations = _operations(df)
    if only:
        operations = {k: v for k, v in operations.items() if any(o.lower() in k.lower() for o in only)}
        if not operations:
            raise ValueError("No operation matches that filter.")
    results = []
    for name, run in operations.items():
        best, error = None, ""
        for _ in range(repeat):
            started = time.perf_counter()
            try:
                run()
            except Exception as e:  # una operación que falla se anota y no impide medir las demás
                error = str(e)[:80]
                break
            took = time.perf_counter() - started
            best = took if best is None else min(best, took)
        results.append({"operation": name, "seconds": round(best, 3) if best is not None else None, "rows per second": int(rows / best) if best else None,
                        "status": "ok" if not error else f"failed: {error}"})
    table = pd.DataFrame(results)
    return {"table": table, "rows": rows, "python": platform.python_version(), "machine": f"{platform.system()} {platform.machine()}",
            "total": float(table["seconds"].sum()),
            "summary": f"{len(table)} operations on {rows:,} rows took {table['seconds'].sum():.1f} s in total."}
