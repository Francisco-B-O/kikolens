"""Libro de Excel con el análisis: datos, perfil de calidad, estadísticas, receta, reglas y métricas de modelos."""
import pandas as pd

from kikolens.analyze.quality import profile_quality
from kikolens.analyze.stats import basic_statistics
from kikolens.utils.writers import build_workbook


def build_analysis_workbook(df: pd.DataFrame, name: str = "dataset", recipe=None, rules_result: dict | None = None,
                            models: dict | None = None, predictions: pd.DataFrame | None = None,
                            include_data: bool = True) -> bytes:
    """Una hoja por cada parte del trabajo. `models` mapea título -> resultado de un modelo (se exportan sus métricas)."""
    quality = profile_quality(df)
    sheets = {}
    if include_data:
        sheets["Data"] = df
    if predictions is not None:
        sheets["Predictions"] = predictions
    sheets["Column profile"] = quality["columns"]
    if len(df.columns):
        sheets["Statistics"] = basic_statistics(df).round(4).reset_index().rename(columns={"index": "column"})
    issues = pd.DataFrame(quality["issues"], columns=["severity", "message"]) if quality["issues"] else \
        pd.DataFrame({"severity": [], "message": []})
    sheets["Quality issues"] = issues
    if recipe is not None and len(recipe):
        sheets["Recipe"] = recipe.describe()
    if rules_result is not None:
        sheets["Rules"] = rules_result["table"].drop(columns=["id"], errors="ignore")
    if models:
        rows = []
        for title, res in models.items():
            for metric, value in res.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    rows.append({"model": title, "target": res.get("target"), "metric": metric, "value": float(value)})
        if rows:
            sheets["Model metrics"] = pd.DataFrame(rows)
    notes = {"Dataset": name, "Rows": f"{len(df):,}", "Columns": len(df.columns),
             "Quality score": f"{quality['score']:.0f}/100", "Missing cells": f"{quality['missing_pct']:.1f}%",
             "Duplicate rows": f"{quality['duplicate_rows']:,}",
             "Sheets": ", ".join(sheets)}
    return build_workbook(sheets, notes, title=f"{name} · analysis")
