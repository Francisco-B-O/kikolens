"""Piloto automático: recorre un dataset de principio a fin y devuelve lo que un analista haría primero. Limpia con
pasos reproducibles (una receta), describe los datos, busca hallazgos, elige una columna a predecir si no se indica,
compara unos pocos modelos y, si hay fechas, prevé la serie. Cada fase que falla se anota y no detiene a las demás."""
import re

import pandas as pd

from kikolens.analyze import forecasting, timeseries
from kikolens.analyze.discovery import run_discovery_engine
from kikolens.analyze.modellab import candidates, detect_task, run_model_lab
from kikolens.analyze.quality import auto_clean_steps, infer_types, profile_quality
from kikolens.pipeline.recipe import Recipe
from kikolens.utils.dtypes import is_text

TARGET_HINTS = ("target", "label", "outcome", "churn", "class", "status", "revenue", "sales", "price", "amount",
                "total", "profit", "score", "y")
FAST_MODELS = ("Linear Regression", "Logistic Regression", "Random Forest", "Hist Gradient Boosting")


def _is_identifier(series: pd.Series) -> bool:
    s = series.dropna()
    if len(s) < 20 or s.nunique() != len(s):
        return False
    return not pd.api.types.is_float_dtype(s) or bool((s % 1 == 0).all())


def column_roles(df: pd.DataFrame) -> dict:
    """Clasifica cada columna: fecha, identificador, constante, texto libre, categórica o numérica."""
    roles = {}
    for col in df.columns:
        s = df[col]
        if s.nunique(dropna=True) <= 1:
            roles[col] = "constant"
        elif pd.api.types.is_datetime64_any_dtype(s):
            roles[col] = "date"
        elif pd.api.types.is_bool_dtype(s):
            roles[col] = "category"
        elif pd.api.types.is_numeric_dtype(s):
            roles[col] = "identifier" if _is_identifier(s) and re.search(r"(^|_)(id|key|code|index)$", str(col).lower()) \
                else "number"
        elif is_text(s):
            lengths = s.dropna().astype(str).str.len()
            if len(s.dropna()) and s.nunique() == s.notna().sum() and len(s) > 20:
                roles[col] = "identifier"
            elif len(lengths) and lengths.mean() > 40 and s.nunique() > 0.5 * len(s):
                roles[col] = "text"
            else:
                roles[col] = "category"
        else:
            roles[col] = "category"
    return roles


def suggest_target(df: pd.DataFrame, roles: dict | None = None) -> str | None:
    """La columna que más probablemente se quiera predecir: nombres típicos primero, si no la última columna útil."""
    roles = roles or column_roles(df)
    usable = [c for c in df.columns if roles[c] in ("number", "category")
              and not (roles[c] == "category" and df[c].nunique() > 20) and df[c].notna().sum() >= 30]
    if not usable:
        return None
    for hint in TARGET_HINTS:
        for col in usable:
            name = str(col).lower()
            if name == hint or name.endswith("_" + hint) or name.startswith(hint + "_"):
                return col
    return usable[-1]


def _headline(insight: dict) -> str:
    return re.sub(r"\*\*", "", insight.get("message", "")).strip()


def _series_for(df: pd.DataFrame, roles: dict, target: str | None):
    dates = [c for c, r in roles.items() if r == "date"]
    if not dates:
        return None
    value = target if target and roles.get(target) == "number" else next((c for c, r in roles.items() if r == "number"), None)
    if value is None:
        return None
    series = timeseries.prepare_series(df, dates[0], value, agg="mean")
    return dates[0], value, series


def run_autopilot(df: pd.DataFrame, target: str | None = None, forecast_horizon: int = 12, fast: bool = True,
                  progress=None) -> dict:
    """Devuelve un dict con `recipe`, `clean` (el DataFrame limpio), `findings`, `model`, `forecast`, `steps` (el
    registro de lo hecho), `problems` (fases que fallaron) y `narrative`."""
    if df is None or df.empty or df.shape[1] < 2:
        raise ValueError("The autopilot needs a dataset with at least 2 columns and some rows.")
    if target is not None and target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    def tick(fraction, text):
        if progress:
            progress(fraction, text)

    log, problems = [], []
    tick(0.05, "Cleaning")
    before = profile_quality(df)
    clean, steps = auto_clean_steps(df)
    recipe = Recipe()
    for name, params, message in steps:
        recipe.add(name, params, message)
        if message:
            log.append(message)
    after = profile_quality(clean)

    typed, converted = infer_types(clean)
    roles = column_roles(typed)
    chosen = target or suggest_target(typed, roles)
    findings = []
    tick(0.2, "Looking for findings")
    try:
        findings = [{"severity": i.get("severity", "low"), "message": _headline(i)} for i in run_discovery_engine(typed)]
    except Exception as e:  # pragma: no cover - defensivo
        problems.append(f"Findings: {e}")

    model = None
    if chosen is not None and roles.get(chosen) in ("number", "category"):
        tick(0.35, f"Modeling {chosen}")
        try:
            task = detect_task(typed[chosen].dropna())
            pool = candidates(task)
            names = [m for m in pool if m.startswith("Baseline")] + [m for m in FAST_MODELS if m in pool]
            drop = [c for c, r in roles.items() if r in ("identifier", "text") and c != chosen]
            model = run_model_lab(typed, chosen, models=names if fast else None, folds=3 if fast else 5,
                                  tune=not fast, drop=drop)
            log.append(f"Compared {len(model['leaderboard'])} models for '{chosen}'; the best was {model['best_model']}.")
        except ValueError as e:
            problems.append(f"Model: {e}")

    forecast = None
    tick(0.85, "Forecasting")
    try:
        found = _series_for(typed, roles, chosen)
        if found and len(found[2]) >= 30:
            date_col, value_col, series = found
            forecast = {"date": date_col, "value": value_col, "series": series,
                        "result": forecasting.forecast_models(series, forecast_horizon)}
            log.append(f"Forecast {value_col} {forecast_horizon} periods ahead with {forecast['result']['best']}.")
        elif found:
            problems.append("Forecast: fewer than 30 dated periods.")
    except ValueError as e:
        problems.append(f"Forecast: {e}")
    tick(1.0, "Done")

    return {"recipe": recipe, "clean": clean, "typed": typed, "converted": converted, "roles": roles, "target": chosen,
            "quality_before": before, "quality_after": after, "findings": findings, "model": model,
            "forecast": forecast, "steps": log, "problems": problems,
            "narrative": _narrative(df, before, after, roles, chosen, model, forecast, findings)}


def _narrative(df, before, after, roles, target, model, forecast, findings) -> str:
    kinds = pd.Series(roles).value_counts().to_dict()
    parts = [f"The data has {len(df):,} rows and {df.shape[1]} columns ({', '.join(f'{n} {k}' for k, n in kinds.items())})."]
    if after["score"] != before["score"]:
        parts.append(f"Safe cleaning raised the quality score from {before['score']:.0f} to {after['score']:.0f}.")
    else:
        parts.append(f"The quality score is {after['score']:.0f}/100.")
    if findings:
        parts.append(f"{len(findings)} finding(s) deserve a look; the first is: {findings[0]['message']}")
    if model:
        m, primary = model["metrics"], model["primary_metric"]
        base = model.get("baseline_cv")
        parts.append(f"To predict '{target}', {model['best_model']} reached {primary} {m[primary]:.3f} on rows it had not seen"
                     + (f" (a naive guess scores {base:.3f})." if base is not None else "."))
    if forecast:
        parts.append(forecast["result"]["summary"])
    if not model and not forecast:
        parts.append("No column was suitable to predict and no dated series was found, so it stopped at the data description.")
    return " ".join(parts)


def next_steps(result: dict) -> list:
    """Sugerencias concretas a partir de lo que encontró el piloto."""
    out = []
    model = result["model"]
    if model:
        if model["leakage"]:
            out.append("Check the columns flagged as possible data leaks before trusting the model.")
        base = model.get("baseline_cv")
        if base is not None and model["metrics"][model["primary_metric"]] - base < 0.05:
            out.append("The model barely beats a naive guess. More informative columns would help more than tuning.")
        else:
            out.append("Run the Model lab with tuning on, then look at where the model fails.")
    if result["quality_after"]["missing_pct"] > 5:
        out.append("Many cells are empty. Look at the missing-data patterns before filling them.")
    if result["forecast"] is None and any(r == "date" for r in result["roles"].values()):
        out.append("There is a date column but too few periods to forecast; aggregate to a coarser period.")
    if not out:
        out.append("Open the Panel and pin the figures you want to follow.")
    return out
