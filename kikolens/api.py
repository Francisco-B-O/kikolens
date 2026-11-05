"""API de Python pensada para cuadernos (Jupyter, VS Code, Colab) y scripts:

    import kikolens as kl
    df = kl.load("sales.csv")
    kl.profile(df)                    # calidad y problemas
    kl.autopilot(df)                  # primer análisis completo
    lab = kl.lab(df, "revenue")       # compara modelos; lab.leaderboard, lab.pipeline, lab.figure

Cada función devuelve un `Result`: un diccionario con acceso por atributo (`res.summary`, `res.table`) que en un cuaderno se
muestra como un resumen legible con su tabla principal, y que además conserva todo lo que devuelve el análisis."""
import html

import pandas as pd

MAIN_TABLES = ("table", "leaderboard", "matches", "anomalies", "ranking", "segments", "rules", "found", "customers", "columns")
FIGURE_KEYS = ("figure", "plot")


class Result(dict):
    """Resultado de un análisis. Se comporta como un dict y se muestra bien en un cuaderno."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"This result has no '{name}'. Available: {', '.join(sorted(map(str, self)))}") from None

    @property
    def main_table(self):
        for key in MAIN_TABLES:
            value = self.get(key)
            if isinstance(value, pd.DataFrame):
                return value
        return None

    @property
    def main_figure(self):
        for key in FIGURE_KEYS:
            if self.get(key) is not None:
                return self[key]
        plots = self.get("plots")
        if isinstance(plots, dict) and plots:
            return plots.get("leaderboard") or next(iter(plots.values()))
        return None

    def plot(self):
        """El gráfico principal (una figura de Plotly; en un cuaderno basta con dejarla como última línea)."""
        fig = self.main_figure
        if fig is None:
            raise ValueError("This result has no chart.")
        return fig

    def __repr__(self) -> str:
        text = str(self.get("summary") or self.get("narrative") or "")
        return f"<kikolens {self.get('kind', 'result')}: {text[:200]}{'…' if len(text) > 200 else ''}>"

    def _repr_html_(self) -> str:
        parts = [f"<div style='font-family:Georgia,serif;font-size:1.05em;font-weight:bold'>{html.escape(str(self.get('kind', 'result')).title())}</div>"]
        text = self.get("summary") or self.get("narrative")
        if text:
            parts.append(f"<p style='max-width:70em'>{html.escape(str(text).replace('**', '').replace('### ', ''))}</p>")
        for note in self.get("notes", []) or []:
            parts.append(f"<p style='color:#9b3412'>{html.escape(str(note))}</p>")
        table = self.main_table
        if table is not None:
            parts.append(table.head(15).to_html(max_rows=15, border=0, classes="dataframe", float_format=lambda v: f"{v:.4g}"))
            if len(table) > 15:
                parts.append(f"<small>{len(table) - 15:,} more rows in <code>.table</code>…</small>")
        extras = [k for k, v in self.items() if k not in ("kind", "summary", "narrative", "notes") and not isinstance(v, (str, int, float, bool))]
        if extras:
            parts.append(f"<small>Also available: {', '.join(f'<code>.{html.escape(str(k))}</code>' for k in extras[:12])}</small>")
        return "".join(parts)


def _wrap(kind: str, data: dict, summary: str | None = None, **extra) -> Result:
    out = Result(kind=kind)
    out.update(data)
    out.update(extra)
    if summary is not None:
        out["summary"] = summary
    if "table" not in out:
        for key in MAIN_TABLES:
            if isinstance(out.get(key), pd.DataFrame):
                out["table"] = out[key]
                break
    return out


# ── Datos ────────────────────────────────────────────────────────────────────
def load(source, **kwargs) -> pd.DataFrame:
    """Carga un archivo, enlace, hoja de Google, base de datos... igual que el panel."""
    from kikolens.utils.loaders import load_data
    return load_data(source, **kwargs)


def query(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    """Consulta SQL de solo lectura sobre el DataFrame, que se llama `df` dentro de la consulta."""
    from kikolens.sql.runner import run_sql_query
    return run_sql_query(df, sql)


def ask(df: pd.DataFrame, question: str, model: str | None = None) -> Result:
    """Pregunta en lenguaje natural: un modelo local (Ollama) escribe el SQL y se ejecuta."""
    from kikolens.sql.generator import generate_sql_from_text
    sql = generate_sql_from_text(df, question, model_name=model)
    return _wrap("answer", {"sql": sql, "table": query(df, sql)}, f"Question: {question}\nSQL: {sql}")


def profile(df: pd.DataFrame) -> Result:
    """Calidad de los datos: puntuación, problemas y perfil de cada columna."""
    from kikolens.analyze.quality import profile_quality
    q = profile_quality(df)
    worst = [i["message"] for i in q["issues"][:5]]
    return _wrap("profile", q, f"Quality {q['score']:.0f}/100 over {q['rows']:,} rows and {q['columns_count']} columns; "
                                f"{len(q['issues'])} issue(s).", table=q["columns"], notes=worst)


def dictionary(df: pd.DataFrame, descriptions: dict | None = None) -> pd.DataFrame:
    """Una ficha por columna (tipo, vacíos, valores típicos, rangos, avisos)."""
    from kikolens.analyze.dictionary import build_dictionary
    return build_dictionary(df, descriptions)


def scan_privacy(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas que parecen contener datos personales."""
    from kikolens.analyze.privacy import scan
    return scan(df)


# ── Análisis ─────────────────────────────────────────────────────────────────
def autopilot(df: pd.DataFrame, target: str | None = None, **kwargs) -> Result:
    """Primer análisis completo: limpieza, hallazgos, un modelo y una previsión si hay fechas."""
    from kikolens.analyze.autopilot import next_steps, run_autopilot
    res = run_autopilot(df, target, **kwargs)
    table = res["model"]["leaderboard"] if res["model"] else None
    extra = {"table": table} if table is not None else {}
    return _wrap("autopilot", res, res["narrative"], notes=next_steps(res), **extra)


def lab(df: pd.DataFrame, target: str, **kwargs) -> Result:
    """Compara modelos con validación cruzada, ajusta el mejor y lo prueba. `kwargs` como en `run_model_lab`."""
    from kikolens.analyze.modellab import run_model_lab
    res = run_model_lab(df, target, **kwargs)
    return _wrap("model lab", res, res["narrative"].replace("\n", " "), notes=[m["message"] for m in res["leakage"]])


def forecast(df: pd.DataFrame, date: str, value: str, horizon: int = 12, agg: str = "mean", **kwargs) -> Result:
    """Compara cinco modelos con pruebas sobre el pasado y prevé con el mejor."""
    from kikolens.analyze import forecasting, timeseries
    series = timeseries.prepare_series(df, date, value, agg=agg)
    res = forecasting.forecast_models(series, horizon, **kwargs)
    return _wrap("forecast", res, res["summary"], table=res["forecast"], series=series,
                 figure=forecasting.forecast_figure(series, res))


def anomalies(df: pd.DataFrame, **kwargs) -> Result:
    """Filas inusuales y por qué."""
    from kikolens.analyze.anomalies import detect_anomalies
    return _wrap("anomalies", detect_anomalies(df, **kwargs))


def spikes(df: pd.DataFrame, date: str, value: str, **kwargs) -> Result:
    """Periodos inusuales de una serie una vez descontadas tendencia y estacionalidad."""
    from kikolens.analyze.tsanomaly import from_frame
    res = from_frame(df, date, value, **kwargs)
    return _wrap("unusual periods", res, table=res["anomalies"])


def drift(reference: pd.DataFrame, current: pd.DataFrame) -> Result:
    """Qué columnas han cambiado entre dos conjuntos de datos (PSI, KS, chi-cuadrado)."""
    from kikolens.analyze.drift import compare_datasets
    res = compare_datasets(reference, current)
    return _wrap("drift", res, f"Drift level: {res['level']}. {res['summary']}")


def why(df: pd.DataFrame, metric: str | None, date: str, **kwargs) -> Result:
    """Por qué cambió una cifra entre dos periodos: qué valores de qué columnas lo explican."""
    from kikolens.analyze.drivers import waterfall_figure, why_changed
    res = why_changed(df, metric, date, **kwargs)
    top = res["ranking"]["dimension"].iloc[0]
    return _wrap("why it changed", res, table=res["ranking"], figure=waterfall_figure(res, top))


def regress(df: pd.DataFrame, target: str, features=None, **kwargs) -> Result:
    """Regresión explicada: lineal para un número, logística para una columna de dos valores."""
    from kikolens.analyze import regression
    two = df[target].nunique(dropna=True) == 2
    res = regression.logit(df, target, features, **kwargs) if two else regression.ols(df, target, features, **kwargs)
    return _wrap("regression", res, notes=res["sentences"] + res["flags"], figure=regression.coefficient_figure(res))


def survival(df: pd.DataFrame, duration: str, event: str, group: str | None = None) -> Result:
    """Curvas de Kaplan-Meier y test de log-rank entre grupos."""
    from kikolens.analyze.survival import survival_by_group
    return _wrap("survival", survival_by_group(df, duration, event, group))


def cohorts(df: pd.DataFrame, user: str, date: str, period: str = "Month") -> Result:
    """Retención por cohorte."""
    from kikolens.analyze.customers import cohort_retention
    res = cohort_retention(df, user, date, period)
    return _wrap("retention", res, table=res["matrix"].reset_index())


def rfm(df: pd.DataFrame, customer: str, date: str, amount: str) -> Result:
    """Segmentos de clientes por recencia, frecuencia y gasto."""
    from kikolens.analyze.customers import rfm as run_rfm
    res = run_rfm(df, customer, date, amount)
    return _wrap("RFM segments", res, table=res["segments"])


def distribution(series: pd.Series) -> Result:
    """Qué distribución describe mejor una columna numérica."""
    from kikolens.analyze.distributions import fit_distributions
    res = fit_distributions(series)
    return _wrap("distribution fit", res, table=res["table"])


def link(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str, threshold: float = 0.85) -> Result:
    """Cruce aproximado por nombres. `res.merged` es la tabla izquierda con las columnas de la pareja."""
    from kikolens.analyze.linkage import link_records, merge_linked
    res = link_records(left, right, left_on, right_on, threshold)
    return _wrap("record matching", res, merged=merge_linked(left, right, res["matches"]), table=res["matches"])


def score(df: pd.DataFrame, model: str, raw: bool = False) -> pd.DataFrame:
    """Predice con un modelo guardado (nombre o ruta de un .klmodel). Solo abras modelos en los que confíes."""
    from kikolens.analyze.predict import load_bundle, predict_frame, resolve_model
    out, _ = predict_frame(load_bundle(resolve_model(model)), df, apply_recipe=raw)
    return out


def save_model(result: Result, df: pd.DataFrame, name: str | None = None):
    """Guarda el modelo de `lab(...)` en tu registro de modelos y devuelve la ruta."""
    from kikolens.analyze.predict import build_bundle, save_bundle
    if result.get("kind") != "model lab":
        raise ValueError("Pass the result of kikolens.lab(...).")
    return save_bundle(build_bundle(result, df, name=name))


def select_features(df: pd.DataFrame, target: str, **kwargs) -> Result:
    """Qué columnas merece la pena conservar para predecir `target` (consenso de cuatro métodos y curva validada)."""
    from kikolens.analyze.featureselect import select_features as run
    return _wrap("feature selection", run(df, target, **kwargs), figure=None)


def segment(df: pd.DataFrame, method: str = "gmm", columns=None, **kwargs) -> Result:
    """Segmenta filas: "gmm" (mezcla gaussiana), "hierarchical" o "dbscan". `res.labels` es el grupo de cada fila."""
    from kikolens.analyze import segments
    runners = {"gmm": segments.gmm_auto, "hierarchical": segments.hierarchical, "dbscan": segments.dbscan_auto}
    if method not in runners:
        raise ValueError("method must be gmm, hierarchical or dbscan.")
    res = runners[method](df, columns, **kwargs)
    return _wrap("segments", res, table=res["profiles"])


def ab_sequential(df: pd.DataFrame, group: str, outcome: str, control, order: str | None = None, **kwargs) -> Result:
    """Prueba A/B secuencial (mSPRT): válida aunque se mire tras cada observación."""
    from kikolens.analyze.sequential import from_frame
    return _wrap("sequential test", from_frame(df, group, outcome, control, order, **kwargs))


def granger(df: pd.DataFrame, date: str, cause: str, effect: str, **kwargs) -> Result:
    """¿Ayuda el pasado de `cause` a predecir `effect`?"""
    from kikolens.analyze.multivar import granger as run
    return _wrap("granger", run(df, date, cause, effect, **kwargs))


def forecast_drivers(df: pd.DataFrame, date: str, target: str, drivers: list, **kwargs) -> Result:
    """Previsión que usa también otras columnas, con escenarios y comprobación de si ayudan."""
    from kikolens.analyze.multivar import forecast_with_drivers
    res = forecast_with_drivers(df, date, target, drivers, **kwargs)
    return _wrap("forecast with drivers", res, table=res["forecast"])


def recommender(df: pd.DataFrame, user: str, item: str, rating: str | None = None):
    """Recomendador por filtrado colaborativo: `.recommend(persona)`, `.similar_items(artículo)`, `.popular()`."""
    from kikolens.analyze.recommend import fit
    return fit(df, user, item, rating)


def synthesize(df: pd.DataFrame, n: int | None = None, seed: int = 0) -> Result:
    """Datos sintéticos con las mismas distribuciones y correlaciones; `res.data` es la tabla y `res.report` mide fidelidad y privacidad."""
    from kikolens.analyze.synthetic import report, synthesize as run
    res = run(df, n, seed)
    return _wrap("synthetic data", res, table=res["data"], report=report(df, res["data"]))


def runs(target: str | None = None) -> pd.DataFrame:
    """El historial de ejecuciones del laboratorio."""
    from kikolens.analyze.runs import list_runs
    return list_runs(target)


def chart(df: pd.DataFrame, request: str, model: str | None = None) -> Result:
    """Un gráfico descrito con palabras. Un modelo local (Ollama) escribe una especificación que se valida y se dibuja: `res.figure`, `res.table`, `res.spec`."""
    from kikolens.ai.analyst import ask_chart
    from kikolens.ai.client import LLM, ollama_transport
    fig, table, spec = ask_chart(df, request, LLM(ollama_transport(model=model)) if model else LLM())
    return _wrap("chart", {"figure": fig, "table": table, "spec": spec}, f"{spec['kind']} chart: {len(table):,} rows drawn")


def analyst(df: pd.DataFrame, model: str | None = None, samples: bool = True):
    """Una conversación con los datos: `bot = kl.analyst(df)`, `bot.ask("por qué bajaron las ventas")` (ver `kikolens.ai.analyst.Analyst`)."""
    from kikolens.ai.analyst import Analyst
    from kikolens.ai.client import LLM, ollama_transport
    return Analyst(df, LLM(ollama_transport(model=model)) if model else LLM(), samples)


def classify_text(texts: pd.Series, labels: list, model: str | None = None, limit: int = 300) -> pd.DataFrame:
    """Clasifica textos en etiquetas que eliges, sin entrenar nada (un modelo local lee cada texto)."""
    from kikolens.ai.analyst import classify_texts
    from kikolens.ai.client import LLM, ollama_transport
    return classify_texts(texts, labels, LLM(ollama_transport(model=model)) if model else LLM(), limit=limit)


def formula(df: pd.DataFrame, instruction: str, model: str | None = None) -> dict:
    """Una columna nueva descrita con palabras: devuelve `name`, `expression` y una vista previa, ya comprobada con el intérprete seguro."""
    from kikolens.ai.analyst import suggest_formula
    from kikolens.ai.client import LLM, ollama_transport
    return suggest_formula(df, instruction, LLM(ollama_transport(model=model)) if model else LLM())


def effect(df: pd.DataFrame, treatment: str, outcome: str, covariates: list | None = None, method: str = "aipw", **kwargs) -> Result:
    """Efecto causal de un tratamiento sobre un resultado ajustando por covariables (regresión, ponderación, emparejamiento y doblemente robusto)."""
    from kikolens.analyze.causality import estimate_effect
    res = estimate_effect(df, treatment, outcome, covariates, method=method, **kwargs)
    return _wrap("causal effect", res)


def uplift(df: pd.DataFrame, treatment: str, outcome: str, covariates: list | None = None, **kwargs) -> Result:
    """Efecto del tratamiento por fila: a quién conviene tratar, con curva de Qini."""
    from kikolens.analyze.causality import uplift as run
    res = run(df, treatment, outcome, covariates, **kwargs)
    return _wrap("uplift", res, table=res["deciles"])


def discontinuity(df: pd.DataFrame, running: str, outcome: str, cutoff: float, **kwargs) -> Result:
    """Efecto de pasar un umbral (discontinuidad en la regresión)."""
    from kikolens.analyze.causality import regression_discontinuity
    return _wrap("regression discontinuity", regression_discontinuity(df, running, outcome, cutoff, **kwargs))


def synthetic_control(df: pd.DataFrame, unit: str, time: str, outcome: str, treated_unit, intervention, **kwargs) -> Result:
    """Efecto sobre una unidad frente a una mezcla de unidades de control, con pruebas de placebo."""
    from kikolens.analyze.causality import synthetic_control as run
    res = run(df, unit, time, outcome, treated_unit, intervention, **kwargs)
    return _wrap("synthetic control", res, table=res["weights"].rename("weight").rename_axis("unit").reset_index())


def instrumental(df: pd.DataFrame, outcome: str, endogenous: str, instruments: list, covariates: list | None = None) -> Result:
    """Variables instrumentales (mínimos cuadrados en dos etapas)."""
    from kikolens.analyze.causality import iv_2sls
    return _wrap("instrumental variables", iv_2sls(df, outcome, endogenous, instruments, covariates))


FUNCTIONS = ("load", "query", "ask", "profile", "dictionary", "scan_privacy", "autopilot", "lab", "forecast", "anomalies", "spikes",
             "drift", "why", "regress", "survival", "cohorts", "rfm", "distribution", "link", "score", "save_model", "Result", "select_features", "segment", "ab_sequential", "granger", "forecast_drivers",
             "recommender", "synthesize", "runs", "chart", "analyst", "classify_text", "formula", "effect", "uplift", "discontinuity", "synthetic_control", "instrumental")
