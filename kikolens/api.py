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
    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found.")
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


def simulate(inputs, expression: str, n: int = 10000, correlations=None, **kwargs) -> Result:
    """Monte Carlo: `inputs` es un diccionario de distribuciones o un DataFrame (se remuestrea y se conserva la correlación)."""
    from kikolens.analyze import simulation as sim
    if isinstance(inputs, pd.DataFrame):
        inputs, found = sim.inputs_from_data(inputs)
        correlations = found if correlations is None else correlations
    res = sim.monte_carlo(inputs, expression, n, correlations, **kwargs)
    return _wrap("simulation", res, table=res["sensitivity"])


def optimize(expression_or_objective, bounds=None, constraints=None, **kwargs) -> Result:
    """Optimiza una fórmula (texto) con límites `{"x": (0, 10)}` y restricciones `["x + y <= 20"]`, o un programa lineal si le das un diccionario de coeficientes."""
    from kikolens.analyze import simulation as sim
    if isinstance(expression_or_objective, dict):
        res = sim.linear_program(expression_or_objective, constraints or [], bounds, **kwargs)
        return _wrap("linear program", res, table=res["constraints"])
    res = sim.optimize_expression(expression_or_objective, bounds or {}, constraints, **kwargs)
    return _wrap("optimization", res, table=res["solution"].rename("value").rename_axis("variable").reset_index())


def marketing_mix(df: pd.DataFrame, date: str, target: str, channels: list, controls: list | None = None, **kwargs) -> Result:
    """Cuánto aporta cada canal a las ventas (arrastre, saturación, tendencia y estacionalidad). Pásalo a `budget` para repartir el gasto."""
    from kikolens.analyze.marketing import marketing_mix as run
    return _wrap("marketing mix", run(df, date, target, channels, controls, **kwargs))


def budget(model, total: float | None = None, low: float = 0.5, high: float = 2.0) -> Result:
    """Reparto del presupuesto entre canales con las curvas de respuesta de `marketing_mix`."""
    from kikolens.analyze.marketing import optimize_budget
    return _wrap("budget", optimize_budget(dict(model), total, low, high))


def elasticity(df: pd.DataFrame, quantity: str, price: str, controls: list | None = None, **kwargs) -> Result:
    """Elasticidad-precio de la demanda (log-log), por grupo y con instrumento si el precio depende de la demanda."""
    from kikolens.analyze.marketing import price_elasticity
    res = price_elasticity(df, quantity, price, controls, **kwargs)
    return _wrap("price elasticity", res, table=res["groups"] if res["groups"] is not None else res["curve"])


def panel(df: pd.DataFrame, path: str | None = None, definition=None, title: str = "Panel", where: list | None = None):
    """Un panel de métricas, gráficos y tablas. Con `path` termina en .html, .pptx o .docx y lo escribe; sin él devuelve el `Panel` para ir añadiendo tarjetas.
    `definition` es un `Panel`, un diccionario/JSON guardado o nada (se propone uno); `where` son condiciones como "region in North,West"."""
    import json
    from kikolens.analyze import panel as pn
    if definition is None:
        built = pn.suggest_panel(df, title)
    elif isinstance(definition, pn.Panel):
        built = definition
    else:
        built = pn.Panel.from_json(definition if isinstance(definition, str) else json.dumps(definition))
    built.filters += [pn.parse_filter(w) for w in where or []]
    if path is None:
        return built
    from pathlib import Path
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in (".html", ".pptx", ".docx"):
        raise ValueError("The file must end in .html, .pptx or .docx.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".html":
        target.write_text(pn.panel_html(df, built), encoding="utf-8")
    else:
        from kikolens.report import office
        target.write_bytes((office.panel_pptx if suffix == ".pptx" else office.panel_docx)(df, built))
    return target


def run(name: str, df: pd.DataFrame, **args) -> Result:
    """Ejecuta un análisis añadido por un plugin (ver `kl.list_plugins()` y `docs/plugins.md`)."""
    from kikolens import plugins
    return _wrap(name, plugins.run(name, df, **args))


def list_plugins() -> dict:
    """Los plugins cargados: sus análisis (con parámetros), los formatos que leen y los errores al cargarlos."""
    from kikolens import plugins as registry
    return {"analyses": [{"name": a.name, "description": a.description, "plugin": a.plugin, "params": a.params} for a in registry.analyses().values()],
            "loaders": sorted(registry.loaders()), "loaded": registry.loaded(), "errors": registry.errors()}


def _series(df: pd.DataFrame, date: str, value: str, agg: str = "mean") -> pd.Series:
    from kikolens.analyze import timeseries as ts
    return ts.prepare_series(df, date, value, agg=agg)


def stationarity(df: pd.DataFrame, date: str, value: str, agg: str = "mean", regression: str = "c") -> Result:
    """¿Es estacionaria la serie? ADF y KPSS juntos, y cuántas diferencias hacen falta."""
    from kikolens.analyze import classic
    return _wrap("stationarity", classic.stationarity(_series(df, date, value, agg), regression))


def autocorrelation(df: pd.DataFrame, date: str, value: str, agg: str = "mean", nlags: int | None = None) -> Result:
    """Autocorrelación y autocorrelación parcial con bandas, y la prueba de Ljung-Box."""
    from kikolens.analyze import classic
    return _wrap("autocorrelation", classic.acf_pacf(_series(df, date, value, agg), nlags))


def decompose_stl(df: pd.DataFrame, date: str, value: str, period: int | None = None, agg: str = "mean", robust: bool = True) -> Result:
    """Tendencia, estacionalidad y resto con STL (robusto a valores extremos)."""
    from kikolens.analyze import classic
    return _wrap("stl", classic.stl(_series(df, date, value, agg), period, robust))


def arima(df: pd.DataFrame, date: str, value: str, order: tuple | None = None, seasonal: tuple = (0, 0, 0), period: int | None = None, horizon: int = 12,
          agg: str = "mean", level: float = 0.95) -> Result:
    """ARIMA estacional con intervalos. Sin `order` busca el mejor por AICc (`res.candidates`); con él ajusta ese modelo."""
    from kikolens.analyze import classic
    series = _series(df, date, value, agg)
    res = classic.auto_arima(series, period, horizon, level=level) if order is None else classic.arima(series, order, seasonal, period, horizon, level)
    return _wrap("arima", res)


def governance(action: str | None = None, model: str | None = None, **kwargs):
    """Gobierno de modelos. Sin argumentos devuelve el resumen de todos; con `action` (`register`, `validate`, `promote`, `approve`, `rollback`, `checks`, `update`)
    y el nombre de un modelo guardado ejecuta esa acción (ver `kikolens.analyze.governance` para los parámetros)."""
    from kikolens.analyze import governance as gv
    if action is None:
        return gv.overview()
    actions = {"register": gv.register, "validate": gv.validate, "promote": gv.promote, "approve": gv.approve, "checks": gv.checks, "update": gv.update_card}
    if action == "rollback":
        return gv.rollback(model, **kwargs)
    if action not in actions:
        raise ValueError(f"Unknown action '{action}'. Use one of: {', '.join([*actions, 'rollback'])}.")
    if model is None:
        raise ValueError("Say which model.")
    return actions[action](model, **kwargs)


def model_card(model: str) -> str:
    """La ficha de un modelo registrado para gobierno, en Markdown."""
    from kikolens.analyze import governance as gv
    return gv.model_card(model)


def topics(df: pd.DataFrame, text: str, k: int | None = None, date: str | None = None, **kwargs) -> Result:
    """Temas en una columna de texto (NMF o LDA); sin `k` se elige por coherencia. Con `date` añade cómo cambian por periodo (`res.over_time`)."""
    from kikolens.analyze import textmining
    if date:
        return _wrap("topics", textmining.topics_over_time(df, text, date, k, **kwargs))
    return _wrap("topics", textmining.topics(df[text], k, **kwargs))


def near_duplicates(df: pd.DataFrame, text: str, threshold: float = 0.9) -> Result:
    """Textos casi idénticos agrupados (erratas, copias con cambios pequeños)."""
    from kikolens.analyze import textmining
    return _wrap("near duplicates", textmining.find_duplicates(df[text], threshold))


def similar(df: pd.DataFrame, text: str, query, n: int = 5) -> Result:
    """Los textos más parecidos a una frase o a la fila con esa etiqueta de índice."""
    from kikolens.analyze import textmining
    table = textmining.similar_texts(df[text], query, n)
    return _wrap("similar texts", {"table": table}, f"Closest text: row {table['row'].iloc[0]} (similarity {table['similarity'].iloc[0]:.2f}).")


def text_patterns(df: pd.DataFrame, text: str, kinds: list | None = None) -> Result:
    """Correos, enlaces, teléfonos, importes, porcentajes, fechas, etiquetas y menciones en un texto; `res.found` lista cada coincidencia."""
    from kikolens.analyze import textmining
    return _wrap("text patterns", textmining.extract_patterns(df[text], kinds))


def keyphrases(df: pd.DataFrame, text: str, n: int = 15, **kwargs) -> Result:
    """Frases clave (RAKE) de una columna de texto."""
    from kikolens.analyze import textmining
    table = textmining.keyphrases(df[text], n, **kwargs)
    return _wrap("keyphrases", {"table": table}, f"Top phrase: {table['phrase'].iloc[0]}.")


def readability(df: pd.DataFrame, text: str, language: str | None = None) -> Result:
    """Facilidad de lectura por texto (Flesch en inglés, Fernández-Huerta en español)."""
    from kikolens.analyze import textmining
    return _wrap("readability", textmining.readability(df[text], language))


def network(df: pd.DataFrame, source: str, target: str, weight: str | None = None, directed: bool = False, **kwargs) -> Result:
    """Análisis de una red a partir de una lista de enlaces: nodos importantes (PageRank, intermediación), comunidades, conectividad y distancias."""
    from kikolens.analyze import network as nw
    return _wrap("network", nw.analyze(df, source, target, weight, directed, **kwargs))


def network_path(df: pd.DataFrame, source: str, target: str, start, end, weight: str | None = None, directed: bool = False) -> Result:
    """El camino con menos saltos entre dos nodos."""
    from kikolens.analyze import network as nw
    return _wrap("network path", nw.shortest_path(df, source, target, start, end, weight, directed))


def suggest_links(df: pd.DataFrame, source: str, target: str, weight: str | None = None, top: int = 10) -> Result:
    """Enlaces que faltan: pares sin enlace que comparten muchos vecinos (Adamic-Adar)."""
    from kikolens.analyze import network as nw
    return _wrap("suggested links", nw.suggest_links(df, source, target, weight, top))


def network_robustness(df: pd.DataFrame, source: str, target: str, weight: str | None = None, steps: int = 10) -> Result:
    """Cuánto se rompe la red al quitar los nodos más conectados frente a quitar nodos al azar."""
    from kikolens.analyze import network as nw
    return _wrap("network robustness", nw.robustness(df, source, target, weight, steps))


def _geo_pair(df: pd.DataFrame, lat: str | None, lon: str | None) -> tuple:
    """Las columnas de latitud y longitud: las que se dan o las que se detectan por el nombre y el rango."""
    if lat and lon:
        return lat, lon
    from kikolens.analyze import geo
    found = geo.detect_coordinates(df)
    if not found:
        raise ValueError("No latitude and longitude columns were found; give them with lat= and lon=.")
    return found[0]


def geo_check(df: pd.DataFrame, lat: str | None = None, lon: str | None = None) -> Result:
    """Qué filas tienen coordenadas utilizables (y qué huele mal: intercambiadas, fuera de rango, 0,0, repetidas). Detecta las columnas si no se dan."""
    from kikolens.analyze import geo
    pair = _geo_pair(df, lat, lon)
    return _wrap("geo check", geo.check_coordinates(df, *pair), pair=pair)


def geo_distances(df: pd.DataFrame, centers, lat: str | None = None, lon: str | None = None, **kwargs) -> Result:
    """Kilómetros hasta un punto `(lat, lon)` o hasta el centro más cercano de una tabla (`center_lat=`, `center_lon=`, `center_name=`), con el reparto por centro."""
    from kikolens.analyze import geo
    return _wrap("geo distances", geo.distances(df, *_geo_pair(df, lat, lon), centers, **kwargs))


def geo_hotspots(df: pd.DataFrame, value: str | None = None, lat: str | None = None, lon: str | None = None, **kwargs) -> Result:
    """Zonas con muchos puntos próximos (DBSCAN sobre la esfera)."""
    from kikolens.analyze import geo
    return _wrap("geo hotspots", geo.hotspots(df, *_geo_pair(df, lat, lon), value, **kwargs))


def spatial_autocorrelation(df: pd.DataFrame, value: str, lat: str | None = None, lon: str | None = None, **kwargs) -> Result:
    """I de Moran y focos calientes y fríos (Getis-Ord Gi*) de una columna numérica."""
    from kikolens.analyze import geo
    return _wrap("spatial autocorrelation", geo.autocorrelation(df, *_geo_pair(df, lat, lon), value, **kwargs))


def geo_grid(df: pd.DataFrame, value: str | None = None, lat: str | None = None, lon: str | None = None, **kwargs) -> Result:
    """Resumen por celdas cuadradas de unos kilómetros."""
    from kikolens.analyze import geo
    return _wrap("geo grid", geo.grid_summary(df, *_geo_pair(df, lat, lon), value, **kwargs))


def experiment(df: pd.DataFrame, group: str, outcome: str, control, pre: str | None = None, date: str | None = None, segments: list | None = None,
               expected: dict | None = None) -> Result:
    """Análisis completo de un experimento: reparto (SRM), efecto de cada grupo frente al control (con CUPED si das `pre`), efecto en el tiempo y por segmento, y una conclusión."""
    from kikolens.analyze import experiment as ex
    return _wrap("experiment", ex.analyze_experiment(df, group, outcome, control, pre, date, segments, expected))


def srm(df: pd.DataFrame, group: str, expected: dict | None = None) -> Result:
    """¿Llegó a cada grupo la proporción planeada? Un reparto roto invalida el experimento."""
    from kikolens.analyze import experiment as ex
    return _wrap("sample ratio", ex.srm(df, group, expected))


def cuped(df: pd.DataFrame, group: str, outcome: str, pre: str, control) -> Result:
    """Reduce el ruido de un experimento con lo que cada fila hacía antes (CUPED)."""
    from kikolens.analyze import experiment as ex
    return _wrap("cuped", ex.cuped(df, group, outcome, pre, control))


def control_chart(df: pd.DataFrame, value: str, chart: str = "individuals", **kwargs) -> Result:
    """Carta de control (`individuals`, `xbar`, `p`, `u`, `c`) con las reglas de señal; `baseline=` fija cuántos puntos iniciales calculan los límites."""
    from kikolens.analyze import spc
    return _wrap("control chart", spc.control_chart(df, value, chart, **kwargs))


def process_capability(df: pd.DataFrame, value: str, lsl: float | None = None, usl: float | None = None, target: float | None = None) -> Result:
    """Cp, Cpk, Pp y Ppk frente a los límites de especificación."""
    from kikolens.analyze import spc
    return _wrap("capability", spc.capability(df[value], lsl, usl, target))


def shift_detection(df: pd.DataFrame, value: str, target: float | None = None, sigma: float | None = None, baseline: int | None = None) -> Result:
    """CUSUM y EWMA juntos para ver cambios pequeños y sostenidos: `res.cusum`, `res.ewma`."""
    from kikolens.analyze import spc
    one, two = spc.cusum(df[value], target, sigma, baseline=baseline), spc.ewma(df[value], target=target, sigma=sigma, baseline=baseline)
    return _wrap("shift detection", {"cusum": one, "ewma": two, "notes": [], "summary": f"CUSUM: {one['summary']} EWMA: {two['summary']}"})


def abc_xyz(df: pd.DataFrame, item: str, quantity: str, date: str | None = None, value: str | None = None, freq: str = "W") -> Result:
    """Clasificación ABC (valor) y, con `date`, XYZ (variación de la demanda) y clase de demanda de cada artículo."""
    from kikolens.analyze import inventory
    return _wrap("abc xyz", inventory.classify_items(df, item, quantity, date, value, freq))


def newsvendor(unit_cost: float, price: float, salvage: float = 0.0, mean: float | None = None, sd: float | None = None, samples=None) -> Result:
    """Cuánto pedir de un producto de una sola venta (con demanda normal o con ventas pasadas en `samples`)."""
    from kikolens.analyze import inventory
    return _wrap("newsvendor", inventory.newsvendor(unit_cost, price, salvage, mean, sd, samples))


def eoq(annual_demand: float, order_cost: float, holding_cost: float, unit_cost: float = 0.0) -> Result:
    """Lote económico de pedido."""
    from kikolens.analyze import inventory
    return _wrap("eoq", inventory.eoq(annual_demand, order_cost, holding_cost, unit_cost))


def reorder_policy(demand_mean: float, demand_sd: float, lead_time: float, lead_time_sd: float = 0.0, service_level: float = 0.95, order_quantity: float | None = None) -> Result:
    """Stock de seguridad y punto de pedido para un nivel de servicio."""
    from kikolens.analyze import inventory
    return _wrap("reorder policy", inventory.reorder_policy(demand_mean, demand_sd, lead_time, lead_time_sd, service_level, order_quantity))


def simulate_stock_policy(demand_mean: float, demand_sd: float, lead_time: int, order_quantity: float, reorder_point: float, **kwargs) -> Result:
    """Simula una política (s, Q) durante muchos años para ver el servicio real."""
    from kikolens.analyze import inventory
    return _wrap("stock simulation", inventory.simulate_policy(demand_mean, demand_sd, lead_time, order_quantity, reorder_point, **kwargs))


def inventory_plan(df: pd.DataFrame, item: str, date: str, quantity: str, lead_time_days: float, **kwargs) -> Result:
    """Una política de reposición para cada artículo de una tabla de ventas."""
    from kikolens.analyze import inventory
    return _wrap("inventory plan", inventory.plan(df, item, date, quantity, lead_time_days, **kwargs))


def attribution(df: pd.DataFrame, user: str, time: str, channel: str, converted: str, value: str | None = None, **kwargs) -> Result:
    """Atribución multicanal: reparte el mérito de las conversiones entre los canales con siete modelos (primer y último contacto, lineal, caída en el tiempo, posición, Markov y Shapley)."""
    from kikolens.analyze import attribution as at
    res = at.attribute(df, user, time, channel, converted, value, **kwargs)
    return _wrap("attribution", res, table=res["table"])


def likert(df: pd.DataFrame, items: list, scale: tuple | None = None, reverse: list | None = None) -> Result:
    """Resumen de preguntas de escala (números o textos como «Totalmente de acuerdo»): media, % favorable y desfavorable y reparto."""
    from kikolens.analyze import survey
    return _wrap("likert", survey.likert_summary(df, items, scale, reverse))


def cronbach(df: pd.DataFrame, items: list, reverse: list | None = None) -> Result:
    """Alfa de Cronbach de un grupo de preguntas, con la correlación de cada una con el resto y el alfa si se quitara."""
    from kikolens.analyze import survey
    return _wrap("reliability", survey.reliability(df, items, reverse))


def nps(df: pd.DataFrame, score: str, group: str | None = None) -> Result:
    """Net Promoter Score (promotores menos detractores) con su intervalo y, con `group`, por grupo."""
    from kikolens.analyze import survey
    return _wrap("nps", survey.nps(df, score, group))


def crosstab(df: pd.DataFrame, row: str, column: str) -> Result:
    """Tabla cruzada con chi-cuadrado, V de Cramér y residuos que señalan las celdas que se salen de lo esperado."""
    from kikolens.analyze import survey
    return _wrap("crosstab", survey.crosstab(df, row, column))


def key_drivers(df: pd.DataFrame, outcome: str, items: list) -> Result:
    """Qué preguntas pesan más en la valoración global (pesos relativos) y cuáles hay que mejorar primero."""
    from kikolens.analyze import survey
    return _wrap("key drivers", survey.key_drivers(df, outcome, items))


def compare_items(df: pd.DataFrame, items: list, group: str) -> Result:
    """Qué preguntas puntúan distinto entre grupos (pruebas para datos ordinales, con corrección por comparaciones múltiples)."""
    from kikolens.analyze import survey
    return _wrap("item comparison", survey.compare_groups(df, items, group))


def response_quality(df: pd.DataFrame, items: list, time_seconds: str | None = None) -> Result:
    """Quién contesta siempre lo mismo, deja muchas en blanco o contesta demasiado rápido."""
    from kikolens.analyze import survey
    return _wrap("response quality", survey.response_quality(df, items, time_seconds))


def concepts() -> pd.DataFrame:
    """Los conceptos del laboratorio: simulaciones que enseñan por qué la estadística funciona como funciona."""
    from kikolens.analyze import concepts as cp
    return pd.DataFrame([{"concept": k, "title": v["title"], "question": v["question"], "parameters": ", ".join(v["params"])} for k, v in cp.CONCEPTS.items()])


def concept(name: str, seed: int = 0, **params) -> Result:
    """Ejecuta un concepto del laboratorio (`kl.concepts()` los lista): `res.summary`, `res.lesson`, `res.figure` y los números de la simulación."""
    from kikolens.analyze import concepts as cp
    res = cp.run(name, seed, **params)
    return _wrap("concept", res, table=res["table"] if res.get("table") is not None else None)


def glossary(term: str | None = None, lang: str = "en"):
    """El glosario: sin argumentos, todos los términos; con una palabra, los que la contienen; con una clave exacta (`"p_value"`), su definición y el «ojo con»."""
    from kikolens.learn import glossary as gl
    if term is None:
        return gl.glossary(lang)
    try:
        return gl.get(term, lang)
    except ValueError:
        found = gl.search(term, lang)
        if len(found):
            return found
        raise


FUNCTIONS = ("load", "query", "ask", "profile", "dictionary", "scan_privacy", "autopilot", "lab", "forecast", "anomalies", "spikes",
             "drift", "why", "regress", "survival", "cohorts", "rfm", "distribution", "link", "score", "save_model", "Result", "select_features", "segment", "ab_sequential", "granger", "forecast_drivers",
             "recommender", "synthesize", "runs", "chart", "analyst", "classify_text", "formula", "effect", "uplift", "discontinuity", "synthetic_control", "instrumental", "simulate", "optimize", "marketing_mix", "budget", "elasticity", "panel", "run", "list_plugins", "stationarity", "autocorrelation", "decompose_stl", "arima", "governance", "model_card", "topics", "near_duplicates", "similar", "text_patterns", "keyphrases", "readability", "network", "network_path", "suggest_links", "network_robustness", "geo_check", "geo_distances", "geo_hotspots", "spatial_autocorrelation", "geo_grid", "experiment", "srm", "cuped", "control_chart", "process_capability", "shift_detection", "abc_xyz", "newsvendor", "eoq", "reorder_policy", "simulate_stock_policy", "inventory_plan", "attribution", "likert", "cronbach", "nps", "crosstab", "key_drivers", "compare_items", "response_quality", "concepts", "concept", "glossary")
