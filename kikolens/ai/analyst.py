"""El analista: entiende una pregunta o un objetivo en lenguaje natural y lo resuelve con las herramientas que ya tiene KikoLens, nunca con código
inventado. El modelo solo *elige* qué herramienta usar y con qué columnas; el programa comprueba que esa elección es válida, la ejecuta y le devuelve
un resumen. Así se puede pedir «¿por qué bajaron las ventas?» y que el analista encadene varios análisis, o «muéstrame las ventas por región» y que dibuje
un gráfico, con cada paso a la vista y reproducible. Sin modelo disponible, hace un análisis automático razonable y lo dice."""
import re
from dataclasses import dataclass, field

import pandas as pd

from kikolens.ai import chartspec
from kikolens.ai.client import LLM, LLMError

MAX_STEPS = 5
MAX_HISTORY = 6


# ── Descripción de los datos para el modelo ──────────────────────────────────
def describe_data(df: pd.DataFrame, samples: bool = True, max_columns: int = 60) -> str:
    """Esquema legible de los datos. Con `samples=False` no se envía ningún valor concreto de los datos, solo nombres, tipos y rangos."""
    from kikolens.analyze.autopilot import column_roles
    from kikolens.analyze.quality import infer_types

    typed, _ = infer_types(df)
    roles = column_roles(typed)
    lines = [f"{len(df):,} rows, {df.shape[1]} columns."]
    for col in list(df.columns)[:max_columns]:
        s, role = typed[col], roles[col]
        empty = df[col].isna().mean()
        info = f"- {col}: {role}, {s.nunique(dropna=True):,} distinct" + (f", {empty:.0%} empty" if empty else "")
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.notna().any():
            info += f", range {s.min():.4g} to {s.max():.4g}"
        elif pd.api.types.is_datetime64_any_dtype(s) and s.notna().any():
            info += f", from {s.min().date()} to {s.max().date()}"
        elif samples and role == "category":
            info += ", values: " + ", ".join(str(v)[:30] for v in s.dropna().astype(str).value_counts().head(8).index)
        lines.append(info)
    if df.shape[1] > max_columns:
        lines.append(f"... and {df.shape[1] - max_columns} more columns.")
    return "\n".join(lines)


# ── Herramientas ─────────────────────────────────────────────────────────────
@dataclass
class Tool:
    name: str
    description: str
    params: dict                 # nombre -> "column" | "columns" | "number" | "text:op1|op2"   (un ? al final: opcional)
    run: object = None


def _api():
    import importlib
    return importlib.import_module("kikolens.api")


def _tools() -> dict:
    a = _api()
    tools = {t.name: t for t in [
        Tool("profile", "Data quality: missing values, duplicates, odd types.", {}, lambda df: a.profile(df)),
        Tool("autopilot", "A first full analysis: clean, find patterns, model a column and forecast dates.", {"target": "column?"},
             lambda df, target=None: a.autopilot(df, target)),
        Tool("regress", "What drives a column (linear or logistic regression) with effects in plain language.", {"target": "column", "features": "columns?"},
             lambda df, target, features=None: a.regress(df, target, features)),
        Tool("select_features", "Which columns are worth keeping to predict a target.", {"target": "column"}, lambda df, target: a.select_features(df, target)),
        Tool("forecast", "Forecast a numeric column over time.", {"date": "column", "value": "column", "horizon": "number?"},
             lambda df, date, value, horizon=12: a.forecast(df, date, value, horizon=int(horizon))),
        Tool("why", "Why a total or average changed between two periods: which values explain it.", {"metric": "column", "date": "column", "agg": "text:sum|mean|count?"},
             lambda df, metric, date, agg="sum": a.why(df, metric if agg != "count" else None, date, agg=agg)),
        Tool("spikes", "Unusual periods in a time series.", {"date": "column", "value": "column"}, lambda df, date, value: a.spikes(df, date, value)),
        Tool("anomalies", "Unusual rows and the reason for each.", {}, lambda df: a.anomalies(df)),
        Tool("segment", "Find groups of similar rows.", {"method": "text:gmm|hierarchical|dbscan?", "columns": "columns?"},
             lambda df, method="gmm", columns=None: a.segment(df, method, columns)),
        Tool("distribution", "Which distribution a numeric column follows.", {"column": "column"}, lambda df, column: a.distribution(df[column])),
        Tool("survival", "Time until an event, with a comparison between groups.", {"duration": "column", "event": "column", "group": "column?"},
             lambda df, duration, event, group=None: a.survival(df, duration, event, group)),
        Tool("granger", "Does the past of one column help predict another over time?", {"date": "column", "cause": "column", "effect": "column"},
             lambda df, date, cause, effect: a.granger(df, date, cause, effect)),
        Tool("rfm", "Customer segments by recency, frequency and spend.", {"customer": "column", "date": "column", "amount": "column"},
             lambda df, customer, date, amount: a.rfm(df, customer, date, amount)),
    ]}
    from kikolens import plugins
    for found in plugins.analyses().values():                       # los análisis de plugins que se pueden describir sin texto libre
        params = found.tool_params()
        if params is not None:
            tools[found.name] = Tool(found.name, found.description, params, lambda df, _name=found.name, **kw: plugins.run(_name, df, **kw))
    return tools


def tool_catalog() -> str:
    lines = []
    for t in _tools().values():
        args = ", ".join(f"{k}: {v}" for k, v in t.params.items()) or "no arguments"
        lines.append(f"- {t.name}({args}): {t.description}")
    return "\n".join(lines)


def validate_call(df: pd.DataFrame, name: str, args) -> dict:
    """Comprueba la elección del modelo contra la herramienta y los datos. Devuelve los argumentos limpios o lanza ValueError con la razón."""
    tools = _tools()
    if name not in tools:
        raise ValueError(f"Unknown tool '{name}'. Tools: {', '.join(tools)}.")
    if not isinstance(args, dict):
        raise ValueError("'args' must be an object.")
    spec, clean = tools[name].params, {}
    unknown = set(args) - set(spec)
    if unknown:
        raise ValueError(f"{name} does not take {sorted(unknown)}. It takes: {', '.join(spec) or 'nothing'}.")
    for key, kind in spec.items():
        optional = kind.endswith("?")
        kind = kind.rstrip("?")
        value = args.get(key)
        if value in (None, "", []):
            if optional:
                continue
            raise ValueError(f"{name} needs '{key}'.")
        if kind == "column":
            clean[key] = _existing(df, value)
        elif kind == "columns":
            if not isinstance(value, list):
                raise ValueError(f"'{key}' must be a list of columns.")
            clean[key] = [_existing(df, v) for v in value]
        elif kind == "number":
            try:
                clean[key] = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"'{key}' must be a number.") from None
        else:
            options = kind.split(":", 1)[1].split("|")
            if str(value) not in options:
                raise ValueError(f"'{key}' must be one of {', '.join(options)}.")
            clean[key] = str(value)
    return clean


def _existing(df: pd.DataFrame, value) -> str:
    if value in df.columns:
        return value
    match = [c for c in df.columns if str(c).lower() == str(value).lower()]
    if match:
        return match[0]
    raise ValueError(f"Column '{value}' does not exist. Columns: {', '.join(map(str, df.columns))}.")


# ── Bucle del analista ───────────────────────────────────────────────────────
@dataclass
class Step:
    tool: str
    args: dict
    thought: str = ""
    summary: str = ""
    figure: object = None
    table: object = None
    error: str = ""


@dataclass
class Report:
    goal: str
    steps: list = field(default_factory=list)
    answer: str = ""
    used_model: bool = True
    notes: list = field(default_factory=list)

    def markdown(self) -> str:
        out = [f"# {self.goal}", "", self.answer, ""]
        for i, s in enumerate(self.steps, 1):
            out += [f"## Step {i}: {s.tool}", f"*{s.thought}*" if s.thought else "", f"Arguments: `{s.args}`", "", s.summary or s.error, ""]
        out += [f"_{n}_" for n in self.notes]
        return "\n".join(out)

    def python(self, source: str = "data.csv") -> str:
        """Las mismas llamadas como código de la API de Python: lo que el analista hizo se puede repetir sin el modelo."""
        lines = ["import kikolens as kl", "", f"df = kl.load({source!r})"]
        for s in self.steps:
            if s.error:
                continue
            args = ", ".join(f"{k}={v!r}" for k, v in s.args.items())
            if hasattr(_api(), s.tool):
                lines.append(f"kl.{s.tool}(df{', ' + args if args else ''})")
            else:
                lines.append(f"kl.run({s.tool!r}, df{', ' + args if args else ''})")          # análisis de un plugin
        return "\n".join(lines) + "\n"


def _agent_prompt(df, goal, steps, samples) -> str:
    done = "\n".join(f"{i}. {s.tool}({s.args}) -> {s.summary or 'ERROR: ' + s.error}" for i, s in enumerate(steps, 1)) or "(nothing yet)"
    return f"""You are a careful data analyst. Reach the user's goal by calling analysis tools, one at a time, then explain the result.

DATA
{describe_data(df, samples)}

TOOLS
{tool_catalog()}

GOAL
{goal}

DONE SO FAR
{done}

Answer with ONLY a JSON object, one of:
{{"thought": "why this step", "tool": "<tool name>", "args": {{...}}}}   to run a tool (use column names exactly as written above)
{{"thought": "why I can stop", "tool": "finish", "answer": "the answer for the user in plain language, citing the numbers found"}}   when you have enough.
Do not repeat a step that already ran. Use at most {MAX_STEPS} steps."""


def run_agent(df: pd.DataFrame, goal: str, llm: LLM, max_steps: int = MAX_STEPS, samples: bool = True, progress=None) -> Report:
    """El bucle: el modelo elige una herramienta, se valida y se ejecuta, el resumen vuelve al modelo, hasta que termina o se acaban los pasos."""
    if not goal or not goal.strip():
        raise ValueError("Say what you want to find out.")
    report, seen = Report(goal=goal.strip()), set()
    tools = _tools()
    for _ in range(max(1, min(max_steps, 10))):
        def check(obj):
            if not isinstance(obj, dict) or "tool" not in obj:
                raise ValueError('The answer must be an object with a "tool" key.')
            if obj["tool"] == "finish":
                return obj
            return {**obj, "args": validate_call(df, obj["tool"], obj.get("args", {}))}
        decision = llm.complete_json(_agent_prompt(df, goal, report.steps, samples), validate=check)
        if decision["tool"] == "finish":
            report.answer = str(decision.get("answer", "")).strip()
            break
        key = (decision["tool"], repr(sorted(decision["args"].items())))
        if key in seen:
            report.notes.append("The model asked to repeat a step, so the analysis stopped there.")
            break
        seen.add(key)
        step = Step(decision["tool"], decision["args"], str(decision.get("thought", ""))[:300])
        if progress:
            progress(len(report.steps) / max_steps, f"{step.tool}")
        try:
            result = tools[step.tool].run(df, **step.args)
            step.summary = str(result.get("summary") or result.get("narrative") or "").replace("\n", " ")[:700]
            step.figure, step.table = getattr(result, "main_figure", None), getattr(result, "main_table", None)
        except Exception as e:  # una herramienta que falla se anota y el modelo sigue con lo demás
            step.error = str(e)[:300]
        report.steps.append(step)
    if not report.answer:
        report.answer = _final_answer(llm, goal, report)
    return report


def _final_answer(llm: LLM, goal: str, report: Report) -> str:
    facts = "\n".join(f"- {s.tool}: {s.summary}" for s in report.steps if s.summary)
    if not facts:
        return "No analysis produced a result, so there is nothing to conclude yet."
    try:
        return llm.complete(f"Goal: {goal}\n\nFindings:\n{facts}\n\nWrite a short answer for a non-expert, citing the numbers. No preamble.").strip()
    except LLMError:
        return facts


# ── Sin modelo ───────────────────────────────────────────────────────────────
def plan_offline(df: pd.DataFrame, goal: str) -> Report:
    """Cuando no hay modelo de lenguaje: se elige por palabras clave entre unas pocas herramientas y se dice claramente."""
    from kikolens.analyze.autopilot import column_roles, suggest_target
    from kikolens.analyze.quality import infer_types

    typed, _ = infer_types(df)
    roles = column_roles(typed)
    dates = [c for c, r in roles.items() if r == "date"]
    numbers = [c for c, r in roles.items() if r == "number"]
    target = suggest_target(typed, roles)
    text = goal.lower()
    tools, steps = _tools(), []
    if re.search(r"forecast|predict(ion)? (the )?future|next (month|week|year)|projection", text) and dates and numbers:
        steps.append(("forecast", {"date": dates[0], "value": target if target in numbers else numbers[-1]}))
    elif re.search(r"\bwhy\b|changed|drop|fell|rose|increase|decrease", text) and dates and numbers:
        steps.append(("why", {"metric": target if target in numbers else numbers[-1], "date": dates[0]}))
    elif re.search(r"outlier|anomal|unusual|strange", text):
        steps.append(("anomalies", {}))
    elif re.search(r"quality|clean|missing|duplicate", text):
        steps.append(("profile", {}))
    else:
        steps.append(("autopilot", {}))
    report = Report(goal=goal.strip(), used_model=False,
                    notes=["The local language model is not available, so KikoLens picked an analysis from your words. Start Ollama for a real conversation."])
    for name, args in steps:
        step = Step(name, args, "Chosen from keywords in the question.")
        try:
            result = tools[name].run(df, **args)
            step.summary = str(result.get("summary") or result.get("narrative") or "").replace("\n", " ")[:700]
            step.figure, step.table = getattr(result, "main_figure", None), getattr(result, "main_table", None)
        except Exception as e:
            step.error = str(e)[:300]
        report.steps.append(step)
    report.answer = " ".join(s.summary for s in report.steps if s.summary) or "The analysis could not run on these data."
    return report


# ── Conversación ─────────────────────────────────────────────────────────────
@dataclass
class Turn:
    question: str
    action: str
    text: str = ""
    figure: object = None
    table: object = None
    spec: dict | None = None
    sql: str = ""
    report: Report | None = None
    error: str = ""


class Analyst:
    """Conversación con los datos. `ask` decide si la pregunta pide un gráfico, una consulta SQL, un análisis o solo una respuesta."""

    def __init__(self, df: pd.DataFrame, llm: LLM | None = None, samples: bool = True):
        self.df, self.llm, self.samples, self.history = df, llm or LLM(), samples, []

    def _context(self) -> str:
        recent = self.history[-MAX_HISTORY:]
        return "\n".join(f"User: {t.question}\nAssistant ({t.action}): {t.text[:300]}" for t in recent) or "(first question)"

    def _route(self, question: str) -> dict:
        prompt = f"""You help a person explore a table. Decide how to answer their message.

DATA
{describe_data(self.df, self.samples)}

CONVERSATION SO FAR
{self._context()}

MESSAGE
{question}

Answer with ONLY a JSON object, one of:
{{"action": "chart", "spec": {{"kind": "bar|line|area|scatter|histogram|box|pie|heatmap", "x": "<column>", "y": "<column or null>", "color": "<column or null>", "agg": "sum|mean|median|min|max|count|nunique", "grain": "day|week|month|quarter|year or null", "filters": [{{"column": "<column>", "op": "==|!=|>|>=|<|<=|in|contains|between", "value": "<value>"}}], "top": <number or null>, "sort": "asc|desc|none", "title": "<title>"}}}}   to draw a chart
{{"action": "sql", "sql": "<one read-only DuckDB SELECT on the table df>"}}   for a number, a list or a table
{{"action": "analysis", "goal": "<what to find out>"}}   for why/what drives/forecast/segments/anything that needs statistics
{{"action": "answer", "text": "<short answer>"}}   if no data work is needed (greetings, explaining a previous result)"""

        def check(obj):
            if not isinstance(obj, dict) or obj.get("action") not in ("chart", "sql", "analysis", "answer"):
                raise ValueError('"action" must be chart, sql, analysis or answer.')
            if obj["action"] == "chart":
                chartspec.validate_spec(self.df, obj.get("spec"))
            if obj["action"] == "sql":
                from kikolens.sql.runner import validate_read_only
                validate_read_only(str(obj.get("sql", "")))
            return obj
        return self.llm.complete_json(prompt, validate=check)

    def ask(self, question: str) -> Turn:
        question = (question or "").strip()
        if not question:
            raise ValueError("Write a question.")
        try:
            decision = self._route(question)
        except LLMError as e:
            return self._remember(self._offline(question, str(e)))
        action = decision["action"]
        turn = Turn(question, action)
        try:
            if action == "chart":
                fig, table, spec = chartspec.render(self.df, decision["spec"])
                turn.figure, turn.table, turn.spec = fig, table, spec
                turn.text = self._explain(fig, question)
            elif action == "sql":
                from kikolens.sql.runner import run_sql_query
                turn.sql = decision["sql"]
                turn.table = run_sql_query(self.df, turn.sql)
                turn.text = f"{len(turn.table):,} row(s)."
            elif action == "analysis":
                turn.report = run_agent(self.df, decision.get("goal") or question, self.llm, samples=self.samples)
                turn.text = turn.report.answer
                charted = [s for s in turn.report.steps if s.figure is not None]
                turn.figure = charted[-1].figure if charted else None
            else:
                turn.text = str(decision.get("text", "")).strip() or "I have nothing to add."
        except (ValueError, LLMError) as e:
            turn.error = str(e)
            turn.text = f"I could not do that: {e}"
        return self._remember(turn)

    def _explain(self, fig, question: str) -> str:
        try:
            return self.llm.complete(f"Question: {question}\nWhat the chart shows:\n{chartspec.numeric_summary(fig)}\n\nIn two sentences, say what it shows for a non-expert, "
                                     "citing the numbers.").strip()
        except LLMError:
            return chartspec.numeric_summary(fig)

    def _offline(self, question: str, reason: str) -> Turn:
        report = plan_offline(self.df, question)
        charted = [s for s in report.steps if s.figure is not None]
        return Turn(question, "analysis", report.answer, figure=charted[-1].figure if charted else None, report=report, error=reason)

    def _remember(self, turn: Turn) -> Turn:
        self.history.append(turn)
        return turn


# ── Utilidades sueltas con el modelo ─────────────────────────────────────────
def ask_chart(df: pd.DataFrame, question: str, llm: LLM | None = None, samples: bool = True) -> tuple:
    """Solo el gráfico: (figura, tabla, especificación). El modelo describe lo que se pide y `chartspec` lo valida y lo dibuja."""
    llm = llm or LLM()
    prompt = f"""Turn the request into a chart specification for this table.

DATA
{describe_data(df, samples)}

REQUEST
{question}

Answer with ONLY a JSON object: {{"kind": "bar|line|area|scatter|histogram|box|pie|heatmap", "x": "<column>", "y": "<column or null>", "color": "<column or null>",
"agg": "sum|mean|median|min|max|count|nunique", "grain": "day|week|month|quarter|year or null", "filters": [{{"column": "<column>", "op": "==|!=|>|>=|<|<=|in|contains|between",
"value": "<value>"}}], "top": <number or null>, "sort": "asc|desc|none", "title": "<title>"}}. For a heatmap also give "value" (a numeric column) or agg "count"."""
    spec = llm.complete_json(prompt, validate=lambda obj: chartspec.validate_spec(df, obj) and obj)
    return chartspec.render(df, spec)


def classify_texts(texts: pd.Series, labels: list, llm: LLM | None = None, batch: int = 15, limit: int = 300) -> pd.DataFrame:
    """Clasifica textos en las etiquetas dadas sin entrenar nada («zero-shot»), de a `batch` por pedido. Una respuesta fuera de las etiquetas se marca «other»."""
    llm = llm or LLM()
    labels = [str(label).strip() for label in labels if str(label).strip()]
    if len(labels) < 2:
        raise ValueError("Give at least two labels.")
    values = texts.dropna().astype(str).str.slice(0, 400)
    if values.empty:
        raise ValueError("There are no texts to classify.")
    values = values.head(limit)
    out = {}
    for start in range(0, len(values), batch):
        chunk = values.iloc[start:start + batch]
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(chunk))
        prompt = (f"Classify each text into exactly one of these labels: {', '.join(labels)}. Use 'other' if none fits.\n\nTEXTS\n{numbered}\n\n"
                  'Answer with ONLY a JSON object mapping the number to the label, like {"0": "label", "1": "label"}.')

        def check(obj, size=len(chunk)):
            if not isinstance(obj, dict):
                raise ValueError("The answer must be a JSON object.")
            if len(obj) < size:
                raise ValueError(f"Expected {size} labels, got {len(obj)}.")
            return obj
        answer = llm.complete_json(prompt, validate=check)
        for i, index in enumerate(chunk.index):
            raw = str(answer.get(str(i), "other")).strip()
            out[index] = next((lab for lab in labels if lab.lower() == raw.lower()), "other")
    frame = pd.DataFrame({"text": values, "label": pd.Series(out)})
    return frame


def suggest_formula(df: pd.DataFrame, instruction: str, llm: LLM | None = None) -> dict:
    """Una columna nueva descrita con palabras: el modelo propone una fórmula y se comprueba con el intérprete seguro; solo se acepta si se puede calcular."""
    from kikolens.analyze.transform import evaluate_expression
    llm = llm or LLM()
    prompt = f"""Write a formula for a new column of this table.

DATA
{describe_data(df, True)}

WHAT THE USER WANTS
{instruction}

Allowed: column names (put names with spaces between `backticks`), numbers, + - * / // % **, comparisons (> < >= <= == !=, which give 1 or 0), and the functions
log, log1p, log10, sqrt, abs, exp, round, floor, ceil, sin, cos, min, max. Nothing else (no if/else, no text functions).
Answer with ONLY a JSON object: {{"name": "<new column name>", "expression": "<formula>", "explanation": "<one sentence>"}}"""

    def check(obj):
        if not isinstance(obj, dict) or not obj.get("expression") or not obj.get("name"):
            raise ValueError('The answer needs "name" and "expression".')
        values = evaluate_expression(df, str(obj["expression"]))
        if values.notna().sum() == 0:
            raise ValueError("The formula gives only empty values.")
        return {**obj, "preview": values.head(8).tolist()}
    return llm.complete_json(prompt, validate=check)
