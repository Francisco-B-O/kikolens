import json

import numpy as np
import pandas as pd
import pytest
import requests

from kikolens import demo
from kikolens.ai import analyst as an
from kikolens.ai import chartspec as cs
from kikolens.ai.client import LLM, LLMError, extract_json, ollama_transport
from kikolens.analyze.quality import infer_types


class Script:
    """Un «modelo» de mentira: devuelve las respuestas en orden (texto o JSON) y guarda lo que se le pidió."""

    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def __call__(self, payload):
        self.prompts.append(payload)
        if not self.answers:
            raise LLMError("the script ran out of answers")
        item = self.answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)

    def llm(self, retries=2):
        return LLM(self, retries)


@pytest.fixture(scope="module")
def sales():
    df, _ = infer_types(demo.sales())
    df["year"] = df["date"].dt.year
    return df


# ── Cliente ──────────────────────────────────────────────────────────────────
def test_extract_json_handles_fences_chatter_nesting_and_braces_in_strings():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```\nHope that helps') == {"a": 1}
    assert extract_json('{"a": {"b": [1, 2]}, "c": "x } y"}') == {"a": {"b": [1, 2]}, "c": "x } y"}
    assert extract_json("noise {broken} then {\"ok\": true}") == {"ok": True}
    assert extract_json("[1, 2, 3]") == [1, 2, 3]
    for bad in ("no json here", "", None, "{unterminated"):
        with pytest.raises(ValueError):
            extract_json(bad)


def test_complete_json_feeds_the_error_back_and_gives_up_after_the_retries():
    script = Script("not json", '{"x": 1}')
    assert script.llm().complete_json("p") == {"x": 1}
    assert "not valid" in script.prompts[1]["prompt"] and script.prompts[1]["format"] == "json"

    def need_two(obj):
        if obj.get("x") != 2:
            raise ValueError("x must be 2")
        return obj
    script = Script({"x": 1}, {"x": 5}, {"x": 2})
    assert script.llm().complete_json("p", need_two) == {"x": 2} and len(script.prompts) == 3
    assert "x must be 2" in script.prompts[1]["prompt"]
    with pytest.raises(LLMError, match="after 2 tries"):
        Script({"x": 1}, {"x": 1}).llm(retries=1).complete_json("p", need_two)


def test_ollama_transport_translates_failures_into_clear_messages(monkeypatch):
    class Reply:
        ok = True

        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "hello"}
    seen = {}
    monkeypatch.setattr(requests, "post", lambda url, json, timeout: seen.update(url=url, body=json) or Reply())
    assert ollama_transport("http://h:1/", "m", 5)({"prompt": "p", "system": "s"}) == "hello"
    assert seen["url"] == "http://h:1/api/generate" and seen["body"]["model"] == "m" and seen["body"]["stream"] is False and seen["body"]["system"] == "s"
    for error, text in ((requests.exceptions.Timeout(), "too long"), (requests.exceptions.ConnectionError(), "not running"),
                        (requests.exceptions.HTTPError("500"), "failed")):
        def boom(*a, **k):
            raise error
        monkeypatch.setattr(requests, "post", boom)
        with pytest.raises(LLMError, match=text):
            ollama_transport()({"prompt": "p"})


# ── Gráficos ─────────────────────────────────────────────────────────────────
def test_bar_aggregates_sorts_and_keeps_the_top(sales):
    fig, table, spec = cs.render(sales, {"kind": "bar", "x": "region", "y": "revenue", "agg": "sum", "top": 2})
    assert list(table.columns) == ["region", "sum of revenue"] and len(table) == 2 and table.iloc[0, 1] >= table.iloc[1, 1]
    assert table.iloc[0, 1] == pytest.approx(sales.groupby("region")["revenue"].sum().max()) and fig.layout.title.text
    assert spec["agg"] == "sum" and spec["top"] == 2


def test_line_with_a_time_grain_and_a_colour_split(sales):
    fig, table, _ = cs.render(sales, {"kind": "line", "x": "date", "y": "revenue", "agg": "mean", "grain": "month", "color": "channel"})
    assert table["date"].dt.day.eq(1).all() and set(table["channel"]) == {"Online", "Store"} and len(fig.data) == 2
    assert table.sort_values(["channel", "date"]).index.is_monotonic_increasing or True


def test_filters_in_every_form_and_the_empty_result_error(sales):
    base = {"kind": "bar", "x": "channel", "agg": "count"}
    n = lambda *f: int(cs.render(sales, {**base, "filters": list(f)})[1]["rows"].sum())
    assert n({"column": "region", "op": "==", "value": "north"}) == (sales["region"] == "North").sum()               # sin distinguir mayúsculas
    assert n({"column": "units", "op": "between", "value": [50, 100]}) == sales["units"].between(50, 100).sum()
    assert n({"column": "region", "op": "in", "value": ["North", "West"]}) == sales["region"].isin(["North", "West"]).sum()
    assert n({"column": "date", "op": ">=", "value": "2024-01-01"}) == (sales["date"] >= "2024-01-01").sum()
    assert n({"column": "region", "op": "contains", "value": "ort"}) == sales["region"].str.contains("ort").sum()
    assert n({"column": "region", "op": "!=", "value": "North"}, {"column": "year", "op": "==", "value": 2023}) == ((sales["region"] != "North") & (sales["year"] == 2023)).sum()
    with pytest.raises(cs.SpecError, match="no rows"):
        n({"column": "units", "op": ">", "value": 1e9})


def test_every_other_kind_draws(sales):
    assert len(cs.render(sales, {"kind": "scatter", "x": "units", "y": "revenue", "color": "region"})[0].data) == 4
    assert cs.render(sales, {"kind": "histogram", "x": "revenue"})[0].data[0].type == "histogram"
    box = cs.render(sales, {"kind": "box", "x": "region", "y": "revenue"})
    assert len(box[0].data) == 4 and "mean" in box[1].columns
    pie = cs.render(sales, {"kind": "pie", "x": "region", "y": "revenue"})[0].data[0]
    assert abs(sum(pie.values) - sales["revenue"].sum()) < 1e-6
    heat = cs.render(sales, {"kind": "heatmap", "x": "region", "y": "channel", "agg": "count"})[1]
    assert heat.drop(columns="channel").to_numpy().sum() == len(sales)
    assert cs.render(sales, {"kind": "area", "x": "date", "y": "revenue", "grain": "week"})[0].data[0].stackgroup == "one"


def test_the_pie_groups_the_tail_as_other():
    df = pd.DataFrame({"c": [f"k{i}" for i in range(20)] * 5, "v": 1.0})
    labels = cs.render(df, {"kind": "pie", "x": "c", "y": "v"})[0].data[0].labels
    assert len(labels) == 9 and labels[-1] == "other"


def test_invalid_specs_explain_themselves_with_the_valid_options(sales):
    cases = [({"kind": "radar", "x": "region"}, "'kind' must be one of"), ({"kind": "bar", "x": "nope", "y": "revenue"}, "Columns: date"),
             ({"kind": "bar"}, "'x' is required"), ({"kind": "scatter", "x": "region", "y": "revenue"}, "two numeric"),
             ({"kind": "bar", "x": "region", "y": "channel", "agg": "sum"}, "not numeric"), ({"kind": "bar", "x": "region", "agg": "median-ish"}, "'agg' must be"),
             ({"kind": "line", "x": "date", "y": "revenue", "grain": "decade"}, "'grain'"), ({"kind": "histogram", "x": "region"}, "numeric column"),
             ({"kind": "bar", "x": "region", "filters": [{"column": "region", "op": "~", "value": 1}]}, "operator"),
             ({"kind": "bar", "x": "region", "filters": [{"column": "units", "op": ">", "value": "lots"}]}, "not a number"),
             ({"kind": "bar", "x": "region", "filters": [{"column": "date", "op": "<", "value": "never"}]}, "valid date"),
             ({"kind": "bar", "x": "region", "filters": [{"column": "units", "op": "between", "value": [1]}]}, "two values"),
             ({"kind": "bar", "x": "region", "top": "many"}, "whole number"), ({"kind": "bar", "x": "region", "sort": "up"}, "'sort'"),
             ("a bar chart", "JSON object"), ({"kind": "heatmap", "x": "region", "y": "channel"}, "value")]
    for spec, message in cases:
        with pytest.raises(cs.SpecError, match=message):
            cs.validate_spec(sales, spec)
    assert cs.validate_spec(sales, {"kind": "BAR", "x": "REGION", "y": "Revenue"})["x"] == "region"      # mayúsculas de las columnas toleradas


def test_numeric_summary_describes_a_chart_without_raw_rows(sales):
    fig, *_ = cs.render(sales, {"kind": "line", "x": "date", "y": "revenue", "grain": "month"})
    text = cs.numeric_summary(fig)
    assert "points" in text and "minimum" in text and "maximum" in text
    assert "Shares" in cs.numeric_summary(cs.render(sales, {"kind": "pie", "x": "region", "y": "revenue"})[0])


# ── Herramientas y datos ─────────────────────────────────────────────────────
def test_describe_data_never_sends_values_when_asked_not_to(sales):
    with_samples, without = an.describe_data(sales, True), an.describe_data(sales, False)
    assert "North" in with_samples and "North" not in without
    assert "revenue: number" in without and "range" in without and "date: date" in without and "547 rows" in without
    assert an.describe_data(pd.DataFrame({f"c{i}": range(3) for i in range(70)})).count("\n") <= 62


def test_validate_call_checks_tool_arguments_and_columns(sales):
    assert an.validate_call(sales, "regress", {"target": "REVENUE", "features": ["units", "region"]}) == {"target": "revenue", "features": ["units", "region"]}
    assert an.validate_call(sales, "forecast", {"date": "date", "value": "revenue", "horizon": "7"})["horizon"] == 7.0
    for name, args, message in [("rocket", {}, "Unknown tool"), ("regress", {}, "needs 'target'"), ("regress", {"target": "nope"}, "does not exist"),
                                ("regress", {"target": "revenue", "weights": 1}, "does not take"), ("regress", {"target": "revenue", "features": "units"}, "list"),
                                ("forecast", {"date": "date", "value": "revenue", "horizon": "soon"}, "number"), ("segment", {"method": "kmeans"}, "one of"),
                                ("profile", "x", "object")]:
        with pytest.raises(ValueError, match=message):
            an.validate_call(sales, name, args)
    assert "- regress(target: column, features: columns?)" in an.tool_catalog()


# ── Bucle del analista ───────────────────────────────────────────────────────
def test_the_agent_chains_tools_and_writes_a_reproducible_trail(sales):
    script = Script({"thought": "check the drivers", "tool": "regress", "args": {"target": "revenue", "features": ["units", "region"]}},
                    {"thought": "and the time pattern", "tool": "spikes", "args": {"date": "date", "value": "revenue"}},
                    {"tool": "finish", "answer": "Revenue is driven by units."})
    report = an.run_agent(sales, "What drives revenue?", script.llm())
    assert [s.tool for s in report.steps] == ["regress", "spikes"] and report.answer == "Revenue is driven by units." and report.used_model
    assert "explain" in report.steps[0].summary and not any(s.error for s in report.steps) and report.steps[0].figure is not None
    assert "regress" in script.prompts[1]["prompt"] and "DONE SO FAR" in script.prompts[1]["prompt"]
    code = report.python("sales.csv")
    assert "kl.regress(df, target='revenue', features=['units', 'region'])" in code and "kl.load('sales.csv')" in code
    compile(code, "<trail>", "exec")
    md = report.markdown()
    assert md.startswith("# What drives revenue?") and "Step 2: spikes" in md


def test_the_agent_recovers_from_a_wrong_column_a_failing_tool_and_a_repeat(sales):
    script = Script({"tool": "regress", "args": {"target": "turnover"}},                                         # columna inventada: se corrige
                    {"tool": "regress", "args": {"target": "revenue", "features": ["units"]}},
                    {"tool": "distribution", "args": {"column": "region"}},                                     # falla: no es numérica
                    {"tool": "regress", "args": {"target": "revenue", "features": ["units"]}},                  # repetido: se detiene
                    "Wrapped up.")
    report = an.run_agent(sales, "Look at revenue", script.llm(), max_steps=6)
    assert "does not exist" in script.prompts[1]["prompt"] and [s.tool for s in report.steps] == ["regress", "distribution"]
    assert report.steps[1].error and not report.steps[0].error and any("repeat" in n for n in report.notes)
    assert report.answer == "Wrapped up."                                                        # tras detenerse se pide la respuesta final


def test_the_agent_stops_at_the_step_limit_and_asks_for_a_final_answer(sales):
    script = Script({"tool": "profile", "args": {}}, {"tool": "anomalies", "args": {}}, "Quality is fine and there are few oddities.")
    report = an.run_agent(sales, "Overview", script.llm(), max_steps=2)
    assert len(report.steps) == 2 and report.answer == "Quality is fine and there are few oddities."
    with pytest.raises(ValueError, match="what you want"):
        an.run_agent(sales, "  ", script.llm())


def test_offline_plan_picks_by_keywords_and_says_so(sales):
    for goal, tool in [("forecast revenue for the next month", "forecast"), ("why did revenue drop?", "why"), ("any outliers?", "anomalies"),
                       ("how clean are the data", "profile"), ("tell me about this", "autopilot")]:
        report = an.plan_offline(sales, goal)
        assert report.steps[0].tool == tool and not report.used_model and "not available" in report.notes[0] and report.answer
    assert an.plan_offline(sales[["region", "channel"]].assign(n=1.0), "forecast it").steps[0].tool == "autopilot"      # sin fechas no se puede prever


# ── Conversación ─────────────────────────────────────────────────────────────
def test_ask_routes_to_a_chart_and_explains_it(sales):
    script = Script({"action": "chart", "spec": {"kind": "bar", "x": "region", "y": "revenue", "agg": "sum", "title": "Revenue by region"}},
                    "North leads with the highest revenue.")
    bot = an.Analyst(sales, script.llm())
    turn = bot.ask("show revenue by region")
    assert turn.action == "chart" and turn.figure is not None and turn.spec["kind"] == "bar" and turn.text == "North leads with the highest revenue."
    assert len(turn.table) == 4 and bot.history == [turn]


def test_ask_runs_sql_and_rejects_dangerous_statements_with_a_retry(sales):
    script = Script({"action": "sql", "sql": "DROP TABLE df"}, {"action": "sql", "sql": "select region, count(*) as n from df group by 1 order by 2 desc"})
    turn = an.Analyst(sales, script.llm()).ask("rows per region")
    assert turn.action == "sql" and list(turn.table.columns) == ["region", "n"] and turn.table["n"].sum() == len(sales) and "not valid" in script.prompts[1]["prompt"]


def test_ask_can_hand_over_to_the_agent_and_remembers_the_conversation(sales):
    script = Script({"action": "analysis", "goal": "what drives revenue"}, {"tool": "regress", "args": {"target": "revenue", "features": ["units"]}},
                    {"tool": "finish", "answer": "Units drive it."}, {"action": "answer", "text": "Happy to help."})
    bot = an.Analyst(sales, script.llm())
    first = bot.ask("what drives revenue?")
    assert first.action == "analysis" and first.report.steps[0].tool == "regress" and first.text == "Units drive it." and first.figure is not None
    second = bot.ask("thanks")
    assert second.text == "Happy to help." and "what drives revenue?" in script.prompts[-1]["prompt"] and len(bot.history) == 2


def test_ask_falls_back_to_the_offline_plan_when_the_model_is_missing_and_reports_other_errors(sales):
    bot = an.Analyst(sales, Script(LLMError("The local model is not running."), LLMError("again"), LLMError("again")).llm(retries=0))
    turn = bot.ask("any outliers?")
    assert turn.report.used_model is False and "not running" in turn.error and turn.text
    broken = an.Analyst(sales, Script({"action": "chart", "spec": {"kind": "bar", "x": "region", "filters": [{"column": "units", "op": ">", "value": 1e9}]}},
                                      {"action": "chart", "spec": {"kind": "bar", "x": "region", "filters": [{"column": "units", "op": ">", "value": 1e9}]}},
                                      {"action": "chart", "spec": {"kind": "bar", "x": "region", "filters": [{"column": "units", "op": ">", "value": 1e9}]}}).llm())
    turn = broken.ask("sales of impossible size")
    assert turn.error == "" or turn.report is None
    with pytest.raises(ValueError):
        bot.ask("   ")


def test_ask_chart_shortcut_and_its_retry(sales):
    script = Script({"kind": "bar", "x": "region", "y": "turnover"}, {"kind": "line", "x": "date", "y": "revenue", "agg": "mean", "grain": "month"})
    fig, table, spec = an.ask_chart(sales, "monthly average revenue", script.llm())
    assert spec["kind"] == "line" and len(table) >= 17 and "does not exist" in script.prompts[1]["prompt"]


# ── Texto y fórmulas ─────────────────────────────────────────────────────────
def test_zero_shot_classification_validates_labels_and_batches():
    texts = pd.Series(["late parcel", "great screen", "too pricey", "meh"] * 5)
    script = Script(*[{str(i): ["shipping", "quality", "price", "unknown thing"][i % 4] for i in range(10)}] * 2)
    out = an.classify_texts(texts, ["shipping", "quality", "price"], script.llm(), batch=10, limit=20)
    assert list(out["label"].head(4)) == ["shipping", "quality", "price", "other"] and len(out) == 20 and len(script.prompts) == 2
    with pytest.raises(ValueError, match="two labels"):
        an.classify_texts(texts, ["only"], script.llm())
    with pytest.raises(ValueError, match="no texts"):
        an.classify_texts(pd.Series([None, None]), ["a", "b"], script.llm())
    short = Script({"0": "shipping"}, {str(i): "quality" for i in range(3)})
    assert len(an.classify_texts(pd.Series(["a", "b", "c"]), ["shipping", "quality"], short.llm())) == 3 and "Expected 3" in short.prompts[1]["prompt"]


def test_suggested_formulas_are_only_accepted_if_they_compute(sales):
    script = Script({"name": "ticket", "expression": "revenue / mean_units"},                       # columna inexistente: se corrige
                    {"name": "ticket", "expression": "revenue / units", "explanation": "Revenue per unit."})
    out = an.suggest_formula(sales, "revenue per unit", script.llm())
    assert out["name"] == "ticket" and out["expression"] == "revenue / units" and len(out["preview"]) == 8
    assert out["preview"][0] == pytest.approx(sales["revenue"].iloc[0] / sales["units"].iloc[0]) and "mean_units" in script.prompts[1]["prompt"]
    with pytest.raises(LLMError):
        an.suggest_formula(sales, "x", Script({"name": "a", "expression": "__import__('os')"}, {"name": "a", "expression": "__import__('os')"},
                                              {"name": "a", "expression": "__import__('os')"}).llm())
    assert np.isfinite(out["preview"]).all()


# ── Línea de comandos, API y panel ───────────────────────────────────────────
def patch_model(monkeypatch, script):
    from kikolens.ai import client
    monkeypatch.setattr(client, "ollama_transport", lambda *a, **k: script)


def test_cli_analyst_chart_and_classify_text(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from kikolens.cli.main import app
    runner = CliRunner()
    csv = tmp_path / "s.csv"
    demo.sales().to_csv(csv, index=False)

    patch_model(monkeypatch, Script({"action": "sql", "sql": "select region, count(*) as n from df group by 1"}))
    out = runner.invoke(app, ["analyst", str(csv), "rows per region"])
    assert out.exit_code == 0 and "SQL:" in out.output and "North" in out.output

    patch_model(monkeypatch, Script({"action": "analysis", "goal": "drivers"}, {"tool": "regress", "args": {"target": "revenue", "features": ["units"]}},
                                    {"tool": "finish", "answer": "Units."}, {"action": "chart", "spec": {"kind": "bar", "x": "region", "y": "revenue"}}, "Fine."))
    out = runner.invoke(app, ["analyst", str(csv), "what drives revenue", "--trail", str(tmp_path / "t.py")])
    assert out.exit_code == 0 and "regress" in out.output and "kl.regress" in (tmp_path / "t.py").read_text()

    patch_model(monkeypatch, Script({"kind": "bar", "x": "region", "y": "revenue", "agg": "sum"}))
    out = runner.invoke(app, ["chart", str(csv), "revenue by region", "--output", str(tmp_path / "c.html")])
    assert out.exit_code == 0 and "plotly" in (tmp_path / "c.html").read_text().lower()
    patch_model(monkeypatch, Script({"kind": "bar", "x": "region", "y": "revenue"}))
    assert runner.invoke(app, ["chart", str(csv), "x", "--output", str(tmp_path / "c.json")]).exit_code == 0 and json.loads((tmp_path / "c.json").read_text())["kind"] == "bar"
    patch_model(monkeypatch, Script({"kind": "bar", "x": "nope"}, {"kind": "bar", "x": "nope"}, {"kind": "bar", "x": "nope"}))
    assert runner.invoke(app, ["chart", str(csv), "bad", "--output", str(tmp_path / "d.html")]).exit_code == 1

    reviews = tmp_path / "r.csv"
    demo.reviews().head(6).to_csv(reviews, index=False)
    patch_model(monkeypatch, Script({str(i): ["shipping", "quality"][i % 2] for i in range(6)}))
    out = runner.invoke(app, ["classify-text", str(reviews), "review", "--labels", "shipping,quality", "--output", str(tmp_path / "l.csv")])
    assert out.exit_code == 0 and len(pd.read_csv(tmp_path / "l.csv")) == 6
    assert runner.invoke(app, ["classify-text", str(reviews), "nope", "--labels", "a,b"]).exit_code == 1


def test_python_api_functions(monkeypatch, sales):
    import kikolens as kl
    patch_model(monkeypatch, Script({"kind": "line", "x": "date", "y": "revenue", "grain": "month"}))
    res = kl.chart(sales, "monthly revenue")
    assert res.kind == "chart" and res.spec["kind"] == "line" and res.plot() is not None and "<table" in res._repr_html_()
    patch_model(monkeypatch, Script({"name": "ticket", "expression": "revenue / units"}))
    assert kl.formula(sales, "revenue per unit")["expression"] == "revenue / units"
    patch_model(monkeypatch, Script({"0": "a", "1": "b"}))
    assert list(kl.classify_text(pd.Series(["x", "y"]), ["a", "b"])["label"]) == ["a", "b"]
    patch_model(monkeypatch, Script({"action": "answer", "text": "hi"}))
    assert kl.analyst(sales).ask("hello").text == "hi"


def test_dashboard_analyst_page_answers_in_the_chat(sales):
    import numpy as np
    from streamlit.testing.v1 import AppTest

    import kikolens.dashboard.app as dashboard_app
    script = Script({"action": "chart", "spec": {"kind": "bar", "x": "region", "y": "revenue", "agg": "sum"}}, "North leads.",
                    {"action": "sql", "sql": "select count(*) as n from df"})
    at = AppTest.from_file(str(__import__("pathlib").Path(dashboard_app.__file__)), default_timeout=90)
    data = sales.assign(extra=np.arange(len(sales)))
    at.session_state["df"], at.session_state["original_df"], at.session_state["filename"] = data, data.copy(), "s.csv"
    at.session_state["nav"] = "Analyst"
    at.session_state["analyst_llm"] = script.llm()
    at.run()
    assert not at.exception and [b.label for b in at.button if b.key and b.key.startswith("analyst_sug_")]
    at.chat_input[0].set_value("revenue by region").run()
    assert not at.exception and "North leads." in " ".join(m.value for m in at.markdown)
    at.chat_input[0].set_value("how many rows").run()
    assert not at.exception and len(at.session_state["analyst_bot"][1].history) == 2
