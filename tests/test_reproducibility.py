import json
import os

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import dictionary as dc
from kikolens.analyze import linkage as lk
from kikolens.analyze.champion import compare_saved_models
from kikolens.analyze.modellab import run_model_lab
from kikolens.analyze.predict import build_bundle, save_bundle
from kikolens.cli.main import app
from kikolens.pipeline.recipe import Recipe
from kikolens.pipeline.runner import load_spec, run_pipeline
from kikolens.report.notebook import build_notebook

runner = CliRunner()


def sales(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"), "region": rng.choice(["N", "S"], n),
                       "price": [f"{v:.1f}" for v in rng.uniform(5, 20, n)], "units": rng.uniform(1, 50, n)})
    df["revenue"] = df["units"] * df["price"].astype(float) + rng.normal(0, 5, n)
    return df


# ── Diccionario ──────────────────────────────────────────────────────────────
def test_dictionary_describes_columns_and_warns():
    df = sales().assign(flat=1, email=[f"a{i}@x.org" for i in range(300)])
    df.loc[df.index[:200], "units"] = np.nan
    table = dc.build_dictionary(df, {"revenue": "Money earned"}).set_index("column")
    assert table.loc["revenue", "description"] == "Money earned" and table.loc["date", "meaning"] == "date"
    assert "single value" in table.loc["flat", "notes"] and "mostly empty" in table.loc["units", "notes"]
    assert "looks numeric" in table.loc["price", "notes"] and "personal data" in table.loc["email", "notes"]
    assert table.loc["units", "empty %"] == pytest.approx(66.7, abs=0.1) and table.loc["date", "min"] == "2024-01-01"
    with pytest.raises(ValueError):
        dc.build_dictionary(pd.DataFrame())


def test_dictionary_markdown_escapes_pipes():
    table = dc.build_dictionary(pd.DataFrame({"a|b": ["x|y"] * 30, "n": range(30)}))
    md = dc.to_markdown(table, "Demo", 30)
    assert md.startswith("# Demo") and "x\\|y" in md and md.count("\n| ") >= 3 and "30 rows" in md


# ── Cuaderno ─────────────────────────────────────────────────────────────────
def test_notebook_is_valid_and_its_code_actually_runs(tmp_path):
    csv = tmp_path / "s.csv"
    sales().to_csv(csv, index=False)
    recipe = Recipe()
    recipe.add("convert_type", {"column": "price", "target": "numeric"}, "Converted price.")
    text = build_notebook(str(csv), recipe, target="revenue", drop=["date"], date="date", value="revenue", horizon=5)
    nb = json.loads(text)
    assert nb["nbformat"] == 4 and all(c["cell_type"] in ("markdown", "code") for c in nb["cells"])
    namespace = {}
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), "<cell>", "exec"), namespace)     # lo que el usuario ejecutaría
    finally:
        os.chdir(cwd)
    assert namespace["result"]["task_type"] == "regression" and len(namespace["forecast"]["forecast"]) == 5
    assert pd.api.types.is_numeric_dtype(namespace["df"]["price"])


def test_notebook_without_optional_parts_and_cli(tmp_path):
    nb = json.loads(build_notebook("data.csv"))
    assert len(nb["cells"]) == 6                                    # título, importaciones, datos, carga, calidad (md + código)
    csv = tmp_path / "s.csv"
    sales().to_csv(csv, index=False)
    out = tmp_path / "n.ipynb"
    assert runner.invoke(app, ["notebook", str(csv), str(out), "--target", "revenue", "--exclude", "date"]).exit_code == 0
    assert any("run_model_lab" in "".join(c["source"]) for c in json.loads(out.read_text(encoding="utf-8"))["cells"])
    assert runner.invoke(app, ["notebook", str(csv), str(tmp_path / "n.txt")]).exit_code == 1
    assert runner.invoke(app, ["notebook", str(csv), str(out), "--target", "nope"]).exit_code == 1


# ── Cruce aproximado ─────────────────────────────────────────────────────────
def test_link_records_handles_accents_order_and_typos():
    left = pd.DataFrame({"name": ["García, Ana", "José Luis Pérez", "Maria Lopez", "Unknown Person", None], "id": range(5)})
    right = pd.DataFrame({"client": ["ana garcia", "Jose Luis Perez", "María López", "Pedro Gómez", "Maria Lopes"], "city": list("abcde")})
    res = lk.link_records(left, right, "name", "client", threshold=0.85)
    m = res["matches"].set_index("left row")
    assert m.loc[0, "right row"] == 0 and m.loc[1, "right row"] == 1 and m.loc[2, "right row"] == 2
    assert 3 not in m.index and 4 not in m.index and res["matched"] == 3 and res["unmatched"] == 2
    assert m.loc[2, "exact"] and not m.loc[2, "ambiguous"]            # una coincidencia exacta no es ambigua
    close = lk.link_records(pd.DataFrame({"n": ["John Smith"]}), pd.DataFrame({"n": ["Jon Smith", "John Smyth", "Mary Jones"]}),
                            "n", "n", 0.85, ambiguity=0.06)
    assert close["matched"] == 1 and close["ambiguous"] == 1 and "runner-up" in close["summary"]
    merged = lk.merge_linked(left, right, res["matches"])
    assert merged.loc[0, "city_right"] == "a" and pd.isna(merged.loc[3, "city_right"]) and len(merged) == len(left)


def test_link_records_validation_and_speed_on_a_big_table():
    left = pd.DataFrame({"n": [f"customer {i} family{i % 300}" for i in range(2000)]})
    right = pd.DataFrame({"n": [f"family{i % 300} customer {i}" for i in range(2000)]})
    res = lk.link_records(left, right, "n", "n", 0.9)
    assert res["matched"] == 2000 and res["matches"]["exact"].all()
    with pytest.raises(ValueError, match="threshold"):
        lk.link_records(left, right, "n", "n", 0.2)
    with pytest.raises(ValueError, match="not found"):
        lk.link_records(left, right, "nope", "n")


# ── Pipeline ─────────────────────────────────────────────────────────────────
def pipeline_files(tmp_path, source_df=None):
    train = sales(seed=1)
    (tmp_path / "train.csv").write_text(train.to_csv(index=False))
    (tmp_path / "today.csv").write_text((source_df if source_df is not None else sales(seed=2)).to_csv(index=False))
    recipe = Recipe()
    recipe.add("convert_type", {"column": "price", "target": "numeric"}, "Converted price.")
    (tmp_path / "recipe.json").write_text(recipe.to_json())
    cleaned, _ = recipe.apply(train)
    res = run_model_lab(cleaned, "revenue", models=["Ridge"], folds=3, tune=False, drop=["date"])
    save_bundle(build_bundle(res, cleaned, name="pipe-model"), path=tmp_path / "m.klmodel")
    from kikolens.analyze.rules import suggest_rules
    (tmp_path / "rules.json").write_text(suggest_rules(cleaned).to_json())
    spec = {"source": "today.csv", "recipe": "recipe.json", "rules": "rules.json", "baseline": "train.csv", "model": "m.klmodel",
            "outputs": {"data": "out/clean.csv", "predictions": "out/scored.csv", "report": "out/report.html", "log": "out/log.json"},
            "fail_on": ["rules", "drift"], "min_quality": 50}
    (tmp_path / "pipeline.json").write_text(json.dumps(spec))
    return tmp_path / "pipeline.json"


def test_pipeline_runs_every_stage_and_writes_outputs(tmp_path):
    result = run_pipeline(pipeline_files(tmp_path))
    assert result["status"] == "ok", result
    assert [s["step"] for s in result["steps"]] == ["load", "recipe", "quality", "rules", "drift", "score"]
    assert len(pd.read_csv(tmp_path / "out" / "scored.csv")) == 300 and "predicted_revenue" in pd.read_csv(tmp_path / "out" / "scored.csv")
    assert (tmp_path / "out" / "report.html").exists() and json.loads((tmp_path / "out" / "log.json").read_text())["status"] == "ok"


def test_pipeline_raises_the_alarm_on_broken_rules_and_drift(tmp_path):
    bad = sales(seed=2)
    bad["units"] = bad["units"] * 40 + 5000            # fuera de lo normal en el entrenamiento
    bad["region"] = "Z"
    path = pipeline_files(tmp_path, bad)
    result = run_pipeline(path)
    assert result["status"] == "alert" and result["alerts"]
    code = runner.invoke(app, ["run", str(path)]).exit_code
    assert code == 2
    good_path = pipeline_files(tmp_path / "ok_dir" if (tmp_path / "ok_dir").mkdir() is None else tmp_path)
    assert runner.invoke(app, ["run", str(good_path)]).exit_code == 0


def test_pipeline_spec_validation(tmp_path):
    def spec(**kw):
        p = tmp_path / "p.json"
        p.write_text(json.dumps(kw))
        return p
    with pytest.raises(ValueError, match="source"):
        load_spec(spec(recipe="r.json"))
    with pytest.raises(ValueError, match="Unknown pipeline key"):
        load_spec(spec(source="a.csv", sourc="x"))
    with pytest.raises(ValueError, match="Unknown output"):
        load_spec(spec(source="a.csv", outputs={"pdf": "x"}))
    with pytest.raises(ValueError, match="fail_on"):
        load_spec(spec(source="a.csv", fail_on=["weather"]))
    with pytest.raises(ValueError, match="Could not read"):
        load_spec(tmp_path / "missing.json")
    assert runner.invoke(app, ["run", str(tmp_path / "missing.json")]).exit_code == 1


# ── Campeón y aspirante ──────────────────────────────────────────────────────
def test_champion_challenger_picks_the_better_model_and_judges_the_gap():
    df = sales(500, seed=3)
    train, test = df.iloc[:300], df.iloc[300:]
    cleaned = train.assign(price=train["price"].astype(float)).drop(columns="date")
    good = build_bundle(run_model_lab(cleaned, "revenue", models=["Ridge"], folds=3, tune=False), cleaned, name="good")
    weak_df = cleaned.drop(columns=["units"])
    weak = build_bundle(run_model_lab(weak_df, "revenue", models=["Ridge"], folds=3, tune=False), weak_df, name="weak")
    test = test.assign(price=test["price"].astype(float))
    res = compare_saved_models({"weak": weak, "good": good}, test)
    assert res["champion"] == "good" and res["reliable"] and res["table"]["rank"].tolist() == [1, 2]
    same = compare_saved_models({"a": good, "b": good}, test)
    assert not same["reliable"] and same["p_value"] == 1.0
    with pytest.raises(ValueError, match="at least two"):
        compare_saved_models({"a": good}, test)
    with pytest.raises(ValueError, match="named 'revenue'"):
        compare_saved_models({"a": good, "b": weak}, test.drop(columns="revenue"))
    with pytest.raises(ValueError, match="At least 30"):
        compare_saved_models({"a": good, "b": weak}, test.head(10))


def test_models_compare_cli(tmp_path):
    df = sales(400, seed=4)
    cleaned = df.assign(price=df["price"].astype(float)).drop(columns="date")
    a = build_bundle(run_model_lab(cleaned, "revenue", models=["Ridge"], folds=3, tune=False), cleaned, name="a")
    b = build_bundle(run_model_lab(cleaned, "revenue", models=["Random Forest"], folds=3, tune=False), cleaned, name="b")
    pa, pb = save_bundle(a, path=tmp_path / "a.klmodel"), save_bundle(b, path=tmp_path / "b.klmodel")
    csv = tmp_path / "t.csv"
    cleaned.to_csv(csv, index=False)
    out = runner.invoke(app, ["models", "compare", str(csv), str(pa), str(pb)])
    assert out.exit_code == 0 and "comes first" in out.output
    assert runner.invoke(app, ["models", "compare", str(csv), str(pa)]).exit_code == 1


def test_dictionary_and_link_cli(tmp_path):
    csv = tmp_path / "s.csv"
    sales().to_csv(csv, index=False)
    out = tmp_path / "d.md"
    assert runner.invoke(app, ["dictionary", str(csv), "--output", str(out)]).exit_code == 0 and "| column |" in out.read_text(encoding="utf-8")
    assert runner.invoke(app, ["dictionary", str(csv)]).exit_code == 0
    assert runner.invoke(app, ["dictionary", str(csv), "--output", str(tmp_path / "d.txt")]).exit_code == 1
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame({"name": ["Ana García", "Luis Pérez"]}).to_csv(a, index=False)
    pd.DataFrame({"client": ["garcia ana", "Perez, Luis"], "vip": [1, 0]}).to_csv(b, index=False)
    joined = tmp_path / "j.csv"
    res = runner.invoke(app, ["link", str(a), str(b), "name", "client", "--output", str(joined)])
    assert res.exit_code == 0 and list(pd.read_csv(joined)["vip_right"]) == [1, 0]
