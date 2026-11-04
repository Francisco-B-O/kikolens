import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import cleaning as cl
from kikolens.analyze import monitor as mt
from kikolens.analyze import rules as rl
from kikolens.cli.main import app
from kikolens.pipeline.recipe import Recipe

runner = CliRunner()


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({"a": rng.normal(10, 2, n), "region": rng.choice(["N", "S", "E"], n),
                          "age": rng.integers(18, 70, n).astype(float)})
    frame["b"] = frame["a"] * 3 + rng.normal(size=n) * 0.3
    frame["income"] = np.where(frame["region"] == "N", 80, 40) + rng.normal(size=n)
    return frame


# ── Datos faltantes ──────────────────────────────────────────────────────────
def test_missing_patterns_detect_dependent_gaps(df):
    gaps = df.copy()
    gaps.loc[gaps["a"] > 12, "age"] = np.nan                 # el hueco depende de `a`
    gaps.loc[::9, "income"] = np.nan                         # el hueco es aleatorio
    info = cl.missing_patterns(gaps)
    related = info["related"].set_index("column")
    assert "age" in related.index and related.loc["age", "depends on"] in ("a", "b")
    assert "income" not in related.index
    assert set(info["by_column"]["column"]) == {"age", "income"}
    assert info["patterns"]["share"].sum() == pytest.approx(1.0) and "not random" in info["summary"]


def test_missing_patterns_without_gaps_and_co_missing(df):
    assert cl.missing_patterns(df)["by_column"].empty and "no missing" in cl.missing_patterns(df)["summary"]
    gaps = df.copy()
    gaps.loc[:30, ["age", "income"]] = np.nan
    co = cl.missing_patterns(gaps)["co_missing"]
    assert co.loc["age", "income"] == pytest.approx(1.0)


# ── Imputación ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("method", ["knn", "iterative"])
def test_model_based_imputation_beats_the_mean(df, method):
    rng = np.random.default_rng(1)
    gaps = df.copy()
    hide = rng.choice(len(df), 60, replace=False)
    truth = gaps.loc[hide, "b"].copy()
    gaps.loc[hide, "b"] = np.nan
    out, message = cl.impute_advanced(gaps, method, columns=["b"])
    assert out["b"].isna().sum() == 0 and "60" in message
    model_error = (out.loc[hide, "b"] - truth).abs().mean()
    mean_error = (gaps["b"].mean() - truth).abs().mean()
    assert model_error < mean_error / 3                      # b = 3a: las demás columnas lo explican
    assert gaps["b"].isna().sum() == 60                      # el original no cambia


def test_group_imputation_and_indicator(df):
    gaps = df.copy()
    gaps.loc[gaps.index[:50], "income"] = np.nan
    gaps.loc[gaps.index[:20], "region"] = np.nan
    out, _ = cl.impute_advanced(gaps, "group", columns=["income"], group_by="region", add_indicator=True)
    assert out["income"].isna().sum() == 0 and out["income_was_missing"].sum() == 50
    north = out[(gaps["region"] == "N") & gaps["income"].isna()]["income"]
    assert north.between(75, 85).all()                        # usa la mediana de su grupo, no la global
    text = gaps.assign(region=gaps["region"])
    filled, _ = cl.impute_advanced(text, "group", columns=["region"], group_by="age")
    assert filled["region"].notna().all()


def test_integer_columns_stay_integer_valued(df):
    gaps = df.copy()
    gaps.loc[::7, "age"] = np.nan
    out, _ = cl.impute_advanced(gaps, "knn", columns=["age"])
    assert (out["age"] % 1 == 0).all()


def test_imputation_validation(df):
    gaps = df.copy()
    gaps.loc[0, "region"] = np.nan
    with pytest.raises(ValueError, match="Method"):
        cl.impute_advanced(df, "magic")
    with pytest.raises(ValueError, match="None of"):
        cl.impute_advanced(df, "knn", columns=["a"])
    with pytest.raises(ValueError, match="numbers only"):
        cl.impute_advanced(gaps, "knn", columns=["region"])
    with pytest.raises(ValueError, match="groups"):
        cl.impute_advanced(gaps, "group", columns=["region"])
    with pytest.raises(ValueError, match="not found"):
        cl.impute_advanced(df, "knn", columns=["ghost"])


def test_imputation_is_a_recipe_step(df):
    gaps = df.copy()
    gaps.loc[::5, "b"] = np.nan
    recipe = Recipe()
    recipe.add("impute_advanced", {"method": "knn", "columns": ["b"], "n_neighbors": 5, "add_indicator": False})
    again = Recipe.from_json(recipe.to_json())
    assert again.apply(gaps)[0]["b"].isna().sum() == 0


# ── Categorías parecidas y casi duplicados ──────────────────────────────────
def test_similar_values_group_spelling_variants():
    s = pd.Series(["New York"] * 10 + ["new york "] * 3 + ["NewYork"] * 2 + ["New Yrok"] * 2 + ["Boston"] * 8 + ["Madrid"] * 5
                  + ["Madird"] * 1 + [None])
    groups = cl.similar_values(s, 0.8)
    by = groups.set_index("keep")["variants"]
    assert set(by["New York"]) == {"new york ", "NewYork", "New Yrok"} and by["Madrid"] == ["Madird"]
    assert "Madrid" not in cl.similar_values(s)["keep"].tolist()      # con 0,85 un solo cambio de orden no basta
    assert "Boston" not in groups["keep"].tolist()
    with pytest.raises(ValueError, match="between 0.6 and 1"):
        cl.similar_values(s, 0.2)


def test_standardize_categories_operation():
    df = pd.DataFrame({"city": ["Paris"] * 6 + ["paris"] * 2 + ["Pariss"] + ["Rome"] * 5, "n": range(14)})
    out, message = cl.standardize_categories(df, "city")
    assert set(out["city"]) == {"Paris", "Rome"} and "2 spelling" in message
    assert set(df["city"]) == {"Paris", "paris", "Pariss", "Rome"}
    with pytest.raises(ValueError, match="No similar"):
        cl.standardize_categories(out, "city")
    with pytest.raises(ValueError, match="not a text"):
        cl.standardize_categories(df, "n")


def test_near_duplicate_rows_find_typos_but_not_exact_copies():
    df = pd.DataFrame({"name": ["Maria Lopez", "Maria Lopes", "John Smith", "John Smith", "Anna Petrova", "Peter Jones"] * 3
                       + ["Maria Lopez"], "city": ["Madrid", "Madrid", "Leeds", "Leeds", "Sofia", "Cardiff"] * 3 + ["Madrid"]})
    pairs = cl.near_duplicate_rows(df, threshold=0.8)
    shown = set(zip(pairs["A"], pairs["B"])) | set(zip(pairs["B"], pairs["A"]))
    assert ("maria lopez | madrid", "maria lopes | madrid") in shown
    assert all(a != b for a, b in zip(pairs["A"], pairs["B"]))          # nunca pares idénticos
    with pytest.raises(ValueError, match="limited"):
        cl.near_duplicate_rows(pd.concat([df] * 400), max_rows=100)
    with pytest.raises(ValueError, match="between"):
        cl.near_duplicate_rows(df, threshold=1.0)
    out, message = cl.drop_rows(df, [1, 3])
    assert len(out) == len(df) - 2 and "2" in message
    with pytest.raises(ValueError, match="None of"):
        cl.drop_rows(df, [9999])


# ── Reglas ───────────────────────────────────────────────────────────────────
@pytest.fixture
def orders():
    return pd.DataFrame({"id": [1, 2, 3, 3, 5], "price": [10.0, -5.0, 20.0, 0.0, None],
                         "status": ["new", "paid", "lost", "new", "paid"],
                         "code": ["AB-12", "AB-1", "XY-99", "XY-98", "bad"],
                         "start": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-03-05", "2030-01-01"]),
                         "end": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-03-10", "2024-03-06", "2030-02-01"])})


@pytest.mark.parametrize("rule,expected", [
    ({"check": "not_null", "column": "price"}, [4]), ({"check": "unique", "column": "id"}, [2, 3]),
    ({"check": "range", "column": "price", "min": 0}, [1]), ({"check": "range", "column": "price", "max": 15}, [2]),
    ({"check": "in_set", "column": "status", "values": ["new", "paid"]}, [2]),
    ({"check": "regex", "column": "code", "pattern": r"[A-Z]{2}-\d{2}"}, [1, 4]),
    ({"check": "compare", "column": "end", "op": ">=", "other": "start"}, [1]),
    ({"check": "not_future", "column": "start"}, [4]),
])
def test_each_check_flags_the_right_rows(orders, rule, expected):
    assert orders.index[rl.violations(orders, rule)].tolist() == expected


def test_rule_validation_errors(orders):
    for rule, text in [({"check": "magic", "column": "id"}, "Unknown check"), ({"check": "range", "column": "price"}, "minimum"),
                       ({"check": "range", "column": "status", "min": 0}, "numeric"), ({"check": "in_set", "column": "status"}, "at least one"),
                       ({"check": "regex", "column": "code", "pattern": "("}, "not valid"),
                       ({"check": "regex", "column": "code", "pattern": "a" * 300}, "characters"),
                       ({"check": "compare", "column": "id", "op": "??", "other": "price"}, "Operator"),
                       ({"check": "not_future", "column": "id"}, "not a date"), ({"check": "unique", "column": "ghost"}, "not found")]:
        with pytest.raises(ValueError, match=text):
            rl.violations(orders, rule)


def test_ruleset_validate_roundtrip_and_a_broken_rule_does_not_stop_the_rest(orders):
    rs = rl.RuleSet()
    rs.add({"check": "unique", "column": "id"})
    rs.add({"check": "range", "column": "price", "min": 0})
    rs.add({"check": "not_null", "column": "ghost"})
    again = rl.RuleSet.from_json(rs.to_json())
    result = again.validate(orders)
    table = result["table"].set_index("rule")
    assert table.loc["id must be unique", "violations"] == 2 and not result["passed"]
    assert "not found" in table.loc["ghost must not be empty", "error"]
    assert result["broken"] == 2 and result["rows_with_problems"] == 3
    assert len(again.violating_rows(orders, again.rules[0]["id"])) == 2
    again.remove(again.rules[0]["id"])
    assert len(again.rules) == 2
    with pytest.raises(ValueError, match="Unknown rule"):
        again.violating_rows(orders, "nope")


def test_ruleset_json_validation():
    with pytest.raises(ValueError, match="valid JSON"):
        rl.RuleSet.from_json("{")
    with pytest.raises(ValueError, match="not a KikoLens rule set"):
        rl.RuleSet.from_json("[]")
    with pytest.raises(ValueError, match="newer"):
        rl.RuleSet.from_json(json.dumps({"kikolens_rules": 9, "rules": []}))
    with pytest.raises(ValueError, match="Unknown check"):
        rl.RuleSet.from_json(json.dumps({"rules": [{"check": "os.system", "column": "a"}]}))
    rs = rl.RuleSet()
    for _ in range(rl.MAX_RULES):
        rs.add({"check": "not_null", "column": "a"})
    with pytest.raises(ValueError, match="At most"):
        rs.add({"check": "not_null", "column": "a"})


def test_suggested_rules_pass_on_the_data_they_came_from(df):
    data = df.assign(id=range(len(df)), when=pd.date_range("2024-01-01", periods=len(df)))
    suggested = rl.suggest_rules(data)
    kinds = {r["check"] for r in suggested.rules}
    assert {"not_null", "unique", "range", "in_set", "not_future"} <= kinds
    assert suggested.validate(data)["passed"]
    broken = data.copy()
    broken.loc[0, "region"] = "Z"
    assert not suggested.validate(broken)["passed"]


def test_monitor_checks_rules(tmp_path, df, monkeypatch):
    monkeypatch.setenv("KIKOLENS_HOME", str(tmp_path / "home"))
    rules = rl.RuleSet()
    rules.add({"check": "range", "column": "age", "min": 18, "max": 70})
    mon = mt.create_monitor("m", "(uploaded)", df, rules=rules.to_json())
    assert mt.run_check(mon, df)["status"] == "ok"
    broken = df.assign(age=df["age"] + 100)
    record = mt.run_check(mon, broken)
    assert record["rules_broken"] == 1 and any("Rule broken" in a["message"] for a in record["alerts"])
    with pytest.raises(ValueError, match="not a KikoLens rule set"):
        mt.create_monitor("bad", "(uploaded)", df, rules="[]")


def test_cli_validate(tmp_path, orders):
    source = tmp_path / "o.csv"
    orders.to_csv(source, index=False)
    ok = runner.invoke(app, ["validate", str(source), "--save", str(tmp_path / "r.json")])
    assert ok.exit_code == 0 and (tmp_path / "r.json").exists()
    rules = rl.RuleSet()
    rules.add({"check": "unique", "column": "id"})
    (tmp_path / "mine.json").write_text(rules.to_json())
    assert runner.invoke(app, ["validate", str(source), str(tmp_path / "mine.json")]).exit_code == 2
    (tmp_path / "bad.json").write_text("{")
    assert runner.invoke(app, ["validate", str(source), str(tmp_path / "bad.json")]).exit_code == 1
