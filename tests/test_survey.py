import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import survey as sv


def likert_items(n=800, seed=0, k=5, noise=0.9):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    cols = {f"q{i}": np.clip(np.round(3 + z + rng.normal(0, noise, n)), 1, 5) for i in range(k)}
    return pd.DataFrame(cols)


# ── Escalas ──────────────────────────────────────────────────────────────────
def test_text_scales_in_both_languages_become_numbers():
    en = pd.Series(["Strongly agree", "agree", "Neither agree nor disagree", "disagree", "Strongly disagree", None])
    values, name = sv.to_numeric_scale(en)
    assert values.dropna().tolist() == [5, 4, 3, 2, 1] and name == "agreement" and values.isna().iloc[-1]
    es, name_es = sv.to_numeric_scale(pd.Series(["Muy satisfecho", "satisfecho", "Regular", "insatisfecho", "muy insatisfecho"]))
    assert es.tolist() == [5, 4, 3, 2, 1] and name_es == "satisfaccion"
    assert sv.to_numeric_scale(pd.Series(["1", "2", "3"] * 10))[0].tolist()[:3] == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="not a numeric scale"):
        sv.to_numeric_scale(pd.Series(["red", "blue", "green"] * 5, name="colour"))
    numbers, none = sv.to_numeric_scale(pd.Series([1, 2, 3]))
    assert none is None and numbers.dtype == float


def test_likert_summary_counts_the_favourable_answers_and_reverses_items():
    answers = [5] * 40 + [4] * 20 + [3] * 20 + [2] * 10 + [1] * 10
    df = pd.DataFrame({"good": answers, "bad": [6 - a for a in answers]})
    res = sv.likert_summary(df, ["good", "bad"])
    t = res["table"].set_index("question")
    assert t.loc["good", "favourable"] == pytest.approx(0.60) and t.loc["good", "unfavourable"] == pytest.approx(0.20) and t.loc["good", "neutral"] == pytest.approx(0.20)
    assert t.loc["bad", "favourable"] == pytest.approx(0.20) and t.loc["good", "mean"] == pytest.approx(3.7) and t.loc["good", "% 5"] == pytest.approx(0.4)
    flipped = sv.likert_summary(df, ["good", "bad"], reverse=["bad"])
    assert flipped["table"].set_index("question").loc["bad", "favourable"] == pytest.approx(0.60) and "Most favourable" in res["summary"] and len(res["figure"].data) == 5
    text = sv.likert_summary(pd.DataFrame({"a": ["Agree"] * 30 + ["Strongly agree"] * 30 + ["Disagree"] * 10}), ["a"])
    assert text["table"].iloc[0]["favourable"] == pytest.approx(60 / 70) and text["text_scales"] == {"a": "agreement"}
    odd = sv.likert_summary(pd.DataFrame({"a": [1, 2, 3, 4, 5, 9] * 10}), ["a"], scale=(1, 5))
    assert any("outside the scale" in n for n in odd["notes"])
    for kwargs, fragment in (({"items": ["nope"]}, "not found"), ({"items": ["good"], "reverse": ["bad"]}, "not one of")):
        with pytest.raises(ValueError, match=fragment):
            sv.likert_summary(df, **kwargs)
    with pytest.raises(ValueError, match="at least 20"):
        sv.likert_summary(df.head(5), ["good"])


# ── Fiabilidad ───────────────────────────────────────────────────────────────
def test_cronbach_alpha_matches_the_theoretical_value_and_its_interval_covers_it():
    rng = np.random.default_rng(1)
    z = rng.normal(size=4000)
    df = pd.DataFrame({f"q{i}": z + rng.normal(0, 1.0, 4000) for i in range(5)})
    rho = 0.5
    expected = 5 * rho / (1 + 4 * rho)
    res = sv.reliability(df, list(df.columns))
    assert res["alpha"] == pytest.approx(expected, abs=0.01) and res["interval"][0] < expected < res["interval"][1] and res["standardized_alpha"] == pytest.approx(expected, abs=0.01)
    assert res["mean_inter_item_correlation"] == pytest.approx(rho, abs=0.02) and res["label"] == "good" and res["items"] == 5 and res["n"] == 4000


def test_a_reverse_worded_item_is_detected_and_fixed_by_reversing_it():
    items = likert_items()
    items["q4"] = 6 - items["q4"]                                                     # redactada al revés
    bad = sv.reliability(items, list(items.columns))
    assert bad["table"].set_index("question").loc["q4", "corrected item-total correlation"] < 0 and any("q4" in n and "negatively" in n for n in bad["notes"])
    fixed = sv.reliability(items, list(items.columns), reverse=["q4"])
    assert fixed["alpha"] > bad["alpha"] + 0.3 and fixed["alpha"] > 0.7 and not any("negatively" in n for n in fixed["notes"])


def test_removing_an_unrelated_question_raises_alpha_and_the_notes_say_so():
    items = likert_items(1000)
    items["noise"] = np.random.default_rng(2).integers(1, 6, len(items))
    res = sv.reliability(items, list(items.columns))
    row = res["table"].set_index("question").loc["noise"]
    assert row["alpha without it"] > res["alpha"] + 0.02 and row["corrected item-total correlation"] < 0.1 and any("noise" in n and "raise alpha" in n for n in res["notes"])
    with pytest.raises(ValueError, match="at least 3"):
        sv.reliability(items, ["q0", "q1"])
    with pytest.raises(ValueError, match="answered every question"):
        sv.reliability(items.assign(q0=np.where(np.arange(len(items)) % 2, np.nan, 3.0)).head(30).iloc[::2].assign(q1=np.nan), ["q0", "q1", "q2"])


# ── NPS ──────────────────────────────────────────────────────────────────────
def test_nps_is_promoters_minus_detractors_with_an_interval():
    d = pd.DataFrame({"s": [10] * 50 + [9] * 10 + [8] * 20 + [7] * 10 + [3] * 10})
    res = sv.nps(d, "s")
    assert res["nps"] == pytest.approx(50) and res["promoters"] == pytest.approx(0.6) and res["detractors"] == pytest.approx(0.1) and res["passives"] == pytest.approx(0.3)
    se = np.sqrt((0.6 + 0.1 - 0.5 ** 2) / 100) * 100
    assert res["interval"][0] == pytest.approx(50 - 1.96 * se, abs=0.01) and any("Only 60 answers" in n for n in sv.nps(pd.DataFrame({"s": [10, 9, 8, 3, 7, 10] * 10}), "s")["notes"])
    assert sv.nps(pd.DataFrame({"s": [0, 6, 7, 8, 9, 10] * 10}), "s")["nps"] == pytest.approx(0)
    for frame, fragment in ((pd.DataFrame({"s": [11] * 30}), "0 to 10"), (pd.DataFrame({"s": [5] * 5}), "At least")):
        with pytest.raises(ValueError, match=fragment):
            sv.nps(frame, "s")


def test_nps_by_group_flags_only_the_group_that_really_differs():
    rng = np.random.default_rng(3)
    n = 3000
    region = rng.choice(["north", "south", "east"], n)
    score = np.where(region == "south", rng.choice(range(0, 11), n, p=[.02, .02, .03, .03, .04, .05, .08, .12, .18, .2, .23]), rng.choice(range(0, 11), n, p=[.06, .04, .06, .06, .08, .1, .1, .12, .12, .1, .16]))
    res = sv.nps(pd.DataFrame({"score": score, "region": region}), "score", "region")
    t = res["table"].set_index("group")
    assert t.loc["south", "nps"] > t.loc["north", "nps"] + 15 and t.loc["south", "p adjusted"] < 0.01
    note = " ".join(res["notes"])
    assert "south" in note and "north" not in note and "east" not in note                       # solo el grupo que de verdad se sale del reparto global
    with pytest.raises(ValueError, match="not found"):
        sv.nps(pd.DataFrame({"score": score}), "score", "region")


# ── Tablas cruzadas ──────────────────────────────────────────────────────────
def test_crosstab_residuals_point_at_the_cells_that_differ_from_chance():
    rng = np.random.default_rng(4)
    n = 2000
    a = rng.choice(["young", "adult", "senior"], n)
    b = np.where(a == "young", rng.choice(["app", "web"], n, p=[0.8, 0.2]), rng.choice(["app", "web"], n, p=[0.4, 0.6]))
    res = sv.crosstab(pd.DataFrame({"age": a, "channel": b}), "age", "channel")
    assert res["p_value"] < 1e-6 and res["cramers_v"] == pytest.approx(0.38, abs=0.06) and res["strength"] == "moderate"
    cells = {(r, c): v for r, c, v in res["standouts"]}
    assert cells[("young", "app")] > 10 and cells[("young", "web")] < -10 and cells[("adult", "app")] < -5 and cells[("senior", "app")] < -5      # respecto al total, los demás usan menos la app
    assert res["residuals"].to_numpy().shape == (3, 2) and (res["counts"].to_numpy().sum() == n) and "related" in res["summary"]
    independent = sv.crosstab(pd.DataFrame({"a": rng.choice(list("xyz"), n), "b": rng.choice(list("uv"), n)}), "a", "b")
    assert independent["p_value"] > 0.01 and independent["strength"] == "negligible"
    sparse = sv.crosstab(pd.DataFrame({"a": list("ab") * 15 + ["c"], "b": list("uv") * 15 + ["u"]}), "a", "b")
    assert any("fewer than 5" in n for n in sparse["notes"])
    for frame, fragment in ((pd.DataFrame({"a": ["x"] * 30, "b": ["u", "v"] * 15}), "two different"), (pd.DataFrame({"a": list("xy"), "b": list("uv")}), "At least")):
        with pytest.raises(ValueError, match=fragment):
            sv.crosstab(frame, "a", "b")


# ── Factores clave ───────────────────────────────────────────────────────────
def test_relative_weights_share_out_exactly_the_explained_variance():
    rng = np.random.default_rng(5)
    n = 4000
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    uncorrelated = X.assign(y=0.6 * X["a"] + 0.3 * X["b"] + rng.normal(0, 0.5, n))
    res = sv.key_drivers(uncorrelated, "y", list("abc"))
    t = res["table"].set_index("question")
    assert t.loc["a", "share of R2"] == pytest.approx(t.loc["a", "correlation with overall"] ** 2, abs=0.01) and t.loc["c", "importance"] < 0.02
    assert list(res["table"]["question"])[:2] == ["a", "b"] and res["table"]["importance"].sum() == pytest.approx(1.0)
    z = rng.normal(size=n)
    corr = pd.DataFrame({"a": z + rng.normal(0, 0.6, n), "b": z + rng.normal(0, 0.6, n), "c": rng.normal(size=n)})
    corr["y"] = 0.5 * corr["a"] + 0.5 * corr["b"] + 0.2 * corr["c"] + rng.normal(0, 1, n)
    both = sv.key_drivers(corr, "y", list("abc"))
    design = np.column_stack([np.ones(n), corr[list("abc")].to_numpy()])
    fitted = design @ np.linalg.lstsq(design, corr["y"].to_numpy(), rcond=None)[0]
    r2 = 1 - np.sum((corr["y"] - fitted) ** 2) / np.sum((corr["y"] - corr["y"].mean()) ** 2)
    assert both["r2"] == pytest.approx(r2, abs=1e-9) and both["table"]["share of R2"].sum() == pytest.approx(r2, abs=1e-9)
    shares = both["table"].set_index("question")["importance"]
    assert abs(shares["a"] - shares["b"]) < 0.08 and shares["a"] > shares["c"]                     # a y b se parecen y se reparten el efecto


def test_priorities_cross_importance_with_how_well_each_question_scores():
    rng = np.random.default_rng(6)
    n = 1500
    d = pd.DataFrame({"delivery": rng.integers(1, 6, n), "price": rng.integers(1, 6, n), "packaging": rng.integers(1, 6, n), "colour": rng.integers(1, 6, n)})
    d["delivery"] = np.clip(d["delivery"] - 1, 1, 5)                              # puntúa bajo
    d["overall"] = np.clip(np.round(1 + 0.6 * d["delivery"] + 0.3 * d["price"] + 0.05 * d["colour"] + rng.normal(0, 0.5, n)), 1, 5)
    res = sv.key_drivers(d, "overall", ["delivery", "price", "packaging", "colour"])
    t = res["table"].set_index("question")
    assert res["table"].iloc[0]["question"] == "delivery" and t.loc["delivery", "priority"] == "improve first" and t.loc["packaging", "importance"] < 0.05
    assert "improve first: delivery" in res["summary"] and len(res["figure"].data) == 1
    for args, fragment in ((("overall", ["overall", "price"]), "own drivers"), (("overall", ["price"]), "at least 2"), (("nope", ["price", "colour"]), "not found")):
        with pytest.raises(ValueError, match=fragment):
            sv.key_drivers(d, *args)
    with pytest.raises(ValueError, match="practically the same"):
        sv.key_drivers(d.assign(twin=d["price"]), "overall", ["price", "twin", "colour"])
    with pytest.raises(ValueError, match="same answer"):
        sv.key_drivers(d.assign(flat=3), "overall", ["price", "flat"])


# ── Grupos y calidad ─────────────────────────────────────────────────────────
def test_group_comparison_finds_the_question_that_differs_and_corrects_for_many():
    rng = np.random.default_rng(7)
    n = 900
    group = rng.choice(["a", "b"], n)
    d = pd.DataFrame({"group": group, **{f"q{i}": rng.integers(1, 6, n) for i in range(6)}})
    d["q0"] = np.clip(d["q0"] + (group == "b"), 1, 5)
    res = sv.compare_groups(d, [f"q{i}" for i in range(6)], "group")
    t = res["table"].set_index("question")
    assert bool(t.loc["q0", "differs"]) and not t.drop(index="q0")["differs"].any() and t.loc["q0", "measure"] == "rank-biserial correlation" and t.loc["q0", "effect"] < 0
    assert (t["p adjusted"] >= t["p"] - 1e-12).all() and "1 of 6" in res["summary"]
    three = d.assign(group=rng.choice(["a", "b", "c"], n))
    assert sv.compare_groups(three, ["q1"], "group")["table"].iloc[0]["measure"] == "epsilon squared"
    with pytest.raises(ValueError, match="two groups"):
        sv.compare_groups(d.assign(group="a"), ["q0"], "group")


def test_response_quality_finds_straight_liners_and_blank_answers():
    rng = np.random.default_rng(8)
    d = pd.DataFrame({f"q{i}": rng.integers(1, 6, 200) for i in range(8)}).astype(float)
    d.iloc[:25] = 4.0
    d.iloc[25:35, 2:] = np.nan
    d["secs"] = rng.normal(300, 30, 200)
    d.loc[40:44, "secs"] = 20
    res = sv.response_quality(d, [f"q{i}" for i in range(8)], "secs")
    t = res["table"]
    assert res["straight_lining"] >= 25 and res["mostly_blank"] == 10 and t.loc[:24, "straight-lining"].all() and t.loc[40:44, "too fast"].all() and res["suspect"] >= 40
    assert any("same answer" in n for n in res["notes"]) and "second look" in res["summary"]
    with pytest.raises(ValueError, match="numeric column"):
        sv.response_quality(d, [f"q{i}" for i in range(8)], "q0x")


def test_python_api_and_terminal(tmp_path):
    import kikolens as kl
    from typer.testing import CliRunner

    from kikolens.cli.main import app
    items = likert_items(300)
    items["overall"] = np.clip(np.round(items.mean(axis=1) + np.random.default_rng(9).normal(0, 0.4, 300)), 1, 5)
    items["recommend"] = np.random.default_rng(10).integers(0, 11, 300)
    items["region"] = np.random.default_rng(11).choice(["n", "s"], 300)
    q = [f"q{i}" for i in range(5)]
    assert kl.likert(items, q)["kind"] == "likert" and kl.cronbach(items, q)["kind"] == "reliability" and kl.nps(items, "recommend", "region")["kind"] == "nps"
    assert kl.crosstab(items, "region", "q0")["kind"] == "crosstab" and kl.key_drivers(items, "overall", q)["kind"] == "key drivers"
    assert kl.compare_items(items, q, "region")["kind"] == "item comparison" and kl.response_quality(items, q)["kind"] == "response quality"
    csv = tmp_path / "s.csv"
    items.to_csv(csv, index=False)
    runner = CliRunner()
    for args, text in ((["likert", str(csv), "--items", ",".join(q)], "Most favourable"), (["reliability", str(csv), "--items", ",".join(q)], "Cronbach"),
                       (["nps", str(csv), "recommend", "--group", "region"], "NPS"), (["crosstab", str(csv), "region", "q0"], "related"),
                       (["key-drivers", str(csv), "overall", "--items", ",".join(q)], "explain"), (["survey-groups", str(csv), "region", "--items", ",".join(q)], "question"),
                       (["response-quality", str(csv), "--items", ",".join(q)], "second look")):
        out = runner.invoke(app, args)
        assert out.exit_code == 0 and text in " ".join(out.output.split()), (args, out.output)
    assert runner.invoke(app, ["nps", str(csv), "nope"]).exit_code == 1
