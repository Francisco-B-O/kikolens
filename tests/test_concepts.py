import numpy as np
import pytest

from kikolens.analyze import concepts as cp


def test_every_concept_runs_and_explains_itself():
    for name, spec in cp.CONCEPTS.items():
        res = cp.run(name)
        assert res["concept"] == name and res["figure"] is not None and len(res["summary"]) > 80 and len(res["lesson"]) > 80 and res["try"]
        assert set(res["params"]) == set(spec["params"])
        for key, value in spec["params"].items():
            assert value["type"] in ("int", "float", "choice") and value["label"]
            if value["type"] != "choice":
                assert value["min"] <= value["default"] <= value["max"], (name, key)


def test_the_same_seed_gives_the_same_answer_and_another_seed_a_different_one():
    a, b, c = cp.run("clt", seed=3), cp.run("clt", seed=3), cp.run("clt", seed=4)
    assert a["observed_sd"] == b["observed_sd"] and a["observed_sd"] != c["observed_sd"]


def test_clt_spread_shrinks_with_the_square_root_of_n_and_skewness_fades():
    small, large = cp.run("clt", n=5), cp.run("clt", n=200)
    assert small["observed_sd"] == pytest.approx(small["predicted_sd"], rel=0.05) and large["observed_sd"] == pytest.approx(large["predicted_sd"], rel=0.05)
    assert large["observed_sd"] < small["observed_sd"] / 5 and abs(large["skew_means"]) < abs(small["skew_means"]) < cp.run("clt", n=1)["skew_population"]
    assert cp.run("clt", n=30)["normality_gap"] < cp.run("clt", n=2)["normality_gap"]
    assert cp.run("clt", distribution="exponential")["population_mean"] == pytest.approx(1.0, abs=0.02)
    assert any("heavy tails" in n for n in cp.run("clt", distribution="heavy", n=30)["notes"]) and not cp.run("clt", distribution="heavy", n=500)["notes"]


def test_a_test_with_no_effect_raises_false_alarms_at_the_stated_rate_and_peeking_inflates_them():
    honest = cp.run("p_values", effect=0.0, looks=1, experiments=8000)
    assert honest["significant_rate"] == pytest.approx(0.05, abs=0.012)
    counts, _ = np.histogram(honest["p_values"], bins=10, range=(0, 1))
    assert counts.max() / counts.min() < 1.25                                       # bajo la hipótesis nula, los p-valores se reparten por igual
    peeking = cp.run("p_values", effect=0.0, looks=10, experiments=8000)
    assert peeking["significant_rate"] > 0.15 and peeking["first_look_rate"] == pytest.approx(0.05, abs=0.015) and "pushed the false alarms" in peeking["summary"]


def test_power_matches_the_textbook_values():
    d = cp.run("p_values", effect=0.5, n=64, experiments=8000)
    assert d["significant_rate"] == pytest.approx(0.80, abs=0.02)
    res = cp.run("power", effect=0.5, alpha=0.05, target=0.8)
    assert res["needed_per_group"] in (63, 64, 65) and res["power_at_n"] == pytest.approx(res["simulated_power"], abs=0.025)
    assert cp._power(0.5, 64, 0.05) == pytest.approx(0.80, abs=0.01) and cp._power(0.8, 26, 0.05) == pytest.approx(0.80, abs=0.015)
    half = cp.run("power", effect=0.25)["needed_per_group"]
    assert 3.6 < half / res["needed_per_group"] < 4.4                                # la mitad de efecto, cuatro veces la gente


def test_confidence_intervals_cover_at_the_promised_rate_except_for_tiny_skewed_samples():
    fine = cp.run("confidence_intervals", population="uniform", n=30, level=0.95)
    assert fine["long_run_coverage"] == pytest.approx(0.95, abs=0.015) and fine["covered"] + fine["missed"] == 100 and not fine["notes"]
    wide = cp.run("confidence_intervals", population="uniform", n=30, level=0.99)
    assert wide["long_run_coverage"] > fine["long_run_coverage"]
    bad = cp.run("confidence_intervals", population="lognormal", n=5)
    assert bad["long_run_coverage"] < 0.90 and any("below the 95%" in n for n in bad["notes"])


def test_overfitting_has_a_best_complexity_and_training_error_always_falls():
    res = cp.run("overfitting")
    t = res["table"]
    assert 2 <= res["best_degree"] <= 6 and t["training error"].iloc[0] > t["training error"].iloc[res["best_degree"] - 1] > t["training error"].iloc[-1]
    assert t["error on new data"].iloc[-1] > 5 * t["error on new data"].min() and t["variance"].iloc[-1] > t["variance"].iloc[0] and t["bias squared"].iloc[0] > t["bias squared"].iloc[3]
    assert any("overfitting" in n for n in res["notes"]) and len(res["curves_figure"].data) >= 7
    big = cp.run("overfitting", n=400, max_degree=8, noise=0.1, reps=60)
    assert big["table"]["error on new data"].iloc[-1] < 3 * big["table"]["error on new data"].min()
    with pytest.raises(ValueError, match="at most n - 2"):
        cp.overfitting(max_degree=12, n=10)


def test_simpsons_paradox_reverses_the_trend_only_when_the_groups_differ_enough():
    res = cp.run("simpson", strength=1.0)
    assert res["reversed"] and res["overall_slope"] < 0 and all(s > 0.5 for s in res["within_slopes"]) and "opposite" in res["summary"]
    mild = cp.run("simpson", strength=0.1)
    assert not mild["reversed"] and mild["overall_slope"] > 0 and len(res["table"]) == 4


def test_regression_to_the_mean_follows_the_reliability():
    pure_luck = cp.run("regression_to_mean", reliability=0.0)
    pure_skill = cp.run("regression_to_mean", reliability=1.0)
    middle = cp.run("regression_to_mean", reliability=0.5)
    assert abs(pure_luck["second_year_mean"]) < 0.1 and pure_skill["share_kept"] == pytest.approx(1.0, abs=0.001)
    assert middle["second_year_mean"] == pytest.approx(middle["expected_second_year"], abs=0.1) and 0.3 < middle["share_kept"] < 0.7


def test_multiple_testing_matches_the_theory_and_corrections_work():
    res = cp.run("multiple_testing", tests=20, real_effects=0, studies=2000)
    t = res["table"].set_index("method")
    assert t.loc["No correction", "studies with at least one false alarm"] == pytest.approx(res["expected_any_false"], abs=0.04) and res["expected_any_false"] == pytest.approx(0.642, abs=0.001)
    assert t.loc["Bonferroni", "studies with at least one false alarm"] < 0.08 and t.loc["Benjamini-Hochberg", "studies with at least one false alarm"] < 0.08
    real = cp.run("multiple_testing", tests=20, real_effects=4, studies=1000).get("table").set_index("method")
    assert real.loc["Benjamini-Hochberg", "real effects found"] > real.loc["Bonferroni", "real effects found"] > 0.2
    assert real.loc["No correction", "real effects found"] > real.loc["Benjamini-Hochberg", "real effects found"] and real.loc["Benjamini-Hochberg", "share of discoveries that are false"] < 0.15


def test_inputs_are_checked_with_clear_messages():
    for name, kwargs, fragment in (("nope", {}, "Unknown concept"), ("clt", {"colour": 1}, "does not take"), ("clt", {"n": 0}, "between"), ("clt", {"distribution": "weird"}, "must be one of"),
                                   ("clt", {"n": "many"}, "must be a number"), ("p_values", {"alpha": 0.9}, "between"), ("clt", {"n": float("nan")}, "between")):
        with pytest.raises(ValueError, match=fragment):
            cp.run(name, **kwargs)
    assert cp.run("clt", n="40")["params"]["n"] == 40                                 # los números escritos como texto (terminal) se aceptan


def test_python_api_and_terminal(tmp_path):
    import kikolens as kl
    from typer.testing import CliRunner

    from kikolens.cli.main import app
    assert set(kl.concepts()["concept"]) == set(cp.CONCEPTS) and kl.concept("power", effect=0.5)["kind"] == "concept"
    assert kl.concept("clt", n=50)["observed_sd"] > 0
    runner = CliRunner()
    out = runner.invoke(app, ["learn"])
    assert out.exit_code == 0 and "clt" in out.output and "multiple_testing" in out.output
    html = tmp_path / "fig.html"
    out = runner.invoke(app, ["learn", "power", "--set", "effect=0.5", "--figure", str(html)])
    text = " ".join(out.output.split())
    assert out.exit_code == 0 and "participants per group" in text and "Lesson" in text and "Try" in text and html.exists() and html.stat().st_size > 5000
    assert runner.invoke(app, ["learn", "clt", "--set", "n=0"]).exit_code == 1 and runner.invoke(app, ["learn", "ghost"]).exit_code == 1
    assert runner.invoke(app, ["learn", "clt", "--set", "oops"]).exit_code == 1
