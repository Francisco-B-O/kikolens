import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import simulation as sim


def test_monte_carlo_matches_the_known_answer_of_a_sum_of_normals():
    inputs = {"a": {"dist": "normal", "mean": 100, "sd": 10}, "b": {"dist": "normal", "mean": 50, "sd": 5}}
    res = sim.monte_carlo(inputs, "a + b", n=40000, threshold=150)
    assert res["mean"] == pytest.approx(150, abs=0.5) and res["sd"] == pytest.approx(np.sqrt(125), rel=0.03)
    assert res["p_above"] == pytest.approx(0.5, abs=0.02) and res["percentiles"]["p5"] < res["percentiles"]["p95"]
    assert res["sensitivity"].iloc[0]["input"] == "a" and res["sensitivity"]["share of variance"].sum() == pytest.approx(1)
    assert res["value_at_risk"] == pytest.approx(res["percentiles"]["p5"]) and res["expected_shortfall"] <= res["value_at_risk"]
    assert len(res["figure"].data) == 1 and "result" in res["samples"].columns


def test_positive_correlation_widens_the_spread_of_a_sum():
    inputs = {"a": {"dist": "normal", "mean": 0, "sd": 1}, "b": {"dist": "normal", "mean": 0, "sd": 1}}
    apart = sim.monte_carlo(inputs, "a + b", n=20000, correlations={("a", "b"): 0.0})["sd"]
    together = sim.monte_carlo(inputs, "a + b", n=20000, correlations={("a", "b"): 0.8})["sd"]
    assert apart == pytest.approx(np.sqrt(2), rel=0.05) and together == pytest.approx(np.sqrt(2 + 2 * 0.8), rel=0.05)


def test_every_distribution_has_the_right_centre_and_an_impossible_correlation_is_repaired():
    specs = {"n": {"dist": "normal", "mean": 5, "sd": 1}, "u": {"dist": "uniform", "low": 0, "high": 10}, "t": {"dist": "triangular", "low": 0, "mode": 3, "high": 9},
             "l": {"dist": "lognormal", "mean": 4, "sd": 2}, "b": {"dist": "beta", "a": 2, "b": 2}, "p": {"dist": "poisson", "mean": 3}, "f": {"dist": "fixed", "value": 7},
             "e": {"dist": "empirical", "values": [1, 2, 3, 4, 5]}}
    centres = {"n": 5, "u": 5, "t": 4, "l": 4, "b": 0.5, "p": 3, "f": 7, "e": 3}
    samples = sim.draw(specs, 20000)
    for name, centre in centres.items():
        assert samples[name].mean() == pytest.approx(centre, rel=0.05), name
    impossible = {("n", "u"): 0.95, ("u", "t"): 0.95, ("n", "t"): -0.95}
    assert len(sim.draw({k: specs[k] for k in "nut"}, 1000, impossible)) == 1000


def test_inputs_from_data_keep_the_observed_correlation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=800)
    df = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(scale=0.5, size=800), "label": "k"})
    inputs, corr = sim.inputs_from_data(df)
    assert set(inputs) == {"x", "y"}
    draws = sim.draw(inputs, 5000, corr)
    assert draws.corr(method="spearman").loc["x", "y"] == pytest.approx(corr.loc["x", "y"], abs=0.05)
    with pytest.raises(ValueError):
        sim.inputs_from_data(df, ["label"])


@pytest.mark.parametrize("inputs,formula,fragment", [
    ({}, "a", "at least one input"),
    ({"a": {"dist": "weird"}}, "a", "unknown distribution"),
    ({"a": {"dist": "normal", "mean": 0}}, "a", "needs: sd"),
    ({"a": {"dist": "uniform", "low": 5, "high": 1}}, "a", "low is above high"),
    ({"a": {"dist": "empirical", "values": [1, 2]}}, "a", "at least 5"),
    ({"a": {"dist": "fixed", "value": 1}}, "zzz + 1", "(?i)unknown"),
    ({"a": {"dist": "fixed", "value": 1}}, "a / 0", "no valid number"),
])
def test_bad_monte_carlo_requests_are_explained(inputs, formula, fragment):
    with pytest.raises(ValueError, match=fragment):
        sim.monte_carlo(inputs, formula, n=500)


def test_the_number_of_simulations_is_bounded():
    with pytest.raises(ValueError, match="between"):
        sim.draw({"a": {"dist": "fixed", "value": 1}}, 10)


def test_linear_program_finds_the_textbook_optimum_and_the_shadow_prices():
    # max 3x + 5y  s.t.  x <= 4, 2y <= 12, 3x + 2y <= 18  -> x=2, y=6, valor 36
    res = sim.linear_program({"x": 3, "y": 5}, [{"coefs": {"x": 1}, "op": "<=", "rhs": 4, "name": "plant 1"}, {"coefs": {"y": 2}, "op": "<=", "rhs": 12, "name": "plant 2"},
                                                {"coefs": {"x": 3, "y": 2}, "op": "<=", "rhs": 18, "name": "plant 3"}])
    assert res["objective"] == pytest.approx(36) and res["solution"]["x"] == pytest.approx(2) and res["solution"]["y"] == pytest.approx(6)
    table = res["constraints"].set_index("constraint")
    assert not table.loc["plant 1", "binding"] and table.loc["plant 2", "binding"] and table.loc["plant 3", "binding"]
    assert table.loc["plant 2", "worth of one more unit"] == pytest.approx(1.5) and table.loc["plant 3", "worth of one more unit"] == pytest.approx(1)
    assert "plant 2" in res["summary"]


def test_integer_variables_and_minimisation():
    # max 5x + 4y  s.t. 6x + 4y <= 24, x + 2y <= 6: continuo (3, 1.5), entero (4, 0) -> 20 o (3, 1) -> 19
    cons = [{"coefs": {"x": 6, "y": 4}, "op": "<=", "rhs": 24}, {"coefs": {"x": 1, "y": 2}, "op": "<=", "rhs": 6}]
    assert sim.linear_program({"x": 5, "y": 4}, cons)["objective"] == pytest.approx(21)
    whole = sim.linear_program({"x": 5, "y": 4}, cons, integer=["x", "y"])
    assert whole["objective"] == pytest.approx(20) and (whole["solution"] % 1 == 0).all()
    low = sim.linear_program({"x": 2, "y": 3}, [{"coefs": {"x": 1, "y": 1}, "op": ">=", "rhs": 10}, {"coefs": {"x": 1}, "op": "==", "rhs": 4}], maximize=False)
    assert low["objective"] == pytest.approx(26) and low["solution"]["y"] == pytest.approx(6)


def test_infeasible_unbounded_and_malformed_programs_are_reported():
    with pytest.raises(ValueError, match="no solution"):
        sim.linear_program({"x": 1}, [{"coefs": {"x": 1}, "op": "<=", "rhs": 1}, {"coefs": {"x": 1}, "op": ">=", "rhs": 5}])
    with pytest.raises(ValueError, match="unbounded"):
        sim.linear_program({"x": 1}, [])
    with pytest.raises(ValueError, match="operator"):
        sim.linear_program({"x": 1}, [{"coefs": {"x": 1}, "op": "<", "rhs": 1}])
    with pytest.raises(ValueError, match="unknown variable"):
        sim.linear_program({"x": 1}, [{"coefs": {"q": 1}, "op": "<=", "rhs": 1}])
    with pytest.raises(ValueError, match="minimum is above"):
        sim.linear_program({"x": 1}, [], bounds={"x": (5, 1)})


def test_optimize_expression_handles_curves_constraints_and_bounds():
    res = sim.optimize_expression("-(x - 3) ** 2 - (y + 1) ** 2", {"x": (-10, 10), "y": (-10, 10)})
    assert res["solution"]["x"] == pytest.approx(3, abs=1e-3) and res["solution"]["y"] == pytest.approx(-1, abs=1e-3) and not res["at_bound"]
    tied = sim.optimize_expression("x * y", {"x": (0, 100), "y": (0, 100)}, ["x + y <= 20"])
    assert tied["solution"]["x"] == pytest.approx(10, abs=0.05) and tied["objective"] == pytest.approx(100, abs=0.5)
    edge = sim.optimize_expression("x", {"x": (0, 5)})
    assert edge["at_bound"] == ["x"] and edge["notes"] and sim.optimize_expression("x", {"x": (0, 5)}, maximize=False)["objective"] == pytest.approx(0, abs=1e-6)
    with pytest.raises(ValueError, match="Cannot read"):
        sim.optimize_expression("x", {"x": (0, 5)}, ["nonsense"])
    with pytest.raises(ValueError, match="No point"):
        sim.optimize_expression("x", {"x": (0, 5)}, ["x >= 9"])
    with pytest.raises(ValueError):
        sim.optimize_expression("import_os(x)", {"x": (0, 5)})


def test_shadow_prices_have_the_right_sign_for_minimum_and_equality_constraints():
    res = sim.linear_program({"x": 2, "y": 3}, [{"coefs": {"x": 1, "y": 1}, "op": ">=", "rhs": 10, "name": "need"}, {"coefs": {"x": 1}, "op": "==", "rhs": 4, "name": "fix"}],
                             maximize=False)
    table = res["constraints"].set_index("constraint")["worth of one more unit"]
    assert table["need"] == pytest.approx(3) and table["fix"] == pytest.approx(-1)


def test_distributions_and_variables_are_read_from_text():
    assert sim.parse_input("price=normal(10, 2)") == ("price", {"dist": "normal", "mean": 10.0, "sd": 2.0})
    assert sim.parse_input("t=triangular(1,2,3)")[1] == {"dist": "triangular", "low": 1.0, "mode": 2.0, "high": 3.0}
    assert sim.parse_input("n=poisson(4)")[1] == {"dist": "poisson", "mean": 4.0} and sim.parse_input("c=fixed(7)")[1]["value"] == 7.0
    assert sim.parse_variable("x=0:10") == ("x", (0.0, 10.0))
    for bad in ("nothing", "a=normal(1)", "a=cauchy(1,2)", "a=normal(x,2)"):
        with pytest.raises(ValueError):
            sim.parse_input(bad)
    with pytest.raises(ValueError):
        sim.parse_variable("x=5")


def test_linear_text_is_read_and_solved():
    assert sim.parse_linear("3x + 5*y - 2z + y") == {"x": 3.0, "y": 6.0, "z": -2.0} and sim.parse_linear("-x") == {"x": -1.0}
    con = sim.parse_linear_constraint("x + 2*y <= 12")
    assert con == {"coefs": {"x": 1.0, "y": 2.0}, "op": "<=", "rhs": 12.0, "name": "x + 2*y <= 12"}
    for bad in ("", "x*y", "x**2", "3 + x", "x / 2", "x +"):
        with pytest.raises(ValueError):
            sim.parse_linear(bad)
    for bad in ("x + y", "x <= y", "x <= "):
        with pytest.raises(ValueError):
            sim.parse_linear_constraint(bad)
    res = sim.linear_program(sim.parse_linear("3x + 5y"), [sim.parse_linear_constraint(c) for c in ("x <= 4", "2y <= 12", "3x + 2y <= 18")])
    assert res["objective"] == pytest.approx(36)


def test_the_python_api_and_the_terminal_run_simulations_and_optimisations(tmp_path):
    import kikolens as kl
    from typer.testing import CliRunner
    from kikolens.cli.main import app
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"price": rng.normal(10, 1, 300), "units": rng.normal(100, 10, 300)})
    res = kl.simulate(df, "price * units", n=2000)
    assert res["kind"] == "simulation" and abs(res["mean"] - 1000) < 60 and isinstance(res["table"], pd.DataFrame)
    assert kl.optimize("-(x-2)**2", {"x": (0, 5)})["solution"]["x"] == pytest.approx(2, abs=1e-3)
    assert kl.optimize({"x": 1, "y": 1}, {"x": (0, 3)}, [{"coefs": {"x": 1, "y": 1}, "op": "<=", "rhs": 5}])["objective"] == pytest.approx(5)
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    runner = CliRunner()
    out = runner.invoke(app, ["simulate", "price * units", "--data", str(csv), "--input", "price=fixed(10)", "--n", "1000", "--output", str(tmp_path / "s.csv")])
    assert out.exit_code == 0 and "averages" in out.output and (tmp_path / "s.csv").exists()
    assert runner.invoke(app, ["simulate", "a", "--input", "a=normal(1)"]).exit_code == 1
    out = runner.invoke(app, ["optimize", "x", "--var", "x=0:5", "--minimize"])
    assert out.exit_code == 0 and "lowest" in out.output
    problem = tmp_path / "p.json"
    problem.write_text('{"objective": {"x": 3, "y": 5}, "constraints": [{"coefs": {"x": 1}, "op": "<=", "rhs": 4}, {"coefs": {"y": 2}, "op": "<=", "rhs": 12}, {"coefs": {"x": 3, "y": 2}, "op": "<=", "rhs": 18}]}')
    out = runner.invoke(app, ["lp", str(problem)])
    assert out.exit_code == 0 and "36" in out.output
