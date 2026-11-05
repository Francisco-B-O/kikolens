import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import marketing as mk


def weekly(n=156, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    tv = rng.gamma(4, 250, n) * (1 + 0.4 * np.sin(t / 9))
    search = rng.gamma(4, 100, n)
    social = rng.gamma(3, 60, n)
    sales = (2000 + 4000 * mk.saturation(mk.adstock(tv, 0.5), 1500) + 2500 * mk.saturation(mk.adstock(search, 0.2), 400)
             + 600 * np.sin(2 * np.pi * t / 52.18) + 3 * t + rng.normal(0, 120, n))
    return pd.DataFrame({"date": pd.date_range("2022-01-03", periods=n, freq="W-MON"), "sales": sales, "tv": tv, "search": search, "social": social})


def test_adstock_and_saturation_behave_as_defined():
    assert mk.adstock([10, 0, 0], 0.5).tolist() == [10, 5, 2.5] and mk.adstock([1, 2], 0).tolist() == [1, 2]
    s = mk.saturation([0, 100, 1e9], half=100)
    assert s[0] == 0 and s[1] == pytest.approx(0.5) and s[2] == pytest.approx(1, abs=1e-6)
    assert mk.saturation(50, 100, 2) < mk.saturation(50, 100, 1)
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError):
            mk.adstock([1], bad)
    with pytest.raises(ValueError):
        mk.saturation([1], 0)


def test_the_model_finds_the_channels_that_work_and_ignores_the_one_that_does_not():
    res = mk.marketing_mix(weekly(), "date", "sales", ["tv", "search", "social"])
    table = res["table"].set_index("channel")
    assert res["r2"] > 0.9 and res["holdout_error"] < 0.1
    assert table.loc["social", "contribution"] < 0.25 * table.loc["tv", "contribution"] and not table.loc["social", "distinguishable from zero"]
    assert table.loc["tv", "distinguishable from zero"] and table.loc["search", "distinguishable from zero"] and any("social" in n for n in res["notes"])
    assert table.loc["tv", "share of outcome"] > 0.25 and table.loc["search", "share of outcome"] > 0.08
    assert abs(res["params"]["tv"]["decay"] - 0.5) <= 0.2 and abs(res["params"]["search"]["decay"] - 0.2) <= 0.2
    assert all(v >= 0 for v in res["coefficients"].values())
    assert res["parts"].sum(axis=1).sub(res["fitted"]).abs().max() < 1e-6
    assert len(res["curves_figure"].data) == 6 and "explains" in res["summary"]


def test_controls_are_used_and_collinear_channels_are_flagged():
    df = weekly()
    df["rain"] = np.random.default_rng(1).normal(size=len(df))
    df["sales"] += 150 * df["rain"]
    res = mk.marketing_mix(df, "date", "sales", ["tv", "search"], controls=["rain"])
    assert "controls" in res["parts"].columns and res["parts"]["controls"].std() > 50
    df["twin"] = df["tv"] * 1.01 + 1
    flagged = mk.marketing_mix(df, "date", "sales", ["tv", "twin"])
    assert any("move together" in n for n in flagged["notes"])


def test_budget_reallocation_gains_without_changing_the_total_and_respects_bounds():
    model = mk.marketing_mix(weekly(), "date", "sales", ["tv", "search", "social"])
    res = mk.optimize_budget(model)
    t = res["table"]
    assert t["proposed"].sum() == pytest.approx(t["scaled today"].sum(), rel=1e-6) and res["gain"] >= -1e-6
    assert (t["proposed"] >= t["today"] * 0.5 - 1e-6).all() and (t["proposed"] <= t["today"] * 2 + 1e-6).all()
    more = mk.optimize_budget(model, total=sum(v["current"] for v in model["_steady"].values()) * 1.5)
    assert more["outcome_proposed"] > res["outcome_proposed"]
    with pytest.raises(ValueError, match="cannot be split"):
        mk.optimize_budget(model, total=1e9)
    with pytest.raises(ValueError):
        mk.optimize_budget(model, low=2)
    with pytest.raises(ValueError, match="Pass the result"):
        mk.optimize_budget({})


@pytest.mark.parametrize("change,fragment", [
    (lambda d: d.head(10), "At least 30 periods"),
    (lambda d: d.assign(tv=0.0), "no spend"),
    (lambda d: d.assign(tv=-d["tv"]), "negative"),
    (lambda d: d.assign(sales="x"), "numeric"),
])
def test_unusable_marketing_data_is_explained(change, fragment):
    with pytest.raises(ValueError, match=fragment):
        mk.marketing_mix(change(weekly()), "date", "sales", ["tv"])
    with pytest.raises(ValueError, match="at least one channel"):
        mk.marketing_mix(weekly(), "date", "sales", [])


def demand(n=3000, seed=0, elasticity=-1.5):
    """El precio sube cuando la demanda es fuerte (sesgo) y el coste del proveedor lo mueve sin tocar la demanda (instrumento)."""
    rng = np.random.default_rng(seed)
    cost = rng.normal(0, 0.3, n)
    shock = rng.normal(0, 0.4, n)
    price = np.exp(2 + 0.6 * cost + 0.5 * shock + rng.normal(0, 0.05, n))
    quantity = np.exp(6 + elasticity * (np.log(price) - 2) + shock + rng.normal(0, 0.1, n))
    return pd.DataFrame({"price": price, "quantity": quantity, "cost": np.exp(cost), "segment": rng.choice(["a", "b"], n), "week": rng.normal(size=n)})


def test_elasticity_is_recovered_when_price_is_exogenous_and_the_curve_finds_the_optimum():
    rng = np.random.default_rng(3)
    price = rng.uniform(5, 15, 1500)
    df = pd.DataFrame({"price": price, "quantity": 500 * price ** -1.8 * np.exp(rng.normal(0, 0.1, 1500))})
    res = mk.price_elasticity(df, "quantity", "price", unit_cost=4.0)
    assert res["elasticity"] == pytest.approx(-1.8, abs=0.1) and res["interval"][0] < -1.8 < res["interval"][1]
    assert res["optimal_price"] == pytest.approx(4.0 * -1.8 / (1 - 1.8), rel=0.05) and len(res["curve"]) == 61 and "elastic" in res["summary"]


def test_an_instrument_removes_the_bias_of_prices_set_by_demand():
    df = demand()
    naive = mk.price_elasticity(df, "quantity", "price")
    iv = mk.price_elasticity(df, "quantity", "price", instrument=["cost"])
    assert naive["elasticity"] > -1.0 and any("range" in n or "positive" in n or "No controls" in n for n in naive["notes"])
    assert iv["elasticity"] == pytest.approx(-1.5, abs=0.25) and iv["method"] == "2sls" and iv["first_stage_f"] > 10
    assert abs(iv["ols"] - naive["elasticity"]) < 1e-9


def test_elasticity_by_group_and_bad_data():
    df = demand(1200)
    res = mk.price_elasticity(df, "quantity", "price", controls=["week"], group="segment", instrument=["cost"])
    assert set(res["groups"]["segment"]) == {"a", "b"}
    with pytest.raises(ValueError, match="barely changes"):
        mk.price_elasticity(df.assign(price=5.0), "quantity", "price")
    with pytest.raises(ValueError, match="At least 30"):
        mk.price_elasticity(df.head(10), "quantity", "price")
    with pytest.raises(ValueError, match="not found"):
        mk.price_elasticity(df, "quantity", "nope")
    with pytest.raises(ValueError, match="numeric"):
        mk.price_elasticity(df, "segment", "price")


def test_python_api_and_terminal(tmp_path):
    import kikolens as kl
    from typer.testing import CliRunner
    from kikolens.cli.main import app
    df = weekly()
    model = kl.marketing_mix(df, "date", "sales", ["tv", "search"])
    assert model["kind"] == "marketing mix" and isinstance(model["table"], pd.DataFrame)
    assert kl.budget(model)["kind"] == "budget" and kl.elasticity(demand(600), "quantity", "price")["kind"] == "price elasticity"
    csv, dem = tmp_path / "m.csv", tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    demand(600).to_csv(dem, index=False)
    runner = CliRunner()
    out = runner.invoke(app, ["mmm", str(csv), "date", "sales", "--channels", "tv,search", "--reallocate"])
    assert out.exit_code == 0 and "By channel" in out.output and "Proposed spend" in out.output
    out = runner.invoke(app, ["elasticity", str(dem), "quantity", "price", "--instrument", "cost", "--group", "segment", "--unit-cost", "5"])
    assert out.exit_code == 0 and "1% higher price" in out.output
    assert runner.invoke(app, ["mmm", str(csv), "date", "sales", "--channels", "nope"]).exit_code == 1
