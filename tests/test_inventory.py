import math

import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import inventory as inv


# ── Vendedor de periódicos ───────────────────────────────────────────────────
def test_newsvendor_orders_the_critical_fractile_and_matches_a_simulation():
    even = inv.newsvendor(unit_cost=6, price=10, salvage=2, mean=100, sd=20)
    assert even["critical_ratio"] == pytest.approx(0.5) and even["quantity"] == pytest.approx(100.0)
    high = inv.newsvendor(4, 10, 2, 100, 20)
    assert high["critical_ratio"] == pytest.approx(0.75) and high["quantity"] == pytest.approx(100 + 0.6744897 * 20, abs=1e-4)
    d = np.maximum(np.random.default_rng(0).normal(100, 20, 2_000_000), 0)
    q = high["quantity"]
    assert high["expected_profit"] == pytest.approx(np.mean(10 * np.minimum(q, d) + 2 * np.maximum(q - d, 0) - 4 * q), abs=0.6)
    assert high["expected_short"] == pytest.approx(np.mean(np.maximum(d - q, 0)), abs=0.05) and high["expected_left_over"] == pytest.approx(np.mean(np.maximum(q - d, 0)), abs=0.1)
    best = high["table"].loc[high["table"]["expected profit"].idxmax(), "quantity"]
    assert best == pytest.approx(q, abs=3) and len(high["figure"].data) == 1


def test_newsvendor_with_observed_demand_and_cheaper_or_dearer_stock():
    rng = np.random.default_rng(1)
    past = rng.gamma(5, 20, 4000)
    res = inv.newsvendor(4, 10, 2, samples=past)
    assert res["distribution"] == "observed" and res["quantity"] == pytest.approx(np.quantile(past, 0.75), rel=1e-6) and res["probability_cover"] == pytest.approx(0.75, abs=0.01)
    cheap = inv.newsvendor(2, 10, 0, 100, 20)["quantity"]
    dear = inv.newsvendor(9, 10, 0, 100, 20)["quantity"]
    assert cheap > 100 > dear                                                    # lo barato se pide de más; lo caro, de menos
    assert any("negative demand" in n for n in inv.newsvendor(4, 10, 2, 100, 70)["notes"])
    for kwargs, fragment in (({"unit_cost": 10, "price": 10}, "above the unit cost"), ({"unit_cost": 4, "price": 10, "salvage": 5}, "salvage"), ({"unit_cost": 4, "price": 10}, "mean and standard"),
                             ({"unit_cost": -1, "price": 10, "mean": 1, "sd": 1}, "positive")):
        with pytest.raises(ValueError, match=fragment):
            inv.newsvendor(**kwargs)
    with pytest.raises(ValueError, match="At least 10"):
        inv.newsvendor(4, 10, 2, samples=[1, 2, 3])


# ── EOQ ──────────────────────────────────────────────────────────────────────
def test_eoq_matches_the_formula_and_is_flat_near_the_optimum():
    res = inv.eoq(1000, 50, 2)
    assert res["quantity"] == pytest.approx(math.sqrt(2 * 1000 * 50 / 2)) and res["orders_per_year"] == pytest.approx(1000 / res["quantity"])
    assert res["ordering_cost"] == pytest.approx(res["holding_cost_total"]) and res["total_cost"] == pytest.approx(2 * res["ordering_cost"]) and res["total_cost"] == pytest.approx(math.sqrt(2 * 1000 * 50 * 2))
    cost = lambda q: 1000 / q * 50 + q / 2 * 2                                    # noqa: E731
    assert cost(res["quantity"] * 1.2) / cost(res["quantity"]) - 1 < 0.025 and cost(res["quantity"] * 0.8) / cost(res["quantity"]) - 1 < 0.03
    assert inv.eoq(1000, 50, 2, unit_cost=10)["total_cost"] == pytest.approx(res["total_cost"] + 10000)
    for kwargs in ({"annual_demand": 0, "order_cost": 1, "holding_cost": 1}, {"annual_demand": 1, "order_cost": -1, "holding_cost": 1}):
        with pytest.raises(ValueError, match="positive"):
            inv.eoq(**kwargs)


# ── Punto de pedido ──────────────────────────────────────────────────────────
def test_safety_stock_and_reorder_point_follow_the_formulas():
    res = inv.reorder_policy(10, 3, 4, 0, 0.95)
    assert res["safety_stock"] == pytest.approx(1.6448536 * 3 * 2, abs=1e-4) and res["reorder_point"] == pytest.approx(40 + res["safety_stock"])
    both = inv.reorder_policy(10, 3, 4, 1.0, 0.95)
    assert both["safety_stock"] == pytest.approx(1.6448536 * math.sqrt(4 * 9 + 100 * 1), abs=1e-4) and any("delivery time matters more" in n for n in both["notes"])
    assert inv.reorder_policy(10, 3, 4, 0, 0.99)["safety_stock"] > res["safety_stock"] > inv.reorder_policy(10, 3, 4, 0, 0.8)["safety_stock"] > 0
    z = 1.2
    sd = 6.0
    loss = math.exp(-z * z / 2) / math.sqrt(2 * math.pi) - z * (1 - 0.8849303)
    assert inv.reorder_policy(10, 3, 4, 0, 0.8849303, order_quantity=60)["fill_rate"] == pytest.approx(1 - sd * loss / 60, abs=2e-4)
    small = inv.reorder_policy(10, 9, 4, 0, 0.9, order_quantity=5)
    assert small["fill_rate"] < 0.9 and any("fill rate" in n for n in small["notes"])
    for kwargs, fragment in (({"service_level": 1.0}, "service level"), ({"service_level": 0.3}, "service level"), ({"demand_sd": -1}, "negative"), ({"lead_time": 0}, "positive")):
        args = {"demand_mean": 10, "demand_sd": 3, "lead_time": 4, **kwargs}
        with pytest.raises(ValueError, match=fragment):
            inv.reorder_policy(**args)


def test_the_simulator_is_exact_when_demand_is_certain():
    for rop, stockouts in ((36, True), (39, True), (41, False), (44, False)):
        sim = inv.simulate_policy(10, 0.0, 4, 60, rop, runs=10, days=120)
        assert (sim["cycle_service_level"] < 1.0) == stockouts, rop
    sim = inv.simulate_policy(10, 0.0, 4, 60, 44, runs=10, days=365)
    assert sim["orders_per_year"] == pytest.approx(10 * 365 / 60, abs=1.0) and sim["fill_rate"] == 1.0


def test_the_formula_is_confirmed_by_the_simulation():
    for lead_sd in (0.0, 1.0):
        rop = inv.reorder_policy(10, 3, 4, lead_sd, 0.95, order_quantity=60)
        sim = inv.simulate_policy(10, 3, 4, 60, rop["reorder_point"], lead_time_sd=lead_sd, runs=500)
        assert 0.90 <= sim["cycle_service_level"] <= 0.97, lead_sd                       # un poco por debajo del 95 %: el pedido sale en pasos discretos
        assert sim["fill_rate"] == pytest.approx(rop["fill_rate"], abs=0.01)
    lower = inv.simulate_policy(10, 3, 4, 60, 40, runs=500)["cycle_service_level"]
    higher = inv.simulate_policy(10, 3, 4, 60, 50, runs=500)["cycle_service_level"]
    assert 0.35 < lower < 0.55 and higher > lower + 0.3
    lost = inv.simulate_policy(10, 3, 4, 60, 30, runs=300, lost_sales=True)
    backorder = inv.simulate_policy(10, 3, 4, 60, 30, runs=300)
    assert 0.5 < backorder["fill_rate"] < lost["fill_rate"] < 1.0                  # con pedidos pendientes parte del lote que llega salda la deuda, no sirve demanda nueva


def test_simulation_inputs_are_checked():
    for kwargs, fragment in (({"days": 5}, "days"), ({"runs": 1}, "runs"), ({"lead_time": 0}, "at least 1"), ({"demand_sd": -1}, "negative")):
        args = {"demand_mean": 10, "demand_sd": 3, "lead_time": 4, "order_quantity": 60, "reorder_point": 50, **kwargs}
        with pytest.raises(ValueError, match=fragment):
            inv.simulate_policy(**args)
    with pytest.raises(ValueError, match="too small"):
        inv.simulate_policy(10, 3, 4, 0.01, 500, runs=10, days=30)


# ── Clasificación ────────────────────────────────────────────────────────────
def sales(seed=0, weeks=52):
    rng = np.random.default_rng(seed)
    rows = []
    specs = {"big_steady": (200, 0.1, 1.0), "big_wild": (150, 1.0, 1.0), "mid_steady": (40, 0.2, 1.0), "mid_steady2": (35, 0.25, 1.0), "small_a": (6, 0.3, 1.0),
             "small_b": (5, 0.3, 1.0), "tiny_rare": (30, 0.5, 0.15), "tiny_lumpy": (80, 1.5, 0.35), "tiny_c": (3, 0.4, 1.0), "tiny_d": (2, 0.4, 1.0)}
    for w in range(0, weeks, 3):                                                   # a rachas a propósito: cada tres semanas, y de tamaño muy desigual (20 o 200)
        q = 20.0 if (w // 3) % 2 == 0 else 200.0
        rows.append({"item": "tiny_lumpy", "week": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=w), "qty": q, "value": q * 2})
    specs.pop("tiny_lumpy")
    for name, (mean, cv, p_active) in specs.items():
        for w in range(weeks):
            if rng.random() < p_active:
                q = float(rng.lognormal(math.log(mean), math.sqrt(math.log(1 + cv ** 2)))) if cv >= 1 else max(0.0, rng.normal(mean, mean * cv))
                rows.append({"item": name, "week": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=w), "qty": q, "value": q * (10 if name.startswith(("big", "mid")) else 2)})
    return pd.DataFrame(rows)


def test_abc_xyz_and_demand_classes_separate_the_items():
    res = inv.classify_items(sales(), "item", "qty", "week", "value")
    t = res["table"].set_index("item")
    assert t.loc["big_steady", "ABC"] == "A" and t.loc["tiny_d", "ABC"] == "C" and t["cumulative share"].is_monotonic_increasing and t["cumulative share"].iloc[-1] == pytest.approx(1.0)
    assert t.loc["big_steady", "XYZ"] == "X" and t.loc["big_wild", "XYZ"] in ("Y", "Z") and t.loc["mid_steady", "XYZ"] == "X"
    assert t.loc["big_steady", "demand class"] == "smooth" and t.loc["big_wild", "demand class"] == "erratic" and t.loc["tiny_rare", "demand class"] in ("intermittent", "lumpy")
    assert t.loc["tiny_lumpy", "demand class"] == "lumpy" and t.loc["tiny_lumpy", "suggested approach"].startswith("Hard to plan")
    assert res["matrix"].to_numpy().sum() == 10 and list(res["matrix"].index) == ["A", "B", "C"] and res["a_items"] >= 1 and "class A" in res["summary"]
    assert (t.loc["big_wild", "XYZ"] != "Z") or any("A and Z" in n for n in res["notes"])


def test_syntetos_boylan_boundaries():
    assert inv.demand_class(1.0, 0.2) == "smooth" and inv.demand_class(1.0, 0.8) == "erratic"
    assert inv.demand_class(2.5, 0.2) == "intermittent" and inv.demand_class(2.5, 0.8) == "lumpy"
    assert inv.demand_class(1.32, 0.49) == "smooth" and inv.demand_class(1.33, 0.49) == "intermittent"


def test_classification_works_without_dates_and_validates_inputs():
    data = sales()
    plain = inv.classify_items(data, "item", "qty")
    assert plain["matrix"] is None and "XYZ" not in plain["table"].columns
    with pytest.raises(ValueError, match="not found"):
        inv.classify_items(data, "nope", "qty")
    with pytest.raises(ValueError, match="numeric"):
        inv.classify_items(data.assign(qty="a"), "item", "qty")
    with pytest.raises(ValueError, match="negative"):
        inv.classify_items(data.assign(qty=-1.0), "item", "qty")
    with pytest.raises(ValueError, match="At least 3 items"):
        inv.classify_items(data[data["item"].isin(["big_steady", "mid_steady"])], "item", "qty")
    with pytest.raises(ValueError, match="6 periods"):
        inv.classify_items(data[data["week"] < "2024-02-05"], "item", "qty", "week")


# ── Plan ─────────────────────────────────────────────────────────────────────
def test_plan_gives_each_item_a_policy_and_the_eoq_when_costs_are_known():
    rng = np.random.default_rng(2)
    days = pd.date_range("2024-01-01", periods=180)
    rows = [{"sku": sku, "day": d, "units": max(0, rng.normal(mean, sd))} for sku, (mean, sd) in {"a": (20, 4), "b": (5, 5), "c": (2, 0.5)}.items() for d in days]
    res = inv.plan(pd.DataFrame(rows), "sku", "day", "units", lead_time_days=5, service_level=0.95, order_cost=40, holding_cost=3)
    t = res["table"].set_index("item")
    assert t.loc["a", "reorder point"] > t.loc["b", "reorder point"] > t.loc["c", "reorder point"] and t.loc["a", "safety stock"] == pytest.approx(1.645 * 4 * math.sqrt(5), rel=0.15)
    assert t.loc["a", "order quantity (EOQ)"] == pytest.approx(math.sqrt(2 * 20 * 365 * 40 / 3), rel=0.05) and "orders per year" in t.columns and "95%" in res["summary"]
    weekly = inv.plan(pd.DataFrame(rows), "sku", "day", "units", lead_time_days=14, freq="W")
    assert "demand per week" in weekly["table"].columns and "order quantity (EOQ)" not in weekly["table"].columns
    for kwargs, fragment in (({"freq": "M"}, "freq"), ({"item": "nope"}, "not found")):
        args = {"item": "sku", "date": "day", "quantity": "units", "lead_time_days": 5, **kwargs}
        with pytest.raises(ValueError, match=fragment):
            inv.plan(pd.DataFrame(rows), **args)
    with pytest.raises(ValueError, match="14 periods"):
        inv.plan(pd.DataFrame(rows).head(10), "sku", "day", "units", 5)


def test_python_api_and_terminal(tmp_path):
    import kikolens as kl
    from typer.testing import CliRunner

    from kikolens.cli.main import app
    data = sales()
    assert kl.abc_xyz(data, "item", "qty", date="week", value="value")["kind"] == "abc xyz"
    assert kl.newsvendor(4, 10, 2, mean=100, sd=20)["quantity"] > 100 and kl.eoq(1000, 50, 2)["quantity"] == pytest.approx(223.6, abs=0.1)
    assert kl.reorder_policy(10, 3, 4)["kind"] == "reorder policy" and kl.simulate_stock_policy(10, 3, 4, 60, 50, runs=50)["kind"] == "stock simulation"
    daily = pd.DataFrame({"sku": np.repeat(["a", "b"], 60), "day": list(pd.date_range("2024-01-01", periods=60)) * 2, "units": np.random.default_rng(3).poisson(8, 120)})
    assert kl.inventory_plan(daily, "sku", "day", "units", lead_time_days=3)["kind"] == "inventory plan"
    csv, plan_csv = tmp_path / "s.csv", tmp_path / "d.csv"
    data.to_csv(csv, index=False)
    daily.to_csv(plan_csv, index=False)
    runner = CliRunner()
    for args, text in ((["abc-xyz", str(csv), "item", "qty", "--date", "week", "--value", "value"], "class A"), (["newsvendor", "--cost", "4", "--price", "10", "--salvage", "2", "--mean", "100",
                                                                                                                 "--sd", "20"], "Order"),
                       (["eoq", "--demand", "1000", "--order-cost", "50", "--holding-cost", "2"], "223"), (["reorder", "--mean", "10", "--sd", "3", "--lead-time", "4"], "Reorder when"),
                       (["stock-simulate", "--mean", "10", "--sd", "3", "--lead-time", "4", "--order-quantity", "60", "--reorder-point", "50", "--runs", "50"], "simulated years"),
                       (["inventory-plan", str(plan_csv), "sku", "day", "units", "--lead-time", "3"], "reorder policy")):
        out = runner.invoke(app, args)
        assert out.exit_code == 0 and text in " ".join(out.output.split()), (args, out.output)
    assert runner.invoke(app, ["eoq", "--demand", "0", "--order-cost", "50", "--holding-cost", "2"]).exit_code == 1
