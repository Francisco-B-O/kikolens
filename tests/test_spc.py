import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import spc


def frame(x, **cols):
    return pd.DataFrame({"v": x, **cols})


# ── Constantes y reglas ──────────────────────────────────────────────────────
def test_the_published_constants_are_reproduced():
    assert spc.c4(5) == pytest.approx(0.9400, abs=1e-4) and spc.c4(2) == pytest.approx(0.7979, abs=1e-4) and spc.c4(10) == pytest.approx(0.9727, abs=1e-4)
    assert 3 / (spc.c4(5) * np.sqrt(5)) == pytest.approx(1.427, abs=1e-3)                           # A3 de las tablas para subgrupos de 5
    assert np.sqrt(1 - spc.c4(5) ** 2) == pytest.approx(0.3412, abs=1e-3)                          # c5
    assert spc.D2 == pytest.approx(1.128, abs=1e-3)


def test_each_rule_fires_on_its_pattern_and_only_there():
    flat = np.zeros(30)
    for rule, series, expected_at in (
            (1, np.r_[flat, 3.5], 30), (2, np.r_[flat, [2.5, 0, 2.5]], 32), (3, np.r_[flat, [1.5, 1.5, 0, 1.5, 1.5]], 34),
            (4, np.r_[flat, np.full(9, 0.3)], 38), (5, np.r_[flat, np.arange(6) * 0.2], 35), (6, np.r_[flat, [1, -1] * 8], 45)):
        v = spc.rule_violations(series, sustained=9)
        assert rule in set(v["rule"]), rule
        assert expected_at in set(v.loc[v["rule"] == rule, "position"]), (rule, v)
    assert spc.rule_violations(np.random.default_rng(0).normal(size=5) * 0.1).empty


def test_false_alarm_rates_match_what_the_notes_promise():
    z = np.random.default_rng(1).normal(size=60000)
    v = spc.rule_violations(z, 9)
    for rule, expected in spc.FALSE_ALARM.items():
        assert (v["rule"] == rule).sum() / len(z) == pytest.approx(expected, abs=0.0012), rule
    assert (v["rule"] == 4).sum() / len(z) == pytest.approx(2 * 0.5 ** 9, abs=0.0015)
    default = v[v["rule"].isin(spc.DEFAULT_RULES)]["position"].nunique() / len(z)
    assert default < 0.008


# ── Cartas ───────────────────────────────────────────────────────────────────
def test_individuals_chart_limits_come_from_the_moving_range():
    x = np.random.default_rng(2).normal(10, 2, 2000)
    res = spc.control_chart(frame(x), "v", rules=(1,))
    assert res["centre"] == pytest.approx(10, abs=0.2) and res["sigma"] == pytest.approx(2.0, abs=0.1)
    t = res["table"]
    assert (t["upper limit"] - t["centre"]).iloc[0] == pytest.approx(3 * res["sigma"]) and t["signal"].mean() < 0.006 and len(res["figure"].data) == 4


def test_a_shift_after_the_baseline_is_signalled_at_the_right_point():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(10, 1, 60), rng.normal(13, 1, 20)]
    res = spc.control_chart(frame(x), "v", baseline=60)
    sig = res["signals"]
    watched = sig[sig["limit_phase"] == "monitoring"]
    first = int(watched["position"].min())
    assert 60 <= first <= 63 and not res["in_control"] and "NOT in control" in res["summary"]
    assert res["table"]["in baseline"].sum() <= 60 and res["table"]["centre"].nunique() == 1
    calm = spc.control_chart(frame(rng.normal(10, 1, 200)), "v", rules=(1,))
    assert calm["in_control"] or len(calm["signals"]) <= 2


def test_phase_one_cleaning_removes_special_causes_before_setting_limits():
    rng = np.random.default_rng(4)
    x = rng.normal(10, 1, 80)
    x[[10, 40]] = [25, -5]
    cleaned = spc.control_chart(frame(x), "v", clean_baseline=True)
    raw = spc.control_chart(frame(x), "v", clean_baseline=False)
    assert cleaned["removed"] == [10, 40] and cleaned["sigma"] < raw["sigma"] and any("set aside" in n for n in cleaned["notes"])
    assert cleaned["sigma"] == pytest.approx(1.0, abs=0.25)


def test_subgroup_means_chart_uses_the_within_subgroup_spread():
    rng = np.random.default_rng(5)
    n_groups, size = 60, 5
    d = pd.DataFrame({"batch": np.repeat(range(n_groups), size), "v": rng.normal(50, 4, n_groups * size)})
    res = spc.control_chart(d, "v", "xbar", subgroup="batch", rules=(1,))
    half = (res["table"]["upper limit"] - res["table"]["centre"]).iloc[0]
    assert res["centre"] == pytest.approx(50, abs=1.0) and half == pytest.approx(3 * 4 / np.sqrt(size), rel=0.2) and len(res["table"]) == 60
    shifted = d.assign(v=d["v"] + np.where(d["batch"] >= 50, 6, 0))
    assert not spc.control_chart(shifted, "v", "xbar", subgroup="batch", baseline=40)["in_control"]
    with pytest.raises(ValueError, match="identifies each subgroup"):
        spc.control_chart(d, "v", "xbar")


def test_attribute_charts_have_limits_that_follow_the_sample_size():
    rng = np.random.default_rng(6)
    n = rng.integers(80, 400, 50)
    d = pd.DataFrame({"bad": rng.binomial(n, 0.05), "n": n})
    p = spc.control_chart(d, "bad", "p", size="n", rules=(1,))
    t = p["table"]
    assert p["centre"] == pytest.approx(0.05, abs=0.01) and t["upper limit"].nunique() > 10 and (t["upper limit"][t["point"] == t["point"][np.argmin(n)]].iloc[0] > t["upper limit"][np.argmax(n)])
    assert (t["lower limit"] >= 0).all() and (t["upper limit"] <= 1).all()
    u = spc.control_chart(pd.DataFrame({"defects": rng.poisson(3 * n / 100), "units": n / 100}), "defects", "u", size="units", rules=(1,))
    assert u["centre"] == pytest.approx(3, abs=0.6)
    c = spc.control_chart(pd.DataFrame({"d": rng.poisson(9, 60)}), "d", "c", rules=(1,))
    assert c["centre"] == pytest.approx(9, abs=1.5) and (c["table"]["lower limit"] >= 0).all()
    for kwargs, fragment in (({"chart": "p"}, "how many were examined"), ({"chart": "z"}, "chart must be")):
        with pytest.raises(ValueError, match=fragment):
            spc.control_chart(d, "bad", **kwargs)
    with pytest.raises(ValueError, match="more defective"):
        spc.control_chart(d.assign(bad=d["n"] + 1), "bad", "p", size="n")


def test_chart_inputs_are_validated():
    d = frame(np.random.default_rng(7).normal(size=40))
    for kwargs, fragment in (({"rules": (9,)}, "rules"), ({"rules": ()}, "rules"), ({"sustained": 3}, "sustained"), ({"baseline": 5}, "baseline")):
        with pytest.raises(ValueError, match=fragment):
            spc.control_chart(d, "v", **kwargs)
    with pytest.raises(ValueError, match="At least"):
        spc.control_chart(d.head(5), "v")
    with pytest.raises(ValueError, match="not found"):
        spc.control_chart(d, "nope")
    with pytest.raises(ValueError, match="numeric"):
        spc.control_chart(d.assign(v="a"), "v")
    skewed = frame(np.random.default_rng(8).exponential(1, 300))
    assert any("skewed" in n for n in spc.control_chart(skewed, "v")["notes"])


# ── Capacidad ────────────────────────────────────────────────────────────────
def test_capability_indices_match_the_formulas():
    x = np.random.default_rng(9).normal(10, 1, 5000)
    res = spc.capability(x, 7, 13)
    assert res["cp"] == pytest.approx(1.0, abs=0.04) and res["cpk"] == pytest.approx(1.0, abs=0.04) and res["pp"] == pytest.approx(1.0, abs=0.04)
    assert res["ppm_expected"] == pytest.approx(2700, rel=0.2) and res["band"] in ("marginal", "not capable")
    off = spc.capability(np.random.default_rng(9).normal(11, 1, 5000), 7, 13)
    assert off["cpk"] == pytest.approx(2 / 3, abs=0.04) and off["cp"] == pytest.approx(1.0, abs=0.04) and any("off-centre" in n for n in off["notes"]) and off["band"] == "not capable"
    good = spc.capability(np.random.default_rng(9).normal(10, 0.5, 5000), 7, 13, target=10)
    assert good["cpk"] == pytest.approx(2.0, abs=0.1) and good["band"] == "capable" and good["cpm"] == pytest.approx(2.0, abs=0.1)
    one_sided = spc.capability(x, usl=13)
    assert one_sided["cp"] is None and one_sided["cpk"] == pytest.approx(1.0, abs=0.04)


def test_capability_warns_about_drift_and_non_normal_data_and_validates():
    rng = np.random.default_rng(10)
    drifting = rng.normal(10, 0.3, 600) + np.linspace(0, 3, 600)
    assert any("drifts over time" in n for n in spc.capability(drifting, 5, 16)["notes"])
    skewed = spc.capability(rng.exponential(1, 500) + 5, 0, 12)
    assert any("do not look normal" in n for n in skewed["notes"]) and skewed["observed_out"] >= 0
    for args, fragment in (((rng.normal(size=100),), "at least one"), ((rng.normal(size=100), 5, 3), "below"), ((rng.normal(size=10), 0, 1), "At least 30"), ((np.ones(50), 0, 2), "do not vary")):
        with pytest.raises(ValueError, match=fragment):
            spc.capability(*args)


# ── CUSUM y EWMA ─────────────────────────────────────────────────────────────
def test_cusum_and_ewma_catch_a_one_sigma_shift_far_sooner_than_shewhart():
    delays_cusum, delays_ewma, delays_shewhart, false_cusum = [], [], [], 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        x = np.r_[rng.normal(0, 1, 50), rng.normal(1, 1, 150)]
        c, e = spc.cusum(x, target=0, sigma=1), spc.ewma(x, target=0, sigma=1, width=2.962)
        if c["signal_at"] is not None and c["signal_at"] >= 50:
            delays_cusum.append(c["signal_at"] - 50)
        if e["signal_at"] is not None and e["signal_at"] >= 50:
            delays_ewma.append(e["signal_at"] - 50)
        hit = np.where(np.abs(x[50:]) > 3)[0]
        delays_shewhart.append(hit[0] if len(hit) else 150)
        false_cusum += spc.cusum(rng.normal(0, 1, 100), target=0, sigma=1)["signal_at"] is not None
    assert np.mean(delays_cusum) < 15 and np.mean(delays_ewma) < 15 and np.mean(delays_shewhart) > 3 * np.mean(delays_cusum)
    assert false_cusum / 40 < 0.25


def test_cusum_dates_the_start_of_the_shift_and_reports_its_direction():
    rng = np.random.default_rng(11)
    while True:                                                      # una semilla sin falsa alarma en el tramo base (la tasa de falsas alarmas del CUSUM no es cero)
        x = np.r_[rng.normal(5, 1, 80), rng.normal(3.5, 1, 40)]
        if spc.cusum(x[:80], target=5, sigma=1)["signal_at"] is None:
            break
    res = spc.cusum(x, target=5, sigma=1)
    assert res["signal_at"] is not None and 80 <= res["signal_at"] < 100 and abs(res["shift_started"] - 80) <= 6 and "down" in res["summary"] and res["notes"][0].endswith("down.")
    calm = next(c for c in (rng.normal(5, 1, 100) for _ in range(50)) if spc.cusum(c, target=5, sigma=1)["signal_at"] is None)
    assert "No sustained shift" in spc.cusum(calm, target=5, sigma=1)["summary"]
    for kwargs, fragment in (({"k": 0}, "positive"), ({"h": -1}, "positive")):
        with pytest.raises(ValueError, match=fragment):
            spc.cusum(x, **kwargs)
    with pytest.raises(ValueError, match="At least"):
        spc.cusum([1, 2, 3])


def test_ewma_limits_widen_from_the_start_and_validate_inputs():
    x = np.random.default_rng(12).normal(0, 1, 60)
    res = spc.ewma(x, target=0, sigma=1)
    width = res["upper"] - res["lower"]
    assert width[0] < width[-1] and width[-1] == pytest.approx(2 * 3 * np.sqrt(0.2 / 1.8), rel=0.02) and len(res["figure"].data) == 4
    for kwargs, fragment in (({"lam": 0}, "lam"), ({"lam": 1.5}, "lam"), ({"width": -1}, "lam")):
        with pytest.raises(ValueError, match=fragment):
            spc.ewma(x, **kwargs)
    with pytest.raises(ValueError, match="zero"):
        spc.ewma(np.ones(40))


def test_python_api_and_terminal(tmp_path):
    import kikolens as kl
    from typer.testing import CliRunner

    from kikolens.cli.main import app
    rng = np.random.default_rng(13)
    d = pd.DataFrame({"t": range(120), "v": np.r_[rng.normal(10, 1, 80), rng.normal(12.5, 1, 40)], "batch": np.repeat(range(24), 5), "bad": rng.binomial(100, 0.05, 120), "n": 100})
    assert kl.control_chart(d, "v", baseline=80)["kind"] == "control chart" and kl.control_chart(d, "bad", "p", size="n")["chart"] == "p"
    assert kl.process_capability(d, "v", lsl=5, usl=15)["kind"] == "capability" and kl.shift_detection(d, "v", baseline=80)["cusum"]["signal_at"] is not None
    csv = tmp_path / "p.csv"
    d.to_csv(csv, index=False)
    runner = CliRunner()
    for args, text in ((["control-chart", str(csv), "v", "--baseline", "80"], "NOT in control"), (["control-chart", str(csv), "bad", "--chart", "p", "--size", "n"], "Centre"),
                       (["capability", str(csv), "v", "--lsl", "5", "--usl", "15"], "Cpk"), (["shift-detect", str(csv), "v", "--baseline", "80"], "shift")):
        out = runner.invoke(app, args)
        assert out.exit_code == 0 and text in " ".join(out.output.split()), (args, out.output)
    assert runner.invoke(app, ["capability", str(csv), "v"]).exit_code == 1
