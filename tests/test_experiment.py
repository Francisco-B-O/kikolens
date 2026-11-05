import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import experiment as ex


def trial(n=4000, effect=0.7, seed=0, rho_noise=5.0, binary=False, split=0.5):
    rng = np.random.default_rng(seed)
    pre = rng.normal(50, 10, n)
    g = np.where(rng.random(n) < split, "control", "new")
    latent = 0.8 * pre + rng.normal(0, rho_noise, n)
    if binary:
        y = (rng.random(n) < 1 / (1 + np.exp(-(latent - 40) / 8 + (-effect) * (g == "new")))).astype(int)
    else:
        y = latent + np.where(g == "new", effect, 0)
    return pd.DataFrame({"g": g, "y": y, "pre": pre, "day": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 56, n), "D"),
                         "device": rng.choice(["mobile", "desktop"], n)})


# ── Reparto ──────────────────────────────────────────────────────────────────
def test_srm_flags_a_broken_split_and_accepts_a_fair_one():
    broken = ex.srm(pd.DataFrame({"g": ["a"] * 5200 + ["b"] * 4800}), "g")
    assert broken["broken"] and broken["p_value"] == pytest.approx(6.3e-5, rel=0.02) and broken["chi2"] == pytest.approx(16.0) and broken["notes"] and "NOT what was planned" in broken["summary"]
    fair = ex.srm(pd.DataFrame({"g": ["a"] * 5030 + ["b"] * 4970}), "g")
    assert not fair["broken"] and fair["p_value"] > 0.5 and not fair["notes"] and len(fair["figure"].data) == 2
    slight = ex.srm(pd.DataFrame({"g": ["a"] * 5120 + ["b"] * 4880}), "g")
    assert not slight["broken"] and slight["notes"] and "slightly" in slight["notes"][0]


def test_srm_with_an_uneven_plan_and_more_than_two_groups():
    uneven = pd.DataFrame({"g": ["a"] * 9000 + ["b"] * 1000})
    assert not ex.srm(uneven, "g", {"a": 90, "b": 10})["broken"] and ex.srm(uneven, "g")["broken"]
    three = pd.DataFrame({"g": ["a"] * 3400 + ["b"] * 3300 + ["c"] * 3300})
    assert ex.srm(three, "g")["table"]["difference"].round(1).tolist() == [66.7, -33.3, -33.3]
    for expected, fragment in (({"a": 1}, "exactly the groups"), ({"a": -1, "b": 2}, "positive")):
        with pytest.raises(ValueError, match=fragment):
            ex.srm(uneven, "g", expected)
    with pytest.raises(ValueError, match="at least two"):
        ex.srm(pd.DataFrame({"g": ["a"] * 10}), "g")
    with pytest.raises(ValueError, match="not found"):
        ex.srm(uneven, "nope")


# ── CUPED ────────────────────────────────────────────────────────────────────
def test_cuped_keeps_the_effect_and_narrows_the_interval_as_theory_says():
    d = trial()
    res = ex.cuped(d, "g", "y", "pre", "control")
    row = res["table"].iloc[0]
    rho = res["correlation"]
    assert res["variance_reduction"] == pytest.approx(rho ** 2, abs=0.03) and rho > 0.8
    assert row["adjusted lift"] == pytest.approx(0.7, abs=0.35) and row["interval narrower by"] == pytest.approx(1 - np.sqrt(1 - rho ** 2), abs=0.05)
    assert row["adjusted low"] < 0.7 < row["adjusted high"] and row["adjusted p"] < row["raw p"] and row["pre-period balance p"] > 0.01 and "removes" in res["summary"]


def test_cuped_is_unbiased_and_covers_the_truth_across_simulations():
    covered, estimates, widths_raw, widths_adj = 0, [], [], []
    for seed in range(60):
        row = ex.cuped(trial(1500, 0.7, seed), "g", "y", "pre", "control")["table"].iloc[0]
        estimates.append(row["adjusted lift"])
        covered += row["adjusted low"] <= 0.7 <= row["adjusted high"]
        widths_raw.append(row["raw high"] - row["raw low"])
        widths_adj.append(row["adjusted high"] - row["adjusted low"])
    assert abs(np.mean(estimates) - 0.7) < 0.08 and 0.88 <= covered / 60 <= 1.0 and np.mean(widths_adj) < 0.6 * np.mean(widths_raw)


def test_a_useless_pre_column_removes_little_and_says_so():
    d = trial()
    d["pre"] = np.random.default_rng(3).normal(size=len(d))
    res = ex.cuped(d, "g", "y", "pre", "control")
    assert res["variance_reduction"] < 0.01 and any("only" in n for n in res["notes"])
    with pytest.raises(ValueError, match="constant"):
        ex.cuped(d.assign(pre=1.0), "g", "y", "pre", "control")
    with pytest.raises(ValueError, match="numeric"):
        ex.cuped(d.assign(pre="a"), "g", "y", "pre", "control")


# ── Tiempo y segmentos ───────────────────────────────────────────────────────
def test_a_novelty_effect_that_fades_is_detected():
    rng = np.random.default_rng(1)
    n = 12000
    day = rng.integers(0, 56, n)
    g = np.where(rng.random(n) < 0.5, "control", "new")
    y = rng.normal(10, 3, n) + np.where(g == "new", 2.0 * np.exp(-day / 14), 0)
    d = pd.DataFrame({"g": g, "y": y, "day": pd.Timestamp("2024-01-01") + pd.to_timedelta(day, "D")})
    res = ex.effect_over_time(d, "g", "y", "control", "day")
    assert res["trend_p"] < 0.01 and res["trend_slope"] < 0 and any("novelty" in n for n in res["notes"]) and res["table"]["lift"].iloc[0] > res["table"]["lift"].iloc[-1] + 0.8
    steady = ex.effect_over_time(trial(12000, 0.7, 4), "g", "y", "control", "day")
    assert steady["trend_p"] > 0.01 and not any("novelty" in n for n in steady["notes"])
    with pytest.raises(ValueError, match="At least 3 periods"):
        ex.effect_over_time(d, "g", "y", "control", "day", freq="Y")


def test_segment_effects_find_the_segment_that_really_differs_and_correct_for_looking():
    rng = np.random.default_rng(2)
    n = 8000
    d = pd.DataFrame({"g": np.where(rng.random(n) < 0.5, "control", "new"), "device": rng.choice(["mobile", "desktop"], n), "country": rng.choice(list("ABCDEFGH"), n)})
    d["y"] = rng.normal(10, 3, n) + np.where((d["g"] == "new") & (d["device"] == "mobile"), 1.5, 0)
    res = ex.segment_effects(d, "g", "y", "control", ["device", "country"])
    hits = res["table"][res["table"]["differs from the rest"]]
    assert set(hits["segment"]) == {"device"} and set(hits["value"]) == {"mobile", "desktop"}                 # un país no se distingue: la mitad de cada uno usa móvil
    mobile = res["table"].query("segment == 'device' and value == 'mobile'").iloc[0]
    desktop = res["table"].query("segment == 'device' and value == 'desktop'").iloc[0]
    assert mobile["lift"] == pytest.approx(1.5, abs=0.4) and abs(desktop["lift"]) < 0.4 and mobile["p adjusted"] >= mobile["p"] and mobile["gap"] > 1.0
    none = ex.segment_effects(d.assign(y=rng.normal(10, 3, n)), "g", "y", "control", ["country"])
    assert not none["table"]["differs from the rest"].any() and "chance alone" in none["notes"][0]
    with pytest.raises(ValueError, match="not found"):
        ex.segment_effects(d, "g", "y", "control", ["nope"])
    with pytest.raises(ValueError, match="No segment"):
        ex.segment_effects(d.head(100), "g", "y", "control", ["country"], min_rows=500)


# ── Análisis completo ────────────────────────────────────────────────────────
def test_the_full_analysis_gives_a_verdict_and_refuses_one_when_the_split_is_broken():
    good = ex.analyze_experiment(trial(), "g", "y", "control", pre="pre", date="day", segments=["device"])
    assert good["verdict"].startswith("new beats control") and good["cuped_theta"] > 0 and not good["srm"]["broken"] and {"cuped", "over_time", "segments"} <= set(good)
    assert good["table"].iloc[0]["low"] > 0 and len(good["figure"].data) == 1
    broken = ex.analyze_experiment(trial(split=0.53, n=20000), "g", "y", "control")
    assert broken["srm"]["broken"] and broken["verdict"].startswith("Do not decide") and any("not what was planned" in n for n in broken["notes"])
    flat = ex.analyze_experiment(trial(800, 0.0, 6), "g", "y", "control")
    assert "No clear difference" in flat["verdict"] and flat["table"].iloc[0]["smallest effect detectable"] > 0


def test_the_full_analysis_handles_conversions_and_several_variants():
    rng = np.random.default_rng(6)
    n = 30000
    g = rng.choice(["control", "b", "c"], n)
    rate = np.select([g == "b", g == "c"], [0.115, 0.10], 0.10)
    d = pd.DataFrame({"g": g, "y": (rng.random(n) < rate).astype(int)})
    res = ex.analyze_experiment(d, "g", "y", "control")
    table = res["table"].set_index("group")
    assert res["binary"] and table.loc["b", "lift"] == pytest.approx(0.015, abs=0.01) and table.loc["b", "p adjusted"] < 0.05 < table.loc["c", "p adjusted"]
    assert "p adjusted" in res["table"].columns and table.loc["b", "p adjusted"] >= table.loc["b", "p"]
    assert res["verdict"].startswith("b beats control")


def test_input_problems_are_explained():
    d = trial(200)
    for kwargs, fragment in (({"control": "ghost"}, "not a value"), ({"outcome": "g"}, "numeric"), ({"group": "nope"}, "not found")):
        args = {"group": "g", "outcome": "y", "control": "control", **kwargs}
        with pytest.raises(ValueError, match=fragment):
            ex.analyze_experiment(d, args["group"], args["outcome"], args["control"])
    with pytest.raises(ValueError, match="fewer than"):
        ex.analyze_experiment(d.assign(g=np.where(np.arange(200) < 5, "control", "new")), "g", "y", "control")
    with pytest.raises(ValueError, match="at least two groups"):
        ex.analyze_experiment(d.assign(g="control"), "g", "y", "control")


def test_python_api_and_terminal(tmp_path):
    import kikolens as kl
    from typer.testing import CliRunner

    from kikolens.cli.main import app
    d = trial(3000)
    assert kl.experiment(d, "g", "y", "control", pre="pre", date="day", segments=["device"])["kind"] == "experiment"
    assert kl.srm(d, "g")["kind"] == "sample ratio" and kl.cuped(d, "g", "y", "pre", "control")["kind"] == "cuped"
    csv = tmp_path / "e.csv"
    d.to_csv(csv, index=False)
    runner = CliRunner()
    for args, text in ((["experiment", str(csv), "g", "y", "control", "--pre", "pre", "--date", "day", "--segments", "device"], "beats control"),
                       (["srm", str(csv), "g"], "split of"), (["cuped", str(csv), "g", "y", "pre", "control"], "removes")):
        out = runner.invoke(app, args)
        assert out.exit_code == 0 and text in " ".join(out.output.split()), (args, out.output)
    assert runner.invoke(app, ["experiment", str(csv), "g", "y", "ghost"]).exit_code == 1
