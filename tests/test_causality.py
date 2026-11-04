import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import causality as cz

TRUE_EFFECT = 2.0


def confounded(n=1500, seed=0, overlap=1.0, effect=TRUE_EFFECT):
    """La edad y los ingresos empujan a recibir el tratamiento y también suben el resultado: la diferencia directa miente."""
    rng = np.random.default_rng(seed)
    age = rng.normal(0, 1, n)
    income = rng.normal(0, 1, n)
    logit = overlap * (1.2 * age + 0.8 * income)
    treated = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    outcome = effect * treated + 3 * age + 2 * income + rng.normal(0, 1, n)
    return pd.DataFrame({"treated": treated, "outcome": outcome, "age": age, "income": income, "group": rng.choice(["a", "b"], n)})


# ── Efecto del tratamiento ───────────────────────────────────────────────────
def test_the_raw_difference_is_biased_and_the_adjusted_methods_recover_the_truth():
    res = cz.estimate_effect(confounded(), "treated", "outcome", ["age", "income", "group"])
    table = res["table"].set_index("method")
    assert table.loc["naive", "effect"] > TRUE_EFFECT + 2                                   # el sesgo es grande y positivo
    for method in ("regression", "ipw", "aipw", "matching"):
        assert abs(table.loc[method, "effect"] - TRUE_EFFECT) < 0.6, method
    assert table.loc["aipw", "low"] < TRUE_EFFECT < table.loc["aipw", "high"] and res["effect"] == pytest.approx(table.loc["aipw", "effect"])
    assert set(table["estimates"]) == {"ATE", "ATT"} and not table.loc["naive", "adjusts for confounding"]
    assert any("confounding matters" in n for n in res["notes"]) and "E-value" in res["summary"]


def test_weighting_balances_the_covariates_the_raw_groups_differ_on():
    res = cz.estimate_effect(confounded(), "treated", "outcome", ["age", "income", "group"])
    bal = res["balance"].set_index("covariate")
    assert abs(bal.loc["age", "before"]) > 0.5 and abs(bal.loc["age", "after"]) < 0.15 and abs(bal.loc["income", "after"]) < 0.15
    assert bal.loc["age", "improved"] and len(res["figure"].data) == 1 and len(res["overlap_figure"].data) == 2 and len(res["balance_figure"].data) == 2


def test_the_e_value_grows_with_the_effect_and_is_one_when_the_interval_includes_zero():
    small, large = cz.e_value(0.2, 1.0, 0.1), cz.e_value(2.0, 1.0, 1.5)
    assert large["point"] > small["point"] > 1 and cz.e_value(0.0, 1.0)["point"] == pytest.approx(1.0)
    assert cz.e_value(1.0, 1.0, -0.2)["interval"] == 1.0 and cz.e_value(-1.0, 1.0)["point"] == pytest.approx(cz.e_value(1.0, 1.0)["point"])


def test_poor_overlap_is_reported():
    res = cz.estimate_effect(confounded(overlap=3.5), "treated", "outcome", ["age", "income"])
    assert res["outside_overlap"] > 0.1 and any("too different" in n for n in res["notes"])


def test_no_effect_gives_an_interval_that_includes_zero():
    res = cz.estimate_effect(confounded(effect=0.0, seed=3), "treated", "outcome", ["age", "income"])
    assert res["interval"][0] < 0 < res["interval"][1] and res["e_value"]["interval"] == 1.0


def test_treatment_values_text_labels_and_validation():
    df = confounded()
    named = df.assign(treated=df["treated"].map({1: "yes", 0: "no"}))
    assert cz.estimate_effect(named, "treated", "outcome", ["age"], method="ipw")["treated_value"] == "yes"
    assert cz.estimate_effect(named, "treated", "outcome", ["age"], treated_value="no", method="ipw")["effect"] < 0
    for kwargs, message in [({"treatment": "nope"}, "not found"), ({"covariates": []}, "covariate"), ({"method": "magic"}, "method"), ({"trim": 0.5}, "trim"),
                            ({"treated_value": "maybe"}, "not a value")]:
        args = {"df": df, "treatment": "treated", "outcome": "outcome", "covariates": ["age"], **kwargs}
        with pytest.raises(ValueError, match=message):
            cz.estimate_effect(**args)
    with pytest.raises(ValueError, match="exactly two"):
        cz.estimate_effect(df.assign(treated=df["group"].map({"a": 0, "b": 1}).where(df.index % 3 != 0, 2)), "treated", "outcome", ["age"])
    with pytest.raises(ValueError, match="at least 60|At least 60"):
        cz.estimate_effect(df.head(30), "treated", "outcome", ["age"])
    with pytest.raises(ValueError, match="numeric"):
        cz.estimate_effect(df.assign(outcome="x"), "treated", "outcome", ["age"])


# ── Uplift ───────────────────────────────────────────────────────────────────
def uplift_data(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    t = rng.integers(0, 2, n)                                           # asignación al azar
    y = 1.0 + z + t * (3.0 * (x > 0)) + rng.normal(0, 1, n)           # el efecto solo existe cuando x > 0
    return pd.DataFrame({"t": t, "y": y, "x": x, "z": z})


def test_uplift_finds_who_benefits_and_beats_random_targeting():
    res = cz.uplift(uplift_data(), "t", "y", ["x", "z"])
    assert res["top_uplift"] > res["average_uplift"] * 1.5 and res["qini"] > 0 and res["randomised_gap"] < 0.1 and not res["notes"]
    d = res["deciles"]
    assert d["predicted uplift"].is_monotonic_decreasing and d["observed uplift"].iloc[0] > d["observed uplift"].iloc[-1] + 1.5
    assert res["drivers"].index[0] == "x" and len(res["figure"].data) == 2 and "Qini" in res["summary"]
    cate = res["cate"]
    assert cate[uplift_data()["x"].to_numpy() > 0.5].mean() > cate[uplift_data()["x"].to_numpy() < -0.5].mean() + 2


def test_uplift_warns_when_the_assignment_is_not_random():
    res = cz.uplift(confounded(), "treated", "outcome", ["age", "income"])
    assert res["randomised_gap"] > 0.1 and any("randomly assigned" in n for n in res["notes"])


# ── Discontinuidad ───────────────────────────────────────────────────────────
def rd_data(n=3000, seed=2, jump=3.0, sort=False):
    rng = np.random.default_rng(seed)
    score = rng.uniform(40, 80, n)
    if sort:                                                            # quienes casi llegan se «empujan» por encima
        score = np.where((score > 57) & (score < 60) & (rng.random(n) < 0.7), 60 + rng.random(n), score)
    outcome = 10 + 0.15 * (score - 60) + 0.002 * (score - 60) ** 2 + jump * (score >= 60) + rng.normal(0, 1, n)
    return pd.DataFrame({"score": score, "outcome": outcome})


def test_the_jump_at_the_cutoff_is_recovered_with_a_curve_underneath():
    res = cz.regression_discontinuity(rd_data(), "score", "outcome", 60)
    assert abs(res["effect"] - 3.0) < 0.5 and res["interval"][0] < 3.0 < res["interval"][1] and res["density_p"] > 0.05
    assert len(res["table"]) == 5 and res["table"]["effect"].between(2.2, 3.8).all() and "Crossing 60" in res["summary"]
    assert abs(cz.regression_discontinuity(rd_data(), "score", "outcome", 60, order=2, kernel="uniform")["effect"] - 3.0) < 0.8


def test_no_jump_and_a_manipulated_running_variable():
    assert cz.regression_discontinuity(rd_data(jump=0.0, seed=5), "score", "outcome", 60)["interval"][0] < 0 < cz.regression_discontinuity(rd_data(jump=0.0, seed=5), "score", "outcome", 60)["interval"][1]
    sorted_res = cz.regression_discontinuity(rd_data(sort=True), "score", "outcome", 60)
    assert sorted_res["density_p"] < 0.05 and any("sorting themselves" in n for n in sorted_res["notes"])


def test_discontinuity_validation():
    df = rd_data()
    for args, message in [(dict(running="nope", outcome="outcome", cutoff=60), "numeric"), (dict(running="score", outcome="outcome", cutoff=200), "30 rows"),
                          (dict(running="score", outcome="outcome", cutoff=60, kernel="gauss"), "kernel"), (dict(running="score", outcome="outcome", cutoff=60, order=3), "order"),
                          (dict(running="score", outcome="outcome", cutoff=60, bandwidth=-1), "positive")]:
        with pytest.raises(ValueError, match=message):
            cz.regression_discontinuity(df, **args)
    with pytest.raises(ValueError, match="Too few"):
        cz.regression_discontinuity(df, "score", "outcome", 60, bandwidth=0.01)


# ── Control sintético ────────────────────────────────────────────────────────
def panel(effect=-5.0, seed=4, units=12, periods=40, cut=30):
    rng = np.random.default_rng(seed)
    t = np.arange(periods)
    common = 50 + 0.4 * t + 3 * np.sin(t / 3)
    rows = []
    donors = {}
    for u in range(units):
        donors[u] = common * rng.uniform(0.6, 1.4) + rng.normal(0, 0.4, periods) + rng.uniform(-5, 5)
    treated = 0.5 * donors[0] + 0.3 * donors[1] + 0.2 * donors[2] + rng.normal(0, 0.3, periods)
    treated[cut:] += effect
    for u, series in donors.items():
        rows += [(f"unit{u + 1}", pd.Timestamp("2020-01-01") + pd.DateOffset(months=int(i)), series[i]) for i in range(periods)]
    rows += [("treated", pd.Timestamp("2020-01-01") + pd.DateOffset(months=int(i)), treated[i]) for i in range(periods)]
    return pd.DataFrame(rows, columns=["unit", "month", "sales"]), pd.Timestamp("2020-01-01") + pd.DateOffset(months=cut)


def test_synthetic_control_recovers_the_effect_and_its_placebo_test_agrees():
    df, cut = panel()
    res = cz.synthetic_control(df, "unit", "month", "sales", "treated", cut)
    assert abs(res["effect"] - (-5.0)) < 1.2 and res["p_value"] <= 0.15 and res["pre_fit"] < 1.0 and not res["notes"]
    assert abs(res["weights"].sum() - 1) < 0.05 and (res["weights"] >= 0).all()        # varias combinaciones de donantes pueden imitar igual de bien
    assert len(res["figure"].data) == 2 and len(res["gap_figure"].data) == res["placebos"] + 1 and "placebo p" in res["summary"]


def test_synthetic_control_with_no_effect_is_not_significant():
    df, cut = panel(effect=0.0, seed=9)
    res = cz.synthetic_control(df, "unit", "month", "sales", "treated", cut)
    assert abs(res["effect"]) < 1.0 and res["p_value"] > 0.2


def test_synthetic_control_validation():
    df, cut = panel()
    for args, message in [(dict(treated_unit="ghost", intervention=cut), "not a value"), (dict(treated_unit="treated", intervention=cut, donors=["unit1", "unit2"]), "3 control"),
                          (dict(treated_unit="treated", intervention=pd.Timestamp("2020-02-01")), "5 periods"), (dict(treated_unit="treated", intervention=pd.Timestamp("2025-01-01")), "after")]:
        with pytest.raises(ValueError, match=message):
            cz.synthetic_control(df, "unit", "month", "sales", **args)
    with pytest.raises(ValueError, match="not found"):
        cz.synthetic_control(df, "nope", "month", "sales", "treated", cut)


# ── Variables instrumentales ─────────────────────────────────────────────────
def iv_data(n=4000, seed=6, strength=1.0):
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=n)                                         # motivación: afecta a la vez a la educación y al salario
    z = rng.normal(size=n)                                              # instrumento (p. ej. cercanía a una universidad)
    educ = strength * z + 0.9 * hidden + rng.normal(size=n)
    wage = 1.5 * educ + 2.0 * hidden + rng.normal(size=n)
    return pd.DataFrame({"wage": wage, "educ": educ, "z": z, "age": rng.normal(size=n)})


def test_iv_removes_the_bias_that_ordinary_regression_has():
    res = cz.iv_2sls(iv_data(), "wage", "educ", ["z"], ["age"])
    assert res["ols"] > 1.9 and abs(res["effect"] - 1.5) < 0.25 and res["interval"][0] < 1.5 < res["interval"][1] and res["first_stage_f"] > 100 and not res["notes"]
    assert "instrument" in res["summary"] and res["p_value"] < 0.001


def test_weak_instruments_are_flagged_and_validation():
    weak = cz.iv_2sls(iv_data(strength=0.03, seed=8), "wage", "educ", ["z"])
    assert weak["first_stage_f"] < 10 and any("Weak instrument" in n for n in weak["notes"])
    two = iv_data().assign(z2=lambda d: d["z"] * 0.5 + np.random.default_rng(0).normal(size=len(d)))
    assert cz.iv_2sls(two, "wage", "educ", ["z", "z2"])["first_stage_f"] > 50
    for args, message in [(dict(instruments=[]), "at least one"), (dict(instruments=["nope"]), "numeric"), (dict(outcome="wage", instruments=["z"], covariates=["nope"]), "numeric")]:
        base = {"outcome": "wage", "endogenous": "educ", **args}
        with pytest.raises(ValueError, match=message):
            cz.iv_2sls(iv_data(), **base)
    with pytest.raises(ValueError, match="50 complete"):
        cz.iv_2sls(iv_data().head(20), "wage", "educ", ["z"])
