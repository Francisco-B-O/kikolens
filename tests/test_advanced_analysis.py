import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from kikolens.analyze import autofeatures as af
from kikolens.analyze import bayes, causal, conformal, counterfactual, customers, fairness, survival
from kikolens.analyze import modellab as lab


# ── Supervivencia ────────────────────────────────────────────────────────────
def lifetimes(n=500, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.choice(["a", "b"], n)
    t = rng.exponential(np.where(g == "a", 10, 20))
    c = rng.exponential(30, n)
    return pd.DataFrame({"time": np.minimum(t, c), "event": (t <= c).astype(int), "group": g})


def test_kaplan_meier_matches_hand_computation():
    curve = survival.kaplan_meier([1, 2, 3, 4, 5], [1, 1, 0, 1, 0])
    s = curve.set_index("time")["survival"]
    assert s[1.0] == pytest.approx(0.8) and s[2.0] == pytest.approx(0.6) and s[4.0] == pytest.approx(0.3)
    assert (curve["lower"] <= curve["survival"]).all() and (curve["survival"] <= curve["upper"]).all()
    assert survival.median_survival(curve) == 4.0


def test_logrank_detects_a_real_difference_and_not_a_fake_one():
    df = lifetimes()
    res = survival.survival_by_group(df, "time", "event", "group")
    assert res["test"]["p_value"] < 0.001 and res["medians"]["a"] < res["medians"]["b"]
    same = df.assign(group=np.random.default_rng(1).choice(["a", "b"], len(df)))
    assert survival.survival_by_group(same, "time", "event", "group")["test"]["p_value"] > 0.01


def test_survival_accepts_yes_no_and_rejects_bad_events():
    df = lifetimes().assign(event=lambda d: d["event"].map({1: "yes", 0: "no"}))
    assert survival.survival_by_group(df, "time", "event")["curves"]["all"]["survival"].iloc[-1] < 1
    with pytest.raises(ValueError, match="0/1"):
        survival.kaplan_meier([1, 2, 3, 4, 5, 6], [0, 1, 2, 1, 0, 1])
    with pytest.raises(ValueError):
        survival.survival_by_group(lifetimes(), "time", "event", "missing")


# ── Clientes ─────────────────────────────────────────────────────────────────
def orders(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(300):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 150)))
        for k in range(int(rng.integers(1, 8))):
            rows.append((u, start + pd.Timedelta(days=int(30 * k * rng.uniform(0.7, 1.3))), rng.uniform(10, 100)))
    return pd.DataFrame(rows, columns=["user", "date", "amount"])


def test_cohort_retention_starts_at_one_and_hides_unknown_future():
    res = customers.cohort_retention(orders(), "user", "date")
    m = res["matrix"]
    assert (m[0] == 1.0).all() and m.max().max() <= 1.0
    assert m.iloc[-1].isna().sum() >= 1                        # la última cohorte no puede tener periodos futuros
    assert res["sizes"].sum() == orders()["user"].nunique()
    with pytest.raises(ValueError):
        customers.cohort_retention(orders(), "user", "date", period="Decade")


def test_rfm_assigns_every_customer_one_segment():
    res = customers.rfm(orders(), "user", "date", "amount")
    table = res["customers"]
    assert len(table) == 300 and table["segment"].notna().all() and table[["R", "F", "M"]].isin(range(1, 6)).all().all()
    assert res["segments"]["customers"].sum() == 300
    assert res["segments"]["share of value"].sum() == pytest.approx(1.0)


def test_funnel_counts_only_users_who_passed_earlier_stages():
    df = pd.DataFrame({"u": [1, 1, 1, 2, 2, 3, 4, 4, 4, 4], "s": ["visit", "cart", "pay", "visit", "pay", "cart",
                                                                  "visit", "cart", "pay", "ship"]})
    res = customers.funnel(df, "u", "s", ["visit", "cart", "pay", "ship"])
    assert res["table"]["users"].tolist() == [3, 2, 2, 1]       # el usuario 2 pagó sin pasar por el carrito
    with pytest.raises(ValueError):
        customers.funnel(df, "u", "s", ["visit"])


def test_association_rules_find_a_planted_pair():
    rng = np.random.default_rng(0)
    rows = []
    for t in range(400):
        items = list(rng.choice(list("abcdefgh"), 2, replace=False))
        if rng.random() < 0.4:
            items += ["bread", "butter"]
        rows += [(t, i) for i in set(items)]
    res = customers.association_rules(pd.DataFrame(rows, columns=["t", "i"]), "t", "i", min_support=0.05)
    rules = res["rules"]
    pair = rules[(rules["if bought"] == "bread") & (rules["then also"] == "butter")].iloc[0]
    assert pair["lift"] > 1.5 and pair["confidence"] > 0.9
    assert rules.iloc[0]["lift"] >= pair["lift"]                    # ordenadas por lift
    assert (res["rules"]["confidence"] >= 0.3).all()


# ── Modelos: equidad, conforme y contrafactuales ─────────────────────────────
@pytest.fixture(scope="module")
def loans():
    rng = np.random.default_rng(2)
    n = 900
    df = pd.DataFrame({"income": rng.normal(50, 15, n), "debt": rng.normal(20, 8, n),
                       "region": rng.choice(["north", "south"], n), "age": rng.integers(20, 65, n)})
    score = 0.08 * df["income"] - 0.1 * df["debt"] + rng.normal(0, 1, n) - 0.8 * (df["region"] == "south")
    df["approved"] = np.where(score > np.median(score), "yes", "no")
    df["amount"] = 3 * df["income"] - 2 * df["debt"] + rng.normal(0, 5, n)
    return df


def fitted(df, target, task, model):
    X, y, _ = lab.prepare_xy(df, target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0)
    pipe = lab.make_pipeline(Xtr, lab.model_pool(task, 0, False, [model])[model][0]).fit(Xtr, ytr)
    return pipe, Xtr, Xte, ytr, yte


def test_fairness_flags_the_group_the_model_penalises(loans):
    pipe, Xtr, Xte, ytr, yte = fitted(loans, "approved", "classification", "Logistic Regression")
    out = fairness.audit(pipe, Xte, yte, loans["region"], "yes")
    rates = out["table"].set_index("group")["selection rate"]
    assert rates["north"] > rates["south"] and out["selection_ratio"] < 0.8 and out["flags"]
    fair = fairness.audit(pipe, Xte, yte, pd.Series(np.random.default_rng(0).choice(["p", "q"], len(loans)), index=loans.index), "yes")
    assert fair["selection_ratio"] > 0.8
    with pytest.raises(ValueError):
        fairness.audit(pipe, Xte, yte, pd.Series("one", index=loans.index), "yes")
    with pytest.raises(ValueError):
        fairness.group_metrics([1, 0], [1, 0], ["a", "b"], "nope")


def test_conformal_regression_reaches_its_coverage(loans):
    pipe, Xtr, Xte, ytr, yte = fitted(loans, "amount", "regression", "Ridge")
    half = len(Xte) // 2
    cm = conformal.ConformalModel(pipe, "regression", 0.9).calibrate(Xte.iloc[:half], yte.iloc[:half])
    got = cm.evaluate(Xte.iloc[half:], yte.iloc[half:])
    assert 0.82 <= got["coverage"] <= 0.98 and got["width"] > 0
    assert (cm.predict_interval(Xte)["lower"] < cm.predict_interval(Xte)["upper"]).all()
    wide = conformal.ConformalModel(pipe, "regression", 0.99).calibrate(Xte.iloc[:half], yte.iloc[:half])
    assert wide.threshold > cm.threshold


def test_conformal_classification_sets_and_errors(loans):
    pipe, Xtr, Xte, ytr, yte = fitted(loans, "approved", "classification", "Logistic Regression")
    half = len(Xte) // 2
    cm = conformal.ConformalModel(pipe, "classification", 0.9).calibrate(Xte.iloc[:half], yte.iloc[:half])
    got = cm.evaluate(Xte.iloc[half:], yte.iloc[half:])
    assert got["coverage"] >= 0.82 and got["average_set_size"] >= 1
    assert cm.predict_set(Xte)["size"].between(1, 2).all()
    with pytest.raises(ValueError):
        cm.predict_interval(Xte)
    with pytest.raises(ValueError):
        conformal.ConformalModel(pipe, "classification").predict_set(Xte)
    with pytest.raises(ValueError):
        conformal.ConformalModel(pipe, "regression", 0.2)


def test_counterfactual_finds_a_small_change_that_flips_the_decision(loans):
    pipe, Xtr, Xte, ytr, yte = fitted(loans, "approved", "classification", "Logistic Regression")
    denied = Xte[pipe.predict(Xte) == "no"].iloc[[0]]
    out = counterfactual.find(pipe, Xtr, denied, "classification", desired="yes", fixed=["age"])
    assert not out["already"] and len(out["found"]) >= 1
    assert "age" not in out["found"].iloc[0]["change"] and out["found"].iloc[0]["variables"] <= 2
    approved = Xte[pipe.predict(Xte) == "yes"].iloc[[0]]
    assert counterfactual.find(pipe, Xtr, approved, "classification", desired="yes")["already"]
    with pytest.raises(ValueError):
        counterfactual.find(pipe, Xtr, denied, "classification", desired="maybe")
    with pytest.raises(ValueError):
        counterfactual.find(pipe, Xtr, Xte.iloc[:2], "classification", desired="yes")


def test_counterfactual_regression_target_range(loans):
    pipe, Xtr, Xte, ytr, yte = fitted(loans, "amount", "regression", "Ridge")
    row = Xte.iloc[[0]]
    now = float(pipe.predict(row)[0])
    out = counterfactual.find(pipe, Xtr, row, "regression", target_range=(now + 20, None))
    assert len(out["found"]) >= 1
    with pytest.raises(ValueError):
        counterfactual.find(pipe, Xtr, row, "regression")


# ── Causal y bayesiano ───────────────────────────────────────────────────────
def traffic(effect=15.0, n=160, at=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n)
    base = 100 + 0.1 * np.arange(n) + 10 * np.sin(2 * np.pi * np.arange(n) / 7)
    control = pd.DataFrame({"c1": base * 0.8 + rng.normal(0, 2, n), "c2": base * 1.1 + rng.normal(0, 2, n)}, index=idx)
    y = pd.Series(base + rng.normal(0, 2, n), index=idx)
    y.iloc[at:] += effect
    return y, control, idx[at]


def test_causal_impact_recovers_the_effect_with_controls():
    y, control, when = traffic()
    res = causal.causal_impact(y, when, control)
    assert res["detected"] and abs(res["average_effect"] - 15) < 3 and res["p_value"] < 0.05
    assert res["interval"][0] < res["average_effect"] < res["interval"][1]


def test_causal_impact_finds_nothing_when_nothing_happened():
    y, control, when = traffic(effect=0.0)
    res = causal.causal_impact(y, when, control)
    assert not res["detected"] and abs(res["average_effect"]) < 3


def test_causal_impact_without_controls_uses_the_trend_and_validates():
    y, _, when = traffic(effect=25.0)
    res = causal.causal_impact(y, when)
    assert res["detected"] and res["average_effect"] > 10 and "forecast" in res["method"]
    with pytest.raises(ValueError):
        causal.causal_impact(y, y.index[5])
    with pytest.raises(ValueError):
        causal.causal_impact(y, y.index[-1])
    with pytest.raises(ValueError):
        causal.causal_impact(y.reset_index(drop=True), 100)


def test_diff_in_diff_recovers_the_treatment_effect():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"treated": rng.integers(0, 2, n), "post": rng.integers(0, 2, n)})
    df["sales"] = 50 + 10 * df["treated"] + 5 * df["post"] + 8 * df["treated"] * df["post"] + rng.normal(0, 3, n)
    res = causal.diff_in_diff(df, "sales", "treated", "post")
    assert abs(res["effect"] - 8) < 1.5 and res["p_value"] < 0.001 and res["interval"][0] < 8 < res["interval"][1]
    with pytest.raises(ValueError):
        causal.diff_in_diff(df.assign(post=df["post"] * 0), "sales", "treated", "post")
    with pytest.raises(ValueError):
        causal.diff_in_diff(df.assign(treated=df["treated"].astype(str) + "x"), "sales", "treated", "post")


def test_bayesian_rates_agree_with_intuition():
    clear = bayes.bayesian_rates(100, 1000, 140, 1000, labels=("old", "new"))
    assert clear["probability_b_better"] > 0.99 and clear["leader"] == "new" and clear["expected_loss_b"] < 0.001
    tie = bayes.bayesian_rates(100, 1000, 101, 1000)
    assert 0.4 < tie["probability_b_better"] < 0.65
    swapped = bayes.bayesian_rates(140, 1000, 100, 1000)
    assert swapped["probability_b_better"] < 0.01
    with pytest.raises(ValueError):
        bayes.bayesian_rates(5, 3, 1, 10)


def test_bayesian_ab_from_a_table_handles_binary_and_numeric_outcomes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"arm": rng.choice(["control", "new"], 2000)})
    df["bought"] = (rng.random(2000) < np.where(df["arm"] == "new", 0.15, 0.10)).astype(int)
    df["spend"] = rng.normal(np.where(df["arm"] == "new", 55, 50), 10)
    assert bayes.bayesian_ab(df, "arm", "bought", "control")["kind"] == "rate"
    res = bayes.bayesian_ab(df, "arm", "spend", "control")
    assert res["kind"] == "mean" and res["probability_b_better"] > 0.95 and res["leader"] == "new"
    with pytest.raises(ValueError):
        bayes.bayesian_ab(df, "arm", "spend", "nope")


# ── Variables nuevas ─────────────────────────────────────────────────────────
def test_feature_search_finds_the_planted_ratio_and_the_steps_replay():
    rng = np.random.default_rng(0)
    n = 700
    df = pd.DataFrame({"revenue": rng.uniform(100, 1000, n), "visits": rng.uniform(10, 100, n), "noise": rng.normal(size=n)})
    df["profit"] = 3 * df["revenue"] / df["visits"] + rng.normal(0, 0.1, n)
    res = af.suggest_features(df, "profit", max_new=2)
    assert res["suggestions"] and res["final"] > res["baseline"] + 0.01
    assert any(s["kind"] == "ratio" for s in res["suggestions"])
    new, steps = af.apply_suggestions(df, res["suggestions"])
    assert set(s["name"] for s in res["suggestions"]) <= set(new.columns) and steps[0][0] == "add_column"
    from kikolens.pipeline.recipe import Recipe
    recipe = Recipe()
    for op, params, message in steps:
        recipe.add(op, params, message)
    replay, _ = recipe.apply(df)
    pd.testing.assert_frame_equal(replay, new)


def test_feature_search_reports_when_nothing_helps_and_validates():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300), "y": rng.normal(size=300)})
    res = af.suggest_features(df, "y", max_new=2, min_gain=0.2)
    assert res["suggestions"] == [] and "None of" in res["summary"]
    with pytest.raises(ValueError):
        af.suggest_features(pd.DataFrame({"x": ["u", "v"] * 40, "y": range(80)}), "y")
    with pytest.raises(ValueError):
        af.suggest_features(df, "y", max_new=0)


# ── Línea de comandos ────────────────────────────────────────────────────────
def test_cli_commands(tmp_path):
    from typer.testing import CliRunner
    from kikolens.cli.main import app
    runner = CliRunner()
    life = tmp_path / "life.csv"
    lifetimes().to_csv(life, index=False)
    out = runner.invoke(app, ["survival", str(life), "time", "event", "--group", "group"])
    assert out.exit_code == 0 and "log-rank" in out.output
    assert runner.invoke(app, ["survival", str(life), "time", "nope"]).exit_code == 1

    orders_csv = tmp_path / "orders.csv"
    orders().to_csv(orders_csv, index=False)
    assert runner.invoke(app, ["cohorts", str(orders_csv), "user", "date"]).exit_code == 0
    seg = tmp_path / "seg.csv"
    out = runner.invoke(app, ["rfm", str(orders_csv), "user", "date", "amount", "--output", str(seg)])
    assert out.exit_code == 0 and len(pd.read_csv(seg)) == 300

    y, control, when = traffic()
    daily = pd.DataFrame({"day": y.index, "sales": y.values, "c1": control["c1"].values})
    csv = tmp_path / "daily.csv"
    daily.to_csv(csv, index=False)
    out = runner.invoke(app, ["impact", str(csv), "day", "sales", "--at", str(when.date()), "--control", "c1"])
    assert out.exit_code == 0 and "effect" in out.output

    rng = np.random.default_rng(0)
    feats = pd.DataFrame({"revenue": rng.uniform(100, 1000, 500), "visits": rng.uniform(10, 100, 500)})
    feats["profit"] = 3 * feats["revenue"] / feats["visits"] + rng.normal(0, 0.1, 500)
    fcsv = tmp_path / "feats.csv"
    feats.to_csv(fcsv, index=False)
    out = runner.invoke(app, ["features", str(fcsv), "profit", "--apply", str(tmp_path / "f2.csv"), "--recipe", str(tmp_path / "r.json"),
                              "--max-new", "1"])
    assert out.exit_code == 0 and (tmp_path / "r.json").exists() and pd.read_csv(tmp_path / "f2.csv").shape[1] == 4

    ab = pd.DataFrame({"arm": ["a"] * 300 + ["b"] * 300, "won": [1] * 30 + [0] * 270 + [1] * 60 + [0] * 240})
    abcsv = tmp_path / "ab.csv"
    ab.to_csv(abcsv, index=False)
    out = runner.invoke(app, ["ab-bayes", str(abcsv), "arm", "won", "a"])
    assert out.exit_code == 0 and "probability" in out.output
