import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression

from kikolens.analyze import regression as rg


def housing(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"size": rng.uniform(40, 200, n), "age": rng.uniform(0, 60, n),
                       "district": rng.choice(["centre", "north", "south"], n, p=[0.5, 0.3, 0.2]), "noise": rng.normal(size=n)})
    df["price"] = 50 + 3 * df["size"] - 1.5 * df["age"] + 40 * (df["district"] == "centre") + rng.normal(0, 20, n)
    return df


def test_ols_recovers_coefficients_and_matches_the_normal_equations():
    df = housing()
    res = rg.ols(df, "price", ["size", "age", "district"])
    t = res["table"].set_index("term")
    assert t.loc["size", "coefficient"] == pytest.approx(3, abs=0.2) and t.loc["age", "coefficient"] == pytest.approx(-1.5, abs=0.3)
    assert t.loc["district = north", "coefficient"] == pytest.approx(-40, abs=8)      # la referencia es el centro (el más frecuente)
    assert res["references"] == {"district": "centre"}
    A = np.column_stack([np.ones(len(df)), df["size"], df["age"], (df["district"] == "north"), (df["district"] == "south")]).astype(float)
    expected = np.linalg.solve(A.T @ A, A.T @ df["price"].to_numpy())
    assert np.allclose(res["table"]["coefficient"], expected)
    assert res["model"]["r2"] > 0.9 and res["model"]["F p-value"] < 1e-10
    low, high = t.loc["size", "low"], t.loc["size", "high"]
    assert low < 3 < high and (t["p-value"] >= 0).all()


def test_ols_p_values_and_intervals_match_scipy_linregress():
    df = housing()
    res = rg.ols(df, "price", ["size"])
    ref = stats.linregress(df["size"], df["price"])
    t = res["table"].set_index("term")
    assert t.loc["size", "coefficient"] == pytest.approx(ref.slope)
    assert t.loc["size", "std error"] == pytest.approx(ref.stderr)
    assert t.loc["size", "p-value"] == pytest.approx(ref.pvalue, rel=1e-6)
    assert res["model"]["r2"] == pytest.approx(ref.rvalue ** 2)


def test_a_useless_column_is_not_significant_and_sentences_say_what_matters():
    res = rg.ols(housing(), "price", ["size", "age", "noise"])
    t = res["table"].set_index("term")
    assert t.loc["noise", "p-value"] > 0.05 and t.loc["size", "p-value"] < 1e-6
    text = " ".join(res["sentences"])
    assert "size" in text and "noise" not in text and "explain" in res["summary"]


def test_collinearity_and_heteroscedasticity_are_flagged():
    df = housing()
    df["size_copy"] = df["size"] * 2 + np.random.default_rng(1).normal(0, 0.5, len(df))
    flags = " ".join(rg.ols(df, "price", ["size", "size_copy", "age"])["flags"])
    assert "VIF" in flags
    rng = np.random.default_rng(2)
    wide = pd.DataFrame({"x": rng.uniform(1, 10, 600)})
    wide["y"] = 2 * wide["x"] + rng.normal(0, 1, 600) * wide["x"]
    assert any("spread of the errors" in f for f in rg.ols(wide, "y")["flags"])


def test_missing_values_dates_and_validation():
    df = housing()
    df.loc[df.index[:20], "size"] = np.nan
    df["when"] = pd.date_range("2020-01-01", periods=len(df))
    res = rg.ols(df, "price", ["size", "when"])
    assert res["rows dropped for missing values"] == 20 and "when (days)" in set(res["table"]["term"])
    with pytest.raises(ValueError, match="must be a number"):
        rg.ols(df, "district")
    with pytest.raises(ValueError, match="not found"):
        rg.ols(df, "nope")
    with pytest.raises(ValueError, match="too few"):
        rg.ols(df.head(12), "price", ["size", "age", "district"])
    with pytest.raises(ValueError, match="at least one"):
        rg.ols(df, "price", [])
    with pytest.raises(ValueError, match="None of the chosen"):
        rg.ols(df.assign(flat=1), "price", ["flat"])


def churn(n=800, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"tenure": rng.uniform(0, 48, n), "tickets": rng.poisson(2, n), "plan": rng.choice(["basic", "pro"], n)})
    z = 1.0 - 0.06 * df["tenure"] + 0.5 * df["tickets"] - 0.8 * (df["plan"] == "pro")
    df["left"] = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "yes", "no")
    return df


def test_logit_matches_unpenalised_sklearn_and_reads_as_odds():
    df = churn()
    res = rg.logit(df, "left", ["tenure", "tickets"], positive="yes")
    ref = LogisticRegression(penalty=None, max_iter=5000).fit(df[["tenure", "tickets"]], df["left"] == "yes")
    t = res["table"].set_index("term")
    assert t.loc["tenure", "coefficient"] == pytest.approx(ref.coef_[0][0], abs=1e-3)
    assert t.loc["tickets", "coefficient"] == pytest.approx(ref.coef_[0][1], abs=1e-3)
    assert t.loc["tickets", "odds ratio"] == pytest.approx(np.exp(t.loc["tickets", "coefficient"]))
    assert res["model"]["converged"] and res["model"]["auc"] > 0.7 and res["model"]["accuracy"] > res["model"]["baseline accuracy"] - 0.05
    assert t.loc["tenure", "p-value"] < 0.001 and t.loc["tenure", "odds ratio"] < 1
    assert any("tickets" in s and "multiplies the odds" in s for s in res["sentences"])


def test_logit_with_categories_and_default_positive_class():
    res = rg.logit(churn(), "left")
    assert res["model"]["positive"] == "yes" and "plan = pro" in set(res["table"]["term"])
    flipped = rg.logit(churn(), "left", positive="no")
    assert flipped["table"].set_index("term").loc["tickets", "coefficient"] < 0


def test_logit_flags_perfect_separation_and_validates():
    df = churn()
    df["leak"] = (df["left"] == "yes").astype(float) * 5 + np.random.default_rng(0).normal(0, 0.01, len(df))
    res = rg.logit(df, "left", ["leak", "tenure"])
    assert any("separation" in f or "converge" in f for f in res["flags"])
    with pytest.raises(ValueError, match="exactly two"):
        rg.logit(df.assign(left=np.where(np.arange(len(df)) % 3 == 0, "maybe", df["left"])), "left")
    with pytest.raises(ValueError, match="not a value"):
        rg.logit(churn(), "left", positive="maybe")
    with pytest.raises(ValueError, match="at least 5"):
        rg.logit(churn().assign(left=lambda d: np.where(np.arange(len(d)) < 3, "yes", "no")), "left")


def test_figures():
    res = rg.ols(housing(), "price", ["size", "age"])
    assert len(rg.coefficient_figure(res).data[0].y) == 2 and len(rg.residual_figure(res).data[0].x) == 400
    log = rg.logit(churn(), "left", ["tenure"])
    assert rg.coefficient_figure(log).layout.xaxis.type == "log"
    with pytest.raises(ValueError):
        rg.residual_figure(log)


def test_cli_regress_and_spikes(tmp_path):
    from typer.testing import CliRunner
    from kikolens.cli.main import app
    runner = CliRunner()
    csv = tmp_path / "h.csv"
    housing().to_csv(csv, index=False)
    out = runner.invoke(app, ["regress", str(csv), "price", "--using", "size,age", "--output", str(tmp_path / "t.csv")])
    assert out.exit_code == 0 and "size" in out.output and (tmp_path / "t.csv").exists()
    churn_csv = tmp_path / "c.csv"
    churn().to_csv(churn_csv, index=False)
    out = runner.invoke(app, ["regress", str(churn_csv), "left", "--positive", "yes"])
    assert out.exit_code == 0 and "odds" in out.output
    assert runner.invoke(app, ["regress", str(csv), "district"]).exit_code == 1
    assert runner.invoke(app, ["regress", str(csv), "nope"]).exit_code == 1

    t = np.arange(300)
    rng = np.random.default_rng(0)
    s = pd.Series(100 + 15 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 2, 300), index=pd.date_range("2024-01-01", periods=300))
    s.iloc[100] += 60
    series = tmp_path / "s.csv"
    pd.DataFrame({"day": s.index, "v": s.values}).to_csv(series, index=False)
    out = runner.invoke(app, ["spikes", str(series), "day", "v", "--output", str(tmp_path / "u.csv")])
    assert out.exit_code == 0 and "unusual" in out.output and (tmp_path / "u.csv").exists()
    assert runner.invoke(app, ["spikes", str(series), "day", "nope"]).exit_code == 1
