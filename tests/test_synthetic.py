import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens import demo
from kikolens.analyze import synthetic as sy
from kikolens.cli.main import app


def people(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 70, n)
    city = rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])
    income = 1000 + 60 * age + 800 * (city == "A") + rng.normal(0, 300, n)
    df = pd.DataFrame({"person": [f"p{i}" for i in range(n)], "age": age, "city": city, "income": income,
                       "joined": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1500, n), "D"), "vip": rng.random(n) < 0.1})
    df.loc[rng.random(n) < 0.1, "income"] = np.nan
    return df


def test_marginals_and_types_are_preserved():
    real = people()
    out = sy.synthesize(real, 1500, seed=1)
    s = out["data"]
    assert list(s.columns) == list(real.columns) and len(s) == 1500 and out["identifiers"] == ["person"]
    assert s["person"].is_unique and not set(s["person"]) & set(real["person"])                   # los identificadores no se copian
    assert pd.api.types.is_integer_dtype(s["age"]) and s["age"].between(20, 69).all() and s["vip"].dtype == bool
    assert abs(s["city"].value_counts(normalize=True)["A"] - 0.5) < 0.05 and abs(s["income"].mean() - real["income"].mean()) < 150
    assert abs(s["income"].isna().mean() - real["income"].isna().mean()) < 0.03
    assert s["joined"].between(real["joined"].min(), real["joined"].max()).all()


def test_correlations_survive_including_numbers_with_categories():
    real = people()
    s = sy.synthesize(real, 3000, seed=2)["data"]
    assert abs(real["age"].corr(real["income"], method="spearman") - s["age"].corr(s["income"], method="spearman")) < 0.08
    assert abs(real.groupby("city")["income"].mean()["A"] - s.groupby("city")["income"].mean()["A"]) < 250
    independent = real.assign(income=real["income"].sample(frac=1, random_state=0).to_numpy())
    assert abs(independent["age"].corr(independent["income"], method="spearman")) < 0.1


def test_same_seed_same_data_and_validation():
    real = people()
    a, b = sy.synthesize(real, 200, seed=5)["data"], sy.synthesize(real, 200, seed=5)["data"]
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(sy.synthesize(real, 200, seed=6)["data"])
    with pytest.raises(ValueError, match="30 rows"):
        sy.synthesize(real.head(10))
    with pytest.raises(ValueError, match="between"):
        sy.synthesize(real, 0)
    with pytest.raises(ValueError, match="No column"):
        sy.synthesize(pd.DataFrame({"id": [f"x{i}" for i in range(60)]}))


def test_report_scores_good_data_high_and_flags_copies():
    real = people()
    good = sy.synthesize(real, 1200, seed=3)["data"]
    rep = sy.report(real, good)
    assert rep["fidelity"] > 85 and rep["exact_copies"] < 0.05 and rep["near_copies"] < 0.2 and "Fidelity" in rep["summary"]
    copy = sy.report(real, real.sample(600, random_state=0))
    assert copy["exact_copies"] > 0.95 and copy["near_copies"] > 0.9 and copy["fidelity"] > 95
    broken = good.assign(age=np.random.default_rng(0).integers(20, 30, len(good)), city="A")
    assert sy.report(real, broken)["fidelity"] < rep["fidelity"] - 15


def test_utility_of_synthetic_data_for_a_model():
    real = demo.sales()
    real = real.assign(unit_price=real["unit_price"].str.replace(",", "").astype(float)).drop(columns="date")
    synth = sy.synthesize(real, 1500, seed=0)["data"]
    res = sy.utility(real, synth, "revenue")
    assert res["metric"] == "R2" and res["trained_on_real"] > 0.5 and res["ratio"] > 0.7 and "of the real performance" in res["summary"]


def test_cli(tmp_path):
    csv = tmp_path / "p.csv"
    people(400).to_csv(csv, index=False)
    out = CliRunner().invoke(app, ["synth", str(csv), str(tmp_path / "s.csv"), "--rows", "300", "--report"])
    assert out.exit_code == 0, out.output
    assert len(pd.read_csv(tmp_path / "s.csv")) == 300 and "Fidelity" in " ".join(out.output.split())
    assert CliRunner().invoke(app, ["synth", str(csv), str(tmp_path / "s.txt")]).exit_code == 1
