import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import sequential as sq
from kikolens.cli.main import app


def test_real_effect_is_detected_early_with_a_valid_interval():
    rng = np.random.default_rng(0)
    r = sq.msprt(rng.random(8000) < 0.10, rng.random(8000) < 0.14)
    assert r["stopped_at"] and r["stopped_at"] < 8000 and r["decision"] == "treatment higher" and r["p_value"] < 0.05
    assert r["interval"][0] < r["difference"] < r["interval"][1] and "can stop" in r["summary"]
    worse = sq.msprt(rng.random(8000) < 0.14, rng.random(8000) < 0.10)
    assert worse["decision"] == "control higher"


def test_peeking_at_every_observation_does_not_inflate_false_alarms():
    rng = np.random.default_rng(1)
    alarms = sum(sq.msprt(rng.random(2500) < 0.1, rng.random(2500) < 0.1, tau=0.03)["stopped_at"] is not None for _ in range(150))
    assert alarms / 150 <= 0.07                                  # alfa 5 %, aunque se mira tras cada dato
    # un test t clásico repetido a cada paso se equivoca mucho más (lo que el mSPRT evita)
    from scipy import stats
    classic = 0
    for _ in range(60):
        a, b = rng.normal(size=400), rng.normal(size=400)
        classic += any(stats.ttest_ind(a[:k], b[:k]).pvalue < 0.05 for k in range(20, 401, 5))
    assert classic / 60 > 0.15


def test_no_decision_means_keep_collecting_and_the_p_value_never_rises():
    rng = np.random.default_rng(2)
    r = sq.msprt(rng.normal(size=60), rng.normal(size=60))
    assert r["decision"] == "keep collecting" and r["stopped_at"] is None and "Keep collecting" in r["summary"]
    assert (np.diff(r["p_path"]) <= 1e-12).all() and (r["p_path"] <= 1).all()
    assert len(r["figure"].data) == 1


def test_validation():
    with pytest.raises(ValueError, match="At least 20"):
        sq.msprt([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError, match="alpha"):
        sq.msprt(np.arange(30.0), np.arange(30.0), alpha=0.9)


def test_from_frame_uses_arrival_order_and_checks_columns():
    rng = np.random.default_rng(5)
    n = 4000
    df = pd.DataFrame({"arm": np.tile(["a", "b"], n // 2), "won": np.where(np.tile(["a", "b"], n // 2) == "a", rng.random(n) < 0.10, rng.random(n) < 0.17).astype(int),
                       "day": np.repeat(pd.date_range("2024-01-01", periods=n // 2), 2)})
    shuffled = df.sample(frac=1, random_state=0)
    r = sq.from_frame(shuffled, "arm", "won", "a", order="day")
    assert r["decision"] == "treatment higher"
    with pytest.raises(ValueError, match="not a value"):
        sq.from_frame(df, "arm", "won", "zzz")
    with pytest.raises(ValueError, match="exactly two"):
        sq.from_frame(df.assign(arm=(["a", "b", "c"] * n)[:n]), "arm", "won", "a")
    with pytest.raises(ValueError, match="numeric"):
        sq.from_frame(df.assign(won="x"), "arm", "won", "a")
    with pytest.raises(ValueError, match="not found"):
        sq.from_frame(df, "arm", "nope", "a")


def test_planning_compares_with_a_fixed_test_and_keeps_false_alarms_low():
    res = sq.planning(0.10, 0.03, runs=60, max_n=8000)
    assert res["fixed_n"] > 1000 and res["median_stop"] < 8000 and res["false_positive_rate"] <= 0.12 and "fixed-size" in res["summary"]
    with pytest.raises(ValueError):
        sq.planning(0.1, 0.0)
    with pytest.raises(ValueError):
        sq.planning(0.98, 0.05)


def test_thompson_sends_traffic_to_the_better_variant_and_beats_an_even_split():
    r = sq.thompson([0.04, 0.10, 0.05], n=6000, seed=1)
    assert r["final_allocation"].argmax() == 1 and r["final_allocation"][1] > 0.6 and r["extra_conversions"] > 0
    assert r["regret"] >= 0 and len(r["figure"].data) == 3 and abs(r["share"].iloc[-1].sum() - 1) < 1e-9
    for bad in ([0.1], [0.1, 1.2], [0.1, 0]):
        with pytest.raises(ValueError):
            sq.thompson(bad)
    with pytest.raises(ValueError, match="200"):
        sq.thompson([0.1, 0.2], n=50)


def test_cli_sequential(tmp_path):
    rng = np.random.default_rng(4)
    n = 4000
    arm = np.tile(["control", "new"], n // 2)
    df = pd.DataFrame({"arm": arm, "won": np.where(arm == "control", rng.random(n) < 0.1, rng.random(n) < 0.15).astype(int)})
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    out = CliRunner().invoke(app, ["ab-seq", str(csv), "arm", "won", "control"])
    assert out.exit_code == 0 and "can stop" in out.output
    assert CliRunner().invoke(app, ["ab-seq", str(csv), "arm", "won", "zzz"]).exit_code == 1
    out = CliRunner().invoke(app, ["bandit", "0.04,0.08", "--visitors", "3000"])
    assert out.exit_code == 0 and "bandit got" in out.output
    assert CliRunner().invoke(app, ["bandit", "0.04"]).exit_code == 1
