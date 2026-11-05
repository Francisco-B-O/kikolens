import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import anomalies as an
from kikolens.analyze import dimension as dm
from kikolens.analyze import drift as dr
from kikolens.analyze import stattests as st


@pytest.fixture
def df():
    rng = np.random.default_rng(1)
    n = 300
    frame = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "g": rng.choice(list("xyz"), n),
                          "c": rng.choice(["u", "v"], n)})
    frame["d"] = frame["a"] * 2 + rng.normal(size=n) * 0.1
    frame["val"] = frame["g"].map({"x": 0.0, "y": 0.0, "z": 1.5}) + rng.normal(size=n)
    frame["conv"] = (rng.random(n) < np.where(frame["c"] == "u", 0.2, 0.35)).astype(int)
    return frame


# ── Anomalías ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("method", an.METHODS)
def test_every_method_finds_the_planted_outlier(df, method):
    data = df[["a", "b", "d"]].copy()
    data.loc[3, ["a", "b"]] = [12, -12]
    result = an.detect_anomalies(data, method=method, contamination=0.02)
    assert 3 in result["table"].index and result["n_flagged"] == 6
    assert result["flags"].sum() == result["n_flagged"] and len(result["scores"]) == len(data)
    assert "a is" in result["table"].loc[3, "why"]


def test_anomaly_input_validation(df):
    with pytest.raises(ValueError, match="Unknown method"):
        an.detect_anomalies(df, method="magic")
    with pytest.raises(ValueError, match="between 0.1% and 30%"):
        an.detect_anomalies(df, contamination=0.9)
    with pytest.raises(ValueError, match="At least 20 rows"):
        an.detect_anomalies(df.head(10))
    with pytest.raises(ValueError, match="at least two"):
        an.detect_anomalies(df, columns=["a"])
    with pytest.raises(ValueError, match="not found"):
        an.detect_anomalies(df, columns=["nope"])
    with pytest.raises(ValueError, match="no numeric"):
        an.detect_anomalies(df[["g", "c"]])
    one = an.detect_anomalies(df, columns=["a"], method="Robust z-score")
    assert one["columns"] == ["a"]


def test_anomalies_ignore_missing_values_and_leave_input_alone(df):
    data = df[["a", "b", "d"]].copy()
    data.loc[::7, "b"] = np.nan
    snapshot = data.copy()
    result = an.detect_anomalies(data)
    assert result["scores"].notna().all()
    pd.testing.assert_frame_equal(data, snapshot)


def test_anomaly_figures_render(df):
    result = an.detect_anomalies(df[["a", "b", "d"]])
    assert len(an.anomaly_map(df, result).data) == 2
    assert an.score_histogram(result).layout.shapes
    single = an.detect_anomalies(df, columns=["a"], method="Robust z-score")
    assert len(an.anomaly_map(df, single).data) == 2


# ── Comparar grupos ──────────────────────────────────────────────────────────
def test_anova_with_post_hoc_finds_the_odd_group(df):
    result = st.compare_groups(df, "val", "g")
    assert result["test"] == "One-way ANOVA" and result["significant"]
    pairs = result["post_hoc"].set_index(["group A", "group B"])["p (Tukey)"]
    assert pairs[("x", "y")] > 0.05 and pairs[("z", "x")] < 0.01
    assert set(result["summary"]["group"]) == {"x", "y", "z"}


def test_two_groups_without_effect_is_not_significant(df):
    result = st.compare_groups(df, "val", "c")
    assert result["test"] == "Welch's t-test" and not result["significant"]
    assert "does not prove" in result["conclusion"]


def test_skewed_data_uses_rank_tests():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"v": np.r_[rng.exponential(1, 25), rng.exponential(4, 25)], "g": ["a"] * 25 + ["b"] * 25})
    result = st.compare_groups(data, "v", "g")
    assert result["test"] == "Mann-Whitney U test" and result["significant"]
    three = pd.DataFrame({"v": np.r_[rng.exponential(1, 20), rng.exponential(1, 20), rng.exponential(6, 20)],
                          "g": list("a" * 20 + "b" * 20 + "c" * 20)})
    result = st.compare_groups(three, "v", "g")
    assert result["test"] == "Kruskal-Wallis test" and "p (Holm-adjusted)" in result["post_hoc"].columns


def test_compare_groups_validation(df):
    with pytest.raises(ValueError, match="numeric"):
        st.compare_groups(df, "g", "c")
    with pytest.raises(ValueError, match="not found"):
        st.compare_groups(df, "val", "nope")
    with pytest.raises(ValueError, match="At least two groups"):
        st.compare_groups(df.assign(one="same"), "val", "one")
    many = df.assign(k=np.arange(len(df)) % 40)
    with pytest.raises(ValueError, match="at most"):
        st.compare_groups(many, "val", "k")


# ── Asociación y correlación ────────────────────────────────────────────────
def test_association_detects_dependence_and_independence(df):
    dependent = df.assign(cc=np.where(df["conv"] == 1, "yes", "no"), dd=np.where(df["conv"] == 1, "p", "q"))
    result = st.association(dependent, "cc", "dd")
    assert result["significant"] and result["cramers_v"] > 0.9 and result["effect_size"] == "large"
    assert not st.association(df, "g", "c")["significant"]


def test_association_small_2x2_uses_fisher():
    tiny = pd.DataFrame({"a": ["x"] * 4 + ["y"] * 4, "b": ["p", "p", "p", "q", "q", "q", "q", "p"]})
    assert st.association(tiny, "a", "b")["test"] == "Fisher's exact test"
    with pytest.raises(ValueError, match="different columns"):
        st.association(tiny, "a", "a")
    with pytest.raises(ValueError, match="at least two"):
        st.association(tiny.assign(c="same"), "a", "c")


def test_correlation_test_flags_outlier_driven_relationships(df):
    strong = st.correlation_test(df, "a", "d")
    assert strong["pearson_r"] > 0.99 and strong["significant"] and strong["ci"][0] > 0.98
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(size=50), 30.0]
    y = np.r_[rng.normal(size=50), 30.0]
    driven = st.correlation_test(pd.DataFrame({"x": x, "y": y}), "x", "y")
    assert "extreme points" in driven["conclusion"]
    with pytest.raises(ValueError, match="numeric"):
        st.correlation_test(df, "a", "g")
    with pytest.raises(ValueError, match="At least 5"):
        st.correlation_test(df.head(3), "a", "b")


# ── A/B ──────────────────────────────────────────────────────────────────────
def test_ab_test_on_rates(df):
    result = st.ab_test(df, "c", "conv", control="u")
    assert result["kind"] == "proportion" and result["treatment"] == "v" and result["significant"]
    assert result["difference"] > 0 and result["prob_treatment_better"] > 0.95
    assert result["ci"][0] < result["difference"] < result["ci"][1]


def test_ab_test_on_means_reports_needed_sample(df):
    result = st.ab_test(df, "c", "val", control="u")
    assert result["kind"] == "mean" and not result["significant"]
    assert result["needed_per_group"] > min(result["n"]) and "rows per group" in result["conclusion"]


def test_ab_test_validation(df):
    with pytest.raises(ValueError, match="exactly two groups"):
        st.ab_test(df, "g", "conv", control="x")
    with pytest.raises(ValueError, match="not a value"):
        st.ab_test(df, "c", "conv", control="zzz")
    with pytest.raises(ValueError, match="numeric or have exactly two"):
        st.ab_test(df.assign(t=df["g"]), "c", "t", control="u")
    with pytest.raises(ValueError, match="not a value of"):
        st.ab_test(df, "c", "conv", control="u", success="7")


def test_sample_size_formulas_match_known_values():
    assert st.sample_size_proportions(0.10, 0.02) == 3841   # valor de referencia (alfa 5%, potencia 80%)
    assert st.sample_size_means(10, 2) == 393
    assert st.sample_size_proportions(0.1, -0.02) > 0
    with pytest.raises(ValueError):
        st.sample_size_proportions(0.0, 0.1)
    with pytest.raises(ValueError):
        st.sample_size_proportions(0.5, 0.6)
    with pytest.raises(ValueError):
        st.sample_size_means(0, 1)


# ── Cribado ──────────────────────────────────────────────────────────────────
def test_benjamini_hochberg_is_monotone_and_bounded():
    p = np.array([0.001, 0.04, 0.03, 0.5, 0.9])
    adjusted = st.benjamini_hochberg(p)
    assert (adjusted >= p).all() and (adjusted <= 1).all()
    assert np.allclose(adjusted, [0.005, 0.0666667, 0.0666667, 0.625, 0.9])
    assert st.benjamini_hochberg([]).size == 0


def test_screening_finds_the_real_driver_and_corrects_for_chance(df):
    table = st.screen_associations(df, "conv")
    assert table.iloc[0]["column"] == "c" and bool(table.iloc[0]["related"])
    noise = {c: np.random.default_rng(i).normal(size=len(df)) for i, c in enumerate("pqrstuvwxy")}
    wide = st.screen_associations(pd.DataFrame({**noise, "conv": df["conv"]}), "conv")
    assert not wide["related"].any()   # con 10 columnas de ruido no debe haber falsos descubrimientos
    with pytest.raises(ValueError, match="not found"):
        st.screen_associations(df, "nope")


# ── Dimensiones ──────────────────────────────────────────────────────────────
def test_pca_recovers_redundancy(df):
    result = dm.pca_analysis(df[["a", "d", "b", "val"]])
    assert result["explained"][0] > 0.45 and abs(result["explained"].sum() - 1) < 1e-9
    assert set(result["top_features"]["PC1"][:2]) == {"a", "d"}
    assert result["scores"].shape == (len(df), 4) and result["components_for_90"] == 3
    assert len(dm.scree_figure(result).data) == 2 and len(dm.loadings_figure(result).data) == 2
    with pytest.raises(ValueError, match="between 2 and"):
        dm.pca_analysis(df[["a", "d", "b"]], n_components=9)
    with pytest.raises(ValueError, match="At least two"):
        dm.pca_analysis(df[["a", "g"]])


def test_projection_figures_color_by_category_or_number(df):
    result = dm.pca_analysis(df[["a", "d", "b", "val"]])
    by_group = dm.projection_figure(result["scores"], "PC1", "PC2", df["g"])
    assert {t.name for t in by_group.data} == {"x", "y", "z"}
    assert len(dm.projection_figure(result["scores"], "PC1", "PC2", df["val"]).data) == 1
    assert len(dm.projection_figure(result["scores"], "PC1", "PC2").data) == 1


def test_tsne_and_vif(df):
    coords, note = dm.tsne_embedding(df[["a", "b", "d"]].head(120))
    assert coords.shape == (120, 2) and note is None
    big = pd.concat([df[["a", "b", "d"]]] * 11, ignore_index=True)
    coords, note = dm.tsne_embedding(big)
    assert len(coords) == dm.MAX_TSNE_ROWS and "sample" in note
    vif = dm.vif_table(df[["a", "b", "d", "val"]]).set_index("column")
    assert vif.loc["a", "verdict"] == "severe" and vif.loc["b", "verdict"] == "fine"
    with pytest.raises(ValueError, match="Too few complete rows"):
        dm.vif_table(df.head(5)[["a", "b", "d", "val"]])


# ── Drift ────────────────────────────────────────────────────────────────────
def test_identical_data_is_stable(df):
    result = dr.compare_datasets(df, df.copy())
    assert result["drifted"] == 0 and result["level"] == "low"
    assert (result["table"]["verdict"] == "stable").all()


def test_shifted_columns_are_flagged(df):
    now = df.copy()
    now["a"] = now["a"] + 1.5
    now.loc[:150, "g"] = "new"
    now["conv"] = (np.random.default_rng(3).random(len(now)) < 0.8).astype(int)
    result = dr.compare_datasets(df, now)
    verdicts = result["table"].set_index("column")["verdict"]
    assert verdicts["a"] == "drift" and verdicts["g"] == "drift" and verdicts["conv"] == "drift"
    assert verdicts["b"] == "stable" and result["level"] == "high"
    assert "new: new" in result["table"].set_index("column").loc["g", "change"]


def test_schema_changes_and_empty_inputs(df):
    now = df.drop(columns=["b"]).assign(extra=1)
    result = dr.compare_datasets(df, now)
    assert result["only_in_reference"] == ["b"] and result["only_in_current"] == ["extra"]
    with pytest.raises(ValueError, match="no columns in common"):
        dr.compare_datasets(df[["a"]], df[["b"]])
    with pytest.raises(ValueError, match="at least one row"):
        dr.compare_datasets(df, df.head(0))


def test_missing_rate_change_is_noticed(df):
    now = df.copy()
    now.loc[:100, "b"] = np.nan
    row = dr.compare_datasets(df, now)["table"].set_index("column").loc["b"]
    assert row["verdict"] == "moderate" and "empty" in row["change"]


def test_psi_known_properties():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(size=2000))
    assert dr.psi_numeric(base, base.copy()) < 1e-6
    assert dr.psi_numeric(base, base + 2) > dr.PSI_HIGH
    assert np.isnan(dr.psi_numeric(base, pd.Series([], dtype=float)))
    assert dr.psi_categorical(pd.Series(list("aabb")), pd.Series(list("aabb"))) < 1e-9


def test_distribution_figures_and_split(df):
    assert dr.distribution_figure(df, df, "a").layout.barmode == "overlay"
    assert dr.distribution_figure(df, df, "g").layout.barmode == "group"
    first, rest = dr.split_by_position(df, 0.5)
    assert len(first) == 150 and len(rest) == 150
    with pytest.raises(ValueError):
        dr.split_by_position(df, 0.95)
    with pytest.raises(ValueError, match="at least 5"):
        dr.split_by_position(df.head(8), 0.5)


# ── CLI ──────────────────────────────────────────────────────────────────────
from typer.testing import CliRunner  # noqa: E402

from kikolens.cli.main import app  # noqa: E402

runner = CliRunner()


@pytest.fixture
def csv_files(tmp_path, df):
    now = df.copy()
    now["a"] = now["a"] + 3
    now["b"] = now["b"] * 4
    now.loc[:150, "g"] = "new"
    ref_path, now_path, same_path = tmp_path / "ref.csv", tmp_path / "now.csv", tmp_path / "same.csv"
    df.to_csv(ref_path, index=False)
    now.to_csv(now_path, index=False)
    df.to_csv(same_path, index=False)
    return str(ref_path), str(now_path), str(same_path)


def test_cli_anomalies(csv_files, tmp_path):
    out = runner.invoke(app, ["anomalies", csv_files[0], "--columns", "a,b,d", "--share", "3", "--output", str(tmp_path / "o.csv")])
    assert out.exit_code == 0 and (tmp_path / "o.csv").exists() and "flagged" in out.output
    assert runner.invoke(app, ["anomalies", csv_files[0], "--method", "magic"]).exit_code == 1


def test_cli_drift_exit_codes(csv_files, tmp_path):
    ref, now, same = csv_files
    assert runner.invoke(app, ["drift", ref, same]).exit_code == 0
    changed = runner.invoke(app, ["drift", ref, now, "--output", str(tmp_path / "d.csv")])
    assert changed.exit_code == 2 and (tmp_path / "d.csv").exists()   # 2 = ha cambiado mucho
    assert runner.invoke(app, ["drift", ref, str(tmp_path / "missing.csv")]).exit_code == 1


def test_cli_screen_and_ab(csv_files):
    out = runner.invoke(app, ["screen", csv_files[0], "conv"])
    assert out.exit_code == 0 and "related to" in out.output
    assert runner.invoke(app, ["screen", csv_files[0], "nope"]).exit_code == 1
    out = runner.invoke(app, ["ab", csv_files[0], "c", "conv", "--control", "u"])
    assert out.exit_code == 0 and "reliably higher" in out.output
    assert runner.invoke(app, ["ab", csv_files[0], "g", "conv", "--control", "x"]).exit_code == 1


# ── Drift con tipos guardados como texto ─────────────────────────────────────
def test_drift_reads_text_dates_numbers_and_skips_identifiers():
    rng = np.random.default_rng(0)
    n = 400
    raw = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d"),
                        "price": [f"{v:,}" for v in rng.integers(1000, 5000, n)],
                        "id": [f"c{i}" for i in range(n)], "ch": rng.choice(["a", "b"], n)})
    result = dr.compare_datasets(raw.iloc[:200], raw.iloc[200:])
    table = result["table"].set_index("column")
    assert table.loc["price", "type"] == "numeric" and table.loc["price", "verdict"] == "stable"
    assert table.loc["date", "type"] == "date" and table.loc["date", "verdict"] == "info"
    assert table.loc["id", "type"] == "identifier" and table.loc["id", "verdict"] == "info"
    assert result["drifted"] == 0 and result["level"] == "low" and "of 2 columns" in result["summary"]
    assert dr.distribution_figure(raw.iloc[:200], raw.iloc[200:], "price").layout.barmode == "overlay"


def test_drift_catches_a_shift_hidden_in_text_numbers():
    rng = np.random.default_rng(0)
    before = pd.DataFrame({"price": [f"{v:,}" for v in rng.integers(1000, 3000, 300)]})
    after = pd.DataFrame({"price": [f"{v:,}" for v in rng.integers(2500, 6000, 300)]})
    row = dr.compare_datasets(before, after)["table"].iloc[0]
    assert row["type"] == "numeric" and row["verdict"] == "drift"
