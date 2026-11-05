import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from typer.testing import CliRunner

from kikolens.analyze import segments as sg
from kikolens.cli.main import app


def blobs(n=450, seed=0):
    rng = np.random.default_rng(seed)
    centres = [(0, 0, 0), (6, 6, 0), (0, 6, 8)]
    truth = np.repeat([0, 1, 2], n // 3)
    pts = np.array([centres[t] for t in truth]) + rng.normal(0, 0.8, (len(truth), 3))
    return pd.DataFrame(pts, columns=["units", "price", "age"]), truth


def test_gmm_finds_three_groups_and_describes_them():
    df, truth = blobs()
    res = sg.gmm_auto(df)
    assert res["n_groups"] == 3 and adjusted_rand_score(truth, res["labels"]) > 0.95 and res["silhouette"] > 0.5
    assert res["table"]["BIC"].idxmin() == 2 and (res["confidence"] > 0.9).mean() > 0.9 and "3 group(s)" in res["summary"]
    p = res["profiles"]
    assert p["rows"].sum() == len(df) and abs(p["share"].sum() - 1) < 1e-9 and p["described as"].str.contains("high|low").all()
    assert len(res["figure"].data) == 1 and len(res["map"].data) == 3


def test_hierarchical_recovers_the_groups_and_samples_large_data():
    df, truth = blobs()
    res = sg.hierarchical(df, n_groups=3)
    assert adjusted_rand_score(truth, res["labels"]) > 0.95 and res["n_groups"] == 3 and len(res["figure"].data) > 5
    big = pd.concat([df] * 8, ignore_index=True)                       # 3.600 filas: se agrupa una muestra
    out = sg.hierarchical(big, n_groups=3)
    assert "sample" in out["summary"] and len(out["labels"]) == len(big) and out["labels"].nunique() == 3
    assert adjusted_rand_score(np.tile(truth, 8), out["labels"]) > 0.9
    with pytest.raises(ValueError, match="method"):
        sg.hierarchical(df, method="nope")
    with pytest.raises(ValueError, match="n_groups"):
        sg.hierarchical(df, n_groups=1)


def test_dbscan_follows_a_non_convex_shape_and_flags_noise():
    X, truth = make_moons(600, noise=0.05, random_state=0)
    df = pd.DataFrame(X, columns=["a", "b"])
    df.loc[len(df)] = [3.0, 3.0]                                         # un punto lejano
    res = sg.dbscan_auto(df)
    assert res["n_groups"] == 2 and res["labels"].iloc[-1] == -1 and res["noise_share"] < 0.05
    assert adjusted_rand_score(truth, res["labels"].iloc[:-1]) > 0.9
    kmeans_like = sg.hierarchical(df.iloc[:-1], n_groups=2, method="ward")
    assert adjusted_rand_score(truth, kmeans_like["labels"]) < adjusted_rand_score(truth, res["labels"].iloc[:-1])
    with pytest.raises(ValueError, match="every row is noise"):
        sg.dbscan_auto(df, eps=1e-4)
    with pytest.raises(ValueError, match="min_samples"):
        sg.dbscan_auto(df, min_samples=1)


def test_prepare_validation_and_missing_values():
    df, _ = blobs()
    df.loc[df.index[:10], "units"] = np.nan
    X, cols, filled = sg.prepare(df)
    assert not np.isnan(X).any() and cols == ["units", "price", "age"] and filled.notna().all().all()
    with pytest.raises(ValueError, match="At least two"):
        sg.prepare(df[["units"]])
    with pytest.raises(ValueError, match="30 rows"):
        sg.prepare(df.head(10))
    with pytest.raises(ValueError, match="k_max"):
        sg.gmm_auto(df, k_max=1)
    assert sg.gmm_auto(df, k_max=4)["labels"].index.equals(df.index)


def test_cli(tmp_path):
    df, _ = blobs()
    csv = tmp_path / "b.csv"
    df.to_csv(csv, index=False)
    for args in (["gmm"], ["hierarchical", "--groups", "3"], ["dbscan"]):
        out = CliRunner().invoke(app, ["segment", str(csv), *args, "--output", str(tmp_path / "out.csv")])
        assert out.exit_code == 0, (args, out.output)
    assert "group" in pd.read_csv(tmp_path / "out.csv").columns
    assert CliRunner().invoke(app, ["segment", str(csv), "magic"]).exit_code == 1
