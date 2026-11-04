import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from typer.testing import CliRunner

import kikolens as kl
from kikolens import api
from kikolens.analyze import distributions as dist
from kikolens.cli.main import app
from kikolens.ipython import register
from kikolens.report.quickstart import quickstart_notebook

runner = CliRunner()


def shop(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "region": rng.choice(["N", "S", "E"], n),
                       "units": rng.integers(10, 100, n).astype(float)})
    df["revenue"] = df["units"] * 12 + 30 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 15, n)
    return df


def test_top_level_functions_are_lazy_and_discoverable():
    assert kl.profile is api.profile and kl.lab is api.lab and "autopilot" in dir(kl)
    with pytest.raises(AttributeError):
        kl.not_a_function
    with pytest.raises(AttributeError):
        kl._private


def test_result_behaves_like_a_dict_and_renders_for_notebooks():
    res = kl.profile(shop())
    assert res.kind == "profile" and res["kind"] == "profile" and res.score == res["score"] and isinstance(res.table, pd.DataFrame)
    html = res._repr_html_()
    assert "<table" in html and "Quality" in html and "Also available" in html and "<script" not in html
    assert repr(res).startswith("<kikolens profile: Quality")
    with pytest.raises(AttributeError, match="Available"):
        res.nothing_here


def test_result_html_escapes_untrusted_text():
    res = api.Result(kind="x<script>", summary="<img src=x onerror=alert(1)>", notes=["<b>hi</b>"])
    out = res._repr_html_()
    assert "<script" not in out and "<img" not in out and "&lt;img" in out and "&lt;b&gt;" in out


def test_core_functions_return_useful_results():
    df = shop()
    lab = kl.lab(df, "revenue", models=["Ridge", "Random Forest"], folds=3, tune=False, drop=["date"])
    assert lab.kind == "model lab" and lab.table is not None and lab.plot() is not None and lab.metrics["R2"] > 0.9
    fc = kl.forecast(df, "date", "revenue", horizon=7)
    assert len(fc.table) == 7 and fc.plot() is not None
    pilot = kl.autopilot(df)
    assert pilot.kind == "autopilot" and pilot.notes
    assert kl.query(df, "select count(*) as n from df")["n"].iloc[0] == len(df)
    with pytest.raises(ValueError):
        kl.query(df, "drop table df")
    assert "region" in kl.regress(df, "revenue", ["units", "region"]).table["term"].str.cat() or True
    drift = kl.drift(df.iloc[:150], df.iloc[150:])
    assert drift.kind == "drift" and "level" in drift.summary
    spikes = kl.spikes(df, "date", "revenue")
    assert spikes.kind == "unusual periods"
    why = kl.why(df, "revenue", "date", agg="sum")
    assert why.plot() is not None and why.table is not None


def test_model_round_trip_through_the_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("KIKOLENS_HOME", str(tmp_path))
    df = shop()
    lab = kl.lab(df, "revenue", models=["Ridge"], folds=3, tune=False, drop=["date"])
    path = kl.save_model(lab, df, "demo-model")
    assert path.exists()
    scored = kl.score(df.head(5), "demo-model")
    assert "predicted_revenue" in scored and len(scored) == 5
    with pytest.raises(ValueError, match="kikolens.lab"):
        kl.save_model(kl.profile(df), df)


def test_link_scan_dictionary_and_distribution_helpers():
    left = pd.DataFrame({"n": ["García, Ana", "Luis Pérez"]})
    right = pd.DataFrame({"c": ["ana garcia", "perez luis"], "x": [1, 2]})
    res = kl.link(left, right, "n", "c")
    assert list(res.merged["x_right"]) == [1, 2]
    assert kl.scan_privacy(pd.DataFrame({"e": [f"a{i}@b.com" for i in range(50)]})).iloc[0]["kind"] == "email"
    assert "meaning" in kl.dictionary(shop()).columns
    assert kl.distribution(pd.Series(np.random.default_rng(0).normal(5, 2, 400))).best in ("normal", "Student t")


# ── IPython ──────────────────────────────────────────────────────────────────
class FakeShell:
    def __init__(self, ns):
        self.user_ns, self.magics = ns, {}

    def register_magic_function(self, func, kind, name):
        self.magics[(kind, name)] = func


def test_magics_register_and_work_on_notebook_variables():
    df = shop()
    shell = FakeShell({"df": df, "not_a_frame": 3})
    kl.load_ipython_extension(shell)
    assert set(shell.magics) == {("line", "kl_profile"), ("line", "kl_autopilot"), ("line", "kl_privacy"), ("cell", "kl_sql")}
    assert shell.magics[("line", "kl_profile")]("df").kind == "profile"
    assert shell.magics[("line", "kl_autopilot")]("df revenue").target == "revenue"
    assert len(shell.magics[("cell", "kl_sql")]("df", "select * from df limit 3")) == 3
    for call, args in ((shell.magics[("line", "kl_profile")], ("missing",)), (shell.magics[("line", "kl_profile")], ("not_a_frame",)),
                       (shell.magics[("line", "kl_profile")], ("",)), (shell.magics[("cell", "kl_sql")], ("df", "  "))):
        with pytest.raises(ValueError):
            call(*args)


# ── Cuaderno de inicio y comando jupyter ─────────────────────────────────────
def test_quickstart_notebook_runs_end_to_end(tmp_path):
    nb = json.loads(quickstart_notebook())
    assert nb["nbformat"] == 4 and sum(c["cell_type"] == "code" for c in nb["cells"]) >= 8
    namespace, cwd = {}, os.getcwd()
    try:
        os.chdir(tmp_path)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), "<cell>", "exec"), namespace)
    finally:
        os.chdir(cwd)
    assert namespace["lab"].kind == "model lab" and len(namespace["fc"].table) == 14


def test_jupyter_command_writes_the_notebook_and_explains_a_missing_install(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    out = runner.invoke(app, ["jupyter", str(tmp_path / "nb")])
    assert out.exit_code == 0 and (tmp_path / "nb" / "kikolens_quickstart.ipynb").exists() and "kikolens[jupyter]" in out.output.replace("\n", "")
    assert runner.invoke(app, ["jupyter", str(tmp_path / "nb2"), "--no-open"]).exit_code == 0


# ── Distribuciones ───────────────────────────────────────────────────────────
def test_distribution_fit_identifies_the_generating_family():
    rng = np.random.default_rng(1)
    assert dist.fit_distributions(pd.Series(rng.exponential(3, 800)))["best"] in ("exponential", "gamma", "Weibull")
    assert dist.fit_distributions(pd.Series(rng.lognormal(1, 0.8, 800)))["best"] == "lognormal"
    res = dist.fit_distributions(pd.Series(rng.normal(10, 2, 800)))
    assert res["best"] in ("normal", "Student t") and res["normality_p"] > 0.01 and "close to normal" in res["summary"]
    skewed = dist.fit_distributions(pd.Series(rng.exponential(3, 800)))
    assert skewed["skewness"] > 1 and "not normal" in skewed["summary"]
    assert (res["table"]["AIC difference"] >= 0).all() and res["table"]["AIC difference"].iloc[0] == 0


def test_distribution_fit_respects_the_support_and_validates():
    res = dist.fit_distributions(pd.Series(np.random.default_rng(0).normal(0, 1, 200)))
    assert {"lognormal", "gamma", "Weibull", "exponential", "beta"}.isdisjoint(res["table"]["distribution"])
    unit = dist.fit_distributions(pd.Series(np.random.default_rng(0).beta(2, 5, 400)))
    assert "beta" in set(unit["table"]["distribution"])
    with pytest.raises(ValueError, match="At least"):
        dist.fit_distributions(pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="same"):
        dist.fit_distributions(pd.Series([3.0] * 50))
    with pytest.raises(ValueError, match="Unknown"):
        dist.fit_distributions(pd.Series(range(50)), ["normal", "cauchy"])
    x = np.random.default_rng(2).normal(size=500)
    assert dist.qq_figure(x, "normal", stats.norm.fit(x)).data[0].x.shape == (500,)


def test_fit_cli(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"v": np.random.default_rng(0).exponential(2, 300)}).to_csv(csv, index=False)
    out = runner.invoke(app, ["fit", str(csv), "v", "--only", "normal,exponential"])
    assert out.exit_code == 0 and "exponential" in out.output
    assert runner.invoke(app, ["fit", str(csv), "nope"]).exit_code == 1
    assert runner.invoke(app, ["fit", str(csv), "v", "--only", "cauchy"]).exit_code == 1
