import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import drivers
from kikolens.cli.main import app

runner = CliRunner()


def shop(seed=0):
    rng = np.random.default_rng(seed)
    n = 1800
    df = pd.DataFrame({"date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 180, n), "D"),
                       "region": rng.choice(["north", "south", "east"], n), "channel": rng.choice(["web", "store"], n),
                       "price": rng.uniform(80, 120, n)})
    late = df["date"] >= pd.Timestamp("2024-05-01")
    df.loc[late & (df["region"] == "south"), "price"] *= 0.4          # el sur se hunde en el último periodo
    return df


def test_the_collapsing_region_is_found_and_contributions_add_up():
    df = shop()
    res = drivers.why_changed(df, "price", "date", ("2024-03-01", "2024-05-01"), ("2024-05-01", "2024-07-01"), agg="sum")
    assert res["change"] < 0 and res["tables"]["region"]["value"].iloc[0] == "south"
    for table in res["tables"].values():
        assert table["contribution"].sum() == pytest.approx(res["change"])
    assert res["ranking"]["dimension"].iloc[0] == "region" and "south" in res["summary"]
    assert res["pairs"].iloc[0]["combination"].startswith("region = south")


def test_mean_change_splits_into_mix_and_rate_exactly():
    df = shop()
    res = drivers.why_changed(df, "price", "date", ("2024-03-01", "2024-05-01"), ("2024-05-01", "2024-07-01"), agg="mean")
    table = res["tables"]["region"]
    assert (table["mix effect"] + table["rate effect"]).sum() == pytest.approx(res["change"])
    assert table.set_index("value").loc["south", "rate effect"] < -10 and res["pairs"] is None


def test_a_value_that_appears_only_later_is_all_mix():
    df = shop()
    df.loc[(df["date"] >= "2024-05-01") & (df["region"] == "east"), "region"] = "new"
    res = drivers.why_changed(df, "price", "date", ("2024-03-01", "2024-05-01"), ("2024-05-01", "2024-07-01"), ["region"], "mean")
    t = res["tables"]["region"].set_index("value")
    assert t.loc["new", "rate effect"] == pytest.approx(0) and t["contribution"].sum() == pytest.approx(res["change"])
    assert t.loc["east", "weight after"] == 0


def test_count_and_default_periods_and_missing_values():
    df = shop()
    df.loc[df.index[:50], "region"] = np.nan
    res = drivers.why_changed(df, None, "date", agg="count")
    assert res["total_before"] > 0 and "(empty)" in set(res["tables"]["region"]["value"])
    (b0, b1), (a0, a1) = drivers.default_periods(df["date"])
    assert b1 == a0 and a1 > a0 > b0


def test_validation_errors():
    df = shop()
    with pytest.raises(ValueError, match="not found"):
        drivers.why_changed(df, "nope", "date")
    with pytest.raises(ValueError, match="numeric"):
        drivers.why_changed(df, "region", "date")
    with pytest.raises(ValueError, match="agg"):
        drivers.why_changed(df, "price", "date", agg="max")
    with pytest.raises(ValueError, match="at least 5"):
        drivers.why_changed(df, "price", "date", ("2030-01-01", "2030-02-01"), ("2030-02-01", "2030-03-01"))
    with pytest.raises(ValueError, match="end after"):
        drivers.why_changed(df, "price", "date", ("2024-03-01", "2024-02-01"), ("2024-05-01", "2024-07-01"))
    with pytest.raises(ValueError, match="distinct values"):
        drivers.why_changed(df[["date", "price"]], "price", "date")
    with pytest.raises(ValueError, match="6 different dates"):
        drivers.default_periods(pd.Series(["2024-01-01"] * 10))


def test_waterfall_ends_where_the_data_ends():
    df = shop()
    res = drivers.why_changed(df, "price", "date", ("2024-03-01", "2024-05-01"), ("2024-05-01", "2024-07-01"))
    fig = drivers.waterfall_figure(res, "region")
    assert fig.data[0].x[0] == "Before" and fig.data[0].x[-1] == "After"
    assert sum(fig.data[0].y[1:-1]) == pytest.approx(res["change"])


def test_cli_why(tmp_path):
    csv = tmp_path / "shop.csv"
    shop().to_csv(csv, index=False)
    out = runner.invoke(app, ["why", str(csv), "date", "--metric", "price", "--before", "2024-03-01,2024-05-01",
                              "--after", "2024-05-01,2024-07-01"])
    assert out.exit_code == 0 and "south" in out.output
    assert runner.invoke(app, ["why", str(csv), "date", "--metric", "price", "--agg", "median"]).exit_code == 1
    assert runner.invoke(app, ["why", str(csv), "date", "--metric", "price", "--before", "2024-03-01"]).exit_code == 1
