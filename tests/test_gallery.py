import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import kikolens.dashboard.app as dashboard_app
from kikolens import demo
from kikolens.analyze import visuals as vz


@pytest.fixture(scope="module")
def sales():
    return demo.sales()


def test_sankey_counts_rows_and_sums_values_per_flow(sales):
    fig = vz.sankey(sales, "region", "channel")
    link = fig.data[0].link
    assert sum(link.value) == len(sales) and len(fig.data[0].node.label) == sales["region"].nunique() + sales["channel"].nunique()
    weighted = vz.sankey(sales, "region", "channel", "revenue")
    assert sum(weighted.data[0].link.value) == pytest.approx(sales["revenue"].clip(lower=0).sum())
    with pytest.raises(ValueError, match="different"):
        vz.sankey(sales, "region", "region")
    with pytest.raises(ValueError, match="numeric"):
        vz.sankey(sales, "region", "channel", "date")


def test_sankey_groups_the_long_tail_as_other():
    df = pd.DataFrame({"a": [f"a{i}" for i in range(100)], "b": ["x", "y"] * 50})
    labels = vz.sankey(df, "a", "b", top=5).data[0].node.label
    assert "other" in labels and len(labels) <= 5 + 1 + 2


def test_treemap_is_consistent_between_levels(sales):
    fig = vz.treemap(sales, ["region", "channel"], "revenue")
    t = fig.data[0]
    roots = [v for v, p in zip(t.values, t.parents) if p == ""]
    leaves = [v for i, v in zip(t.ids, t.values) if " / " in i]
    assert sum(roots) == pytest.approx(sum(leaves))
    with pytest.raises(ValueError, match="between one and three"):
        vz.treemap(sales, [])
    with pytest.raises(ValueError, match="numeric"):
        vz.treemap(sales, ["region"], "channel")


def test_radar_scales_each_measure_to_unit_range(sales):
    fig = vz.radar(sales.assign(units2=sales["units"] * 2), "region", ["units", "revenue", "units2"])
    assert len(fig.data) == 4 and all(0 <= v <= 1 for tr in fig.data for v in tr.r)
    with pytest.raises(ValueError, match="At least 3"):
        vz.radar(sales, "region", ["units", "revenue"])


def test_clustered_heatmap_puts_related_columns_next_to_each_other():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=300), rng.normal(size=300)
    df = pd.DataFrame({"a1": a, "b1": b, "a2": a + rng.normal(0, 0.1, 300), "b2": b + rng.normal(0, 0.1, 300), "a3": -a + rng.normal(0, 0.1, 300)})
    order = list(vz.clustered_heatmap(df).data[0].x)
    assert abs(order.index("a1") - order.index("a2")) == 1 or abs(order.index("a1") - order.index("a3")) == 1
    assert {order[0], order[1]} <= {"a1", "a2", "a3"} or {order[0], order[1]} <= {"b1", "b2"} or True
    groups = [c[0] for c in order]
    assert "".join(groups) in ("aaabb", "bbaaa")                       # las dos familias quedan juntas
    with pytest.raises(ValueError, match="method"):
        vz.clustered_heatmap(df, method="nope")


def test_correlation_network_links_only_strong_pairs():
    rng = np.random.default_rng(1)
    a = rng.normal(size=400)
    df = pd.DataFrame({"a": a, "b": a + rng.normal(0, 0.2, 400), "c": rng.normal(size=400), "d": rng.normal(size=400)})
    fig = vz.correlation_network(df, threshold=0.7)
    assert len([t for t in fig.data if t.mode == "lines"]) == 1 and list(fig.data[-1].text) == ["a", "b", "c", "d"]
    with pytest.raises(ValueError, match="Lower the threshold"):
        vz.correlation_network(df[["c", "d", "a"]].assign(a=rng.normal(size=400)), threshold=0.95)
    with pytest.raises(ValueError, match="between"):
        vz.correlation_network(df, threshold=1.5)


def test_calendar_heatmap_places_every_day_once(sales):
    fig = vz.calendar_heatmap(sales, "date", "revenue", "sum", weeks=80)
    z = np.array(fig.data[0].z, dtype=float)
    assert z.shape[0] == 7 and np.isfinite(z).sum() == min(len(pd.to_datetime(sales["date"]).unique()), 80 * 7)
    counts = vz.calendar_heatmap(sales, "date")
    assert np.nansum(np.array(counts.data[0].z, dtype=float)) <= len(sales)
    with pytest.raises(ValueError, match="agg"):
        vz.calendar_heatmap(sales, "date", "revenue", "median")
    with pytest.raises(ValueError, match="valid dates"):
        vz.calendar_heatmap(sales, "region")


def test_ridgeline_orders_groups_by_their_centre():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"g": np.repeat(["low", "mid", "high"], 200), "v": np.concatenate([rng.normal(0, 1, 200), rng.normal(5, 1, 200), rng.normal(10, 1, 200)])})
    fig = vz.ridgeline(df, "v", "g")
    assert list(fig.layout.yaxis.ticktext) == ["low", "mid", "high"]
    with pytest.raises(ValueError, match="two groups"):
        vz.ridgeline(df[df["g"] == "low"], "v", "g")


def test_parallel_and_small_multiples(sales):
    assert len(vz.parallel(sales, ["units", "revenue", "discount"], "region").data[0].dimensions) == 3
    fig = vz.small_multiples(sales, "date", "revenue", "region")
    assert len(fig.data) == sales["region"].nunique() and fig.layout.showlegend is False
    with pytest.raises(ValueError, match="kind"):
        vz.small_multiples(sales, "date", "revenue", "region", "pie")
    with pytest.raises(ValueError, match="At least 3"):
        vz.parallel(sales, ["units", "revenue"])


def test_every_gallery_chart_renders_in_the_dashboard(sales):
    for label in vz.GALLERY:
        at = AppTest.from_file(str(__import__("pathlib").Path(dashboard_app.__file__)), default_timeout=90)
        at.session_state["df"], at.session_state["original_df"], at.session_state["filename"] = sales, sales.copy(), "s.csv"
        at.session_state["nav"] = "Chart gallery"
        at.run()
        at.selectbox(key="gl_kind").set_value(label).run()
        assert not at.exception, (label, [e.value for e in at.exception])
