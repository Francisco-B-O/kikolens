import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import geo

MADRID, BARCELONA, LONDON, PARIS = (40.4168, -3.7038), (41.3874, 2.1686), (51.5074, -0.1278), (48.8566, 2.3522)


def two_cities(n=600, seed=0):
    rng = np.random.default_rng(seed)
    centres = [(40.4, -3.7), (41.4, 2.2)]
    rows = [(centres[i % 2][0] + rng.normal(0, 0.1), centres[i % 2][1] + rng.normal(0, 0.1)) for i in range(n)]
    d = pd.DataFrame(rows, columns=["lat", "lon"])
    d["smooth"] = np.sin(d["lat"] * 20) + np.cos(d["lon"] * 20) + rng.normal(0, 0.1, n)
    d["noise"] = rng.normal(size=n)
    return d


# ── Distancias y coordenadas ─────────────────────────────────────────────────
def test_haversine_matches_known_distances():
    assert geo.haversine(*MADRID, *BARCELONA) == pytest.approx(505, abs=2)
    assert geo.haversine(*LONDON, *PARIS) == pytest.approx(344, abs=2)
    assert geo.haversine(0, 0, 1, 0) == pytest.approx(111.2, abs=0.1) and geo.haversine(0, 0, 0, 180) == pytest.approx(20015, abs=5)
    assert geo.haversine(10, 20, 10, 20) == 0
    assert np.allclose(geo.haversine(np.array([0, 0]), np.array([0, 0]), np.array([1, 2]), np.array([0, 0])), [111.19, 222.39], atol=0.1)


def test_coordinate_columns_are_detected_and_problems_named():
    df = pd.DataFrame({"latitude": [1.0, 2.0], "lng": [3.0, 4.0], "x": [5, 6], "lat_bad": [500.0, 600.0]})
    assert geo.detect_coordinates(df) == [("latitude", "lng")]
    swapped = pd.DataFrame({"lat": [100.0, 120.0], "lon": [40.0, 50.0]})
    assert geo.detect_coordinates(swapped) == [("lat", "lon")]
    messy = pd.DataFrame({"la": [1, 2, 0, 95, None, 10, 170], "lo": [1, 2, 0, 10, 3, 200, 40]})
    res = geo.check_coordinates(messy, "la", "lo")
    assert res["usable"] == 2 and res["null_island"] == 1 and res["missing"] == 1 and res["out_of_range"] == 3 and res["swapped"] == 2
    text = " ".join(res["notes"])
    assert "swapped" in text and "0, 0" in text and "no coordinates" in text and "outside the valid range" in text
    repeated = geo.check_coordinates(pd.DataFrame({"a": [40.0] * 10 + [41.0], "b": [3.0] * 10 + [4.0]}), "a", "b")
    assert any("repeat" in n for n in repeated["notes"])
    with pytest.raises(ValueError, match="not found"):
        geo.check_coordinates(messy, "nope", "lo")
    with pytest.raises(ValueError, match="numeric"):
        geo.check_coordinates(pd.DataFrame({"a": ["x"], "b": [1.0]}), "a", "b")


def test_distance_to_the_nearest_centre_and_catchments():
    clients = pd.DataFrame({"lat": [40.42, 40.38, 41.39, 41.41, 51.5], "lon": [-3.70, -3.72, 2.17, 2.20, -0.12]})
    shops = pd.DataFrame({"city": ["Madrid", "Barcelona"], "la": [*(MADRID[0], BARCELONA[0])], "lo": [MADRID[1], BARCELONA[1]]})
    res = geo.distances(clients, "lat", "lon", shops, "la", "lo", "city", reach_km=50)
    by = res["points"]["nearest"].tolist()
    assert by == ["Madrid", "Madrid", "Barcelona", "Barcelona", "Barcelona"]
    table = res["table"].set_index("centre")
    assert table.loc["Madrid", "points"] == 2 and table.loc["Barcelona", "points"] == 3 and table.loc["Barcelona", "farthest km"] > 1000
    assert res["covered"] == pytest.approx(0.8) and any("more than 50 km" in n for n in res["notes"]) and res["median_km"] < 10
    one = geo.distances(clients, "lat", "lon", MADRID)
    assert one["points"]["km"].iloc[0] == pytest.approx(geo.haversine(40.42, -3.70, *MADRID)) and len(one["table"]) == 1
    with pytest.raises(ValueError, match="latitude and longitude columns"):
        geo.distances(clients, "lat", "lon", shops)
    with pytest.raises(ValueError, match="no centres"):
        geo.distances(clients, "lat", "lon", pd.DataFrame({"la": [999.0], "lo": [999.0]}), "la", "lo")


# ── Agrupaciones ─────────────────────────────────────────────────────────────
def test_hotspots_find_the_two_cities():
    d = two_cities()
    res = geo.hotspots(d, "lat", "lon", "smooth", eps_km=10, min_points=10)
    t = res["table"]
    assert res["zones"] == 2 and sorted(t["points"]) == sorted(t["points"]) and t["points"].sum() > 560
    centres = sorted(zip(t["centre lat"].round(1), t["centre lon"].round(1)))
    assert centres == [(40.4, -3.7), (41.4, 2.2)] and {"mean value", "vs the rest"} <= set(t.columns) and len(res["figure"].data) >= 2
    sparse = geo.hotspots(d.sample(30, random_state=1), "lat", "lon", eps_km=1, min_points=10)
    assert sparse["zones"] == 0 and any("larger radius" in n for n in sparse["notes"])
    with pytest.raises(ValueError, match="eps_km"):
        geo.hotspots(d, "lat", "lon", eps_km=0)


# ── Autocorrelación ──────────────────────────────────────────────────────────
def test_morans_i_separates_a_spatial_pattern_from_noise():
    d = two_cities()
    smooth = geo.autocorrelation(d, "lat", "lon", "smooth")
    noise = geo.autocorrelation(d, "lat", "lon", "noise")
    assert smooth["moran"] > 0.6 and smooth["p_value"] < 0.01 and smooth["verdict"].startswith("clustered")
    assert abs(noise["moran"]) < 0.1 and noise["p_value"] > 0.05 and noise["verdict"] == "no clear spatial pattern" and noise["hot"] == 0 and noise["cold"] == 0
    assert any("No local hot" in n for n in noise["notes"]) and noise["expected"] == pytest.approx(-1 / 599)


def test_a_planted_hot_zone_is_found_where_it_is():
    rng = np.random.default_rng(2)
    d = pd.DataFrame({"lat": rng.uniform(40, 41, 800), "lon": rng.uniform(-4, -3, 800)})
    inside = (d["lat"] > 40.7) & (d["lon"] > -3.3)
    d["sales"] = rng.normal(100, 10, 800) + 40 * inside
    res = geo.autocorrelation(d, "lat", "lon", "sales", k=8)
    local = res["local"]
    hot = local[local["kind"] == "hot spot"]
    assert res["hot"] > 40 and (hot["lat"] > 40.65).mean() > 0.9 and (hot["lon"] > -3.35).mean() > 0.9
    assert res["cold"] == 0 or (local[local["kind"] == "cold spot"]["lat"] < 40.7).all()
    assert (local["p_adjusted"] >= 0).all() and (local["p_adjusted"] <= 1).all() and "Moran" in res["summary"]


def test_autocorrelation_validates_its_inputs():
    d = two_cities(100)
    for kwargs, fragment in (({"k": 1}, "k must"), ({"permutations": 5}, "permutations")):
        with pytest.raises(ValueError, match=fragment):
            geo.autocorrelation(d, "lat", "lon", "noise", **kwargs)
    with pytest.raises(ValueError, match="same everywhere"):
        geo.autocorrelation(d.assign(flat=1.0), "lat", "lon", "flat")
    with pytest.raises(ValueError, match="At least"):
        geo.autocorrelation(d.head(5), "lat", "lon", "noise")


# ── Rejilla y mapa ───────────────────────────────────────────────────────────
def test_grid_cells_keep_every_point_and_have_the_requested_size():
    d = two_cities()
    res = geo.grid_summary(d, "lat", "lon", "smooth", cell_km=20, agg="mean")
    assert res["table"]["points"].sum() == 600 and res["cells"] < 600 and {"value", "lat", "lon"} <= set(res["table"].columns)
    fine = geo.grid_summary(d, "lat", "lon", cell_km=2)
    assert fine["cells"] > res["cells"] and "value" not in fine["table"].columns
    assert geo.grid_summary(d, "lat", "lon", "smooth", 20, "sum")["table"]["value"].sum() == pytest.approx(d["smooth"].sum())
    for kwargs, fragment in (({"cell_km": 0}, "positive"), ({"agg": "mode"}, "agg")):
        with pytest.raises(ValueError, match=fragment):
            geo.grid_summary(d, "lat", "lon", **kwargs)


def test_maps_do_not_ask_for_outlines_unless_requested():
    d = two_cities(60).assign(group=lambda x: np.where(x["lat"] > 41, "north", "south"))
    plain = geo.points_figure(d, "group")
    assert plain.layout.geo.showcountries is False and plain.layout.geo.showcoastlines is False and len(plain.data) == 2
    assert geo.points_figure(d, "smooth", outlines=True).layout.geo.showcountries is True
    assert len(geo.points_figure(d).data) == 1


def test_python_api_and_terminal(tmp_path):
    import kikolens as kl
    from typer.testing import CliRunner

    from kikolens.cli.main import app
    d = two_cities(300)
    assert kl.geo_check(d)["usable"] == 300 and kl.geo_check(d)["pair"] == ("lat", "lon")
    assert kl.geo_distances(d, MADRID)["kind"] == "geo distances" and kl.geo_hotspots(d, eps_km=10)["zones"] == 2
    assert kl.spatial_autocorrelation(d, "smooth")["kind"] == "spatial autocorrelation" and kl.geo_grid(d, cell_km=20)["cells"] > 1
    csv = tmp_path / "g.csv"
    d.to_csv(csv, index=False)
    runner = CliRunner()
    for args, text in ((["geo-check", str(csv)], "usable coordinates"), (["geo-distances", str(csv), "--point", "40.4168,-3.7038"], "median point"),
                       (["geo-hotspots", str(csv), "--eps-km", "10"], "zone(s)"), (["spatial", str(csv), "smooth"], "Moran"), (["geo-grid", str(csv), "--cell-km", "20"], "cell(s)")):
        out = runner.invoke(app, args)
        assert out.exit_code == 0 and text in " ".join(out.output.split()), (args, out.output)
    assert runner.invoke(app, ["geo-check", str(tmp_path / "missing.csv")]).exit_code != 0
    assert runner.invoke(app, ["geo-distances", str(csv)]).exit_code == 1
