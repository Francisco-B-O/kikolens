import pandas as pd
import streamlit as st

from kikolens.analyze import geo
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Geography"
SUBTITLE = "Check coordinates, measure distances, find zones and see whether nearby places behave alike. No map tiles are downloaded."


def _run(key: str, signature, fn):
    if st.button("Run", key=f"{key}_go", type="primary"):
        try:
            st.session_state[key] = (signature, fn())
        except ValueError as e:
            st.session_state.pop(key, None)
            ui.note(str(e), "high")
    cached = st.session_state.get(key)
    return cached[1] if cached and cached[0] == signature else None


def _notes(res: dict) -> None:
    for note in res.get("notes", []):
        ui.note(note, "medium")


def _check(df, lat, lon) -> None:
    res = geo.check_coordinates(df, lat, lon)
    ui.note(res["summary"], "ok" if res["usable"] == res["rows"] else "accent")
    ui.stats([("Usable", f"{res['usable']:,}"), ("Missing", f"{res['missing']:,}"), ("Out of range", f"{res['out_of_range']:,}"), ("At 0, 0", f"{res['null_island']:,}")])
    _notes(res)
    if not res["usable"]:
        return
    usable = df[res["valid"]]
    numeric = [c for c in df.select_dtypes("number").columns if c not in (lat, lon)]
    colour = st.selectbox("Colour the points by", ["(none)"] + numeric + [c for c in df.columns if df[c].nunique() <= 12 and c not in numeric and c not in (lat, lon)], key="geo_colour")
    outlines = st.checkbox("Draw coastlines and countries", key="geo_outlines", help="Plotly loads these shapes from its server the first time: nothing about your data is sent, but it needs internet.")
    pts = usable[[lat, lon] + ([colour] if not colour.startswith("(") else [])].rename(columns={lat: "lat", lon: "lon"})
    sample = pts.sample(min(len(pts), 5000), random_state=0) if len(pts) > 5000 else pts
    ui.chart(geo.points_figure(sample, None if colour.startswith("(") else colour, "Where the points are", outlines), 480)
    if len(pts) > len(sample):
        st.caption(t("Showing a sample of 5,000 points."))


def _distances(df, lat, lon) -> None:
    mode = st.radio("Distance to", ["One place", "The nearest of a list of centres"], horizontal=True, key="geo_mode")
    if mode == "One place":
        c1, c2 = st.columns(2)
        plat = c1.number_input("Latitude", -90.0, 90.0, float(df[lat].median()), format="%.5f", key="geo_plat")
        plon = c2.number_input("Longitude", -180.0, 180.0, float(df[lon].median()), format="%.5f", key="geo_plon")
        target, extra, signature = (plat, plon), {}, ("point", plat, plon)
    else:
        upload = st.file_uploader("Centres (CSV) with latitude and longitude columns", type=["csv"], key="geo_centres")
        if upload is None:
            ui.note("Upload a CSV with one row per centre (a shop, a warehouse).", "low")
            return
        centres = pd.read_csv(upload)
        found = geo.detect_coordinates(centres)
        if not found:
            ui.note("No latitude and longitude columns were found in that file.", "high")
            return
        names = ["(none)"] + [c for c in centres.columns if c not in found[0]]
        name = st.selectbox("Name of each centre", names, key="geo_cname")
        target = centres
        extra = {"center_lat": found[0][0], "center_lon": found[0][1], "center_name": None if name.startswith("(") else name}
        signature = ("centres", upload.file_id, name)
    reach = st.number_input("Count the points within (km, 0 for no limit)", 0.0, 20000.0, 0.0, key="geo_reach")
    res = _run("geo_dist", (lat, lon, reach, signature), lambda: geo.distances(df, lat, lon, target, reach_km=reach or None, **extra))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    ui.table(res["table"].round(2))


def _zones(df, lat, lon) -> None:
    numeric = ["(none)"] + [c for c in df.select_dtypes("number").columns if c not in (lat, lon)]
    c1, c2, c3 = st.columns(3)
    value = c1.selectbox("Average this inside each zone", numeric, key="geo_zval")
    eps = c2.number_input("Largest gap between neighbours (km)", 0.1, 500.0, 5.0, key="geo_eps")
    min_points = int(c3.number_input("Fewest points in a zone", 2, 500, 10, key="geo_minp"))
    value = None if value.startswith("(") else value
    res = _run("geo_zones", (lat, lon, value, eps, min_points), lambda: geo.hotspots(df, lat, lon, value, eps, min_points))
    if res is None:
        return
    ui.note(res["summary"], "ok" if res["zones"] else "medium")
    _notes(res)
    ui.chart(res["figure"], 480)
    if len(res["table"]):
        ui.table(res["table"].round(3))


def _pattern(df, lat, lon) -> None:
    numeric = [c for c in df.select_dtypes("number").columns if c not in (lat, lon)]
    if not numeric:
        ui.note("This needs a numeric column to look for a pattern in.", "low")
        return
    c1, c2 = st.columns(2)
    value = c1.selectbox("Value", numeric, key="geo_sval")
    k = int(c2.slider("Neighbours per point", 3, 30, 8, key="geo_k"))
    res = _run("geo_spatial", (lat, lon, value, k), lambda: geo.autocorrelation(df, lat, lon, value, k))
    if res is None:
        return
    ui.note(res["summary"], "ok" if res["p_value"] < 0.05 else "accent")
    ui.stats([("Moran's I", f"{res['moran']:.3f}"), ("p-value", f"{res['p_value']:.3f}"), ("Hot spots", f"{res['hot']}"), ("Cold spots", f"{res['cold']}")])
    _notes(res)
    ui.chart(res["figure"], 480)
    if len(res["table"]):
        ui.table(res["table"].head(100).round(3))


def _grid(df, lat, lon) -> None:
    numeric = ["(none)"] + [c for c in df.select_dtypes("number").columns if c not in (lat, lon)]
    c1, c2, c3 = st.columns(3)
    value = c1.selectbox("Summarise", numeric, key="geo_gval")
    cell = c2.number_input("Cell size (km)", 0.5, 500.0, 10.0, key="geo_cell")
    agg = c3.selectbox("Calculation", ["mean", "sum", "median", "max", "min"], key="geo_agg")
    value = None if value.startswith("(") else value
    res = _run("geo_grid", (lat, lon, value, cell, agg), lambda: geo.grid_summary(df, lat, lon, value, cell, agg))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    ui.chart(res["figure"], 480)
    ui.table(res["table"].head(200).round(4))
    ui.download_csv(res["table"], "geo_grid.csv", "Download the cells (CSV)")


def render(df: pd.DataFrame) -> None:
    pairs = geo.detect_coordinates(df)
    numeric = list(df.select_dtypes("number").columns)
    if not pairs and len(numeric) < 2:
        ui.note("No latitude and longitude columns found. This section needs two numeric columns with coordinates in degrees.", "low")
        return
    options = [f"{a} / {b}" for a, b in pairs] or [f"{numeric[0]} / {numeric[1]}"]
    choice = st.selectbox("Latitude / longitude columns", options, key="geo_pair", help="Detected from the names and the range of the values.")
    lat, lon = choice.split(" / ")
    if not pairs:
        ui.note("These columns were not recognised as coordinates by name: check they really are latitude and longitude.", "medium")
    try:
        usable = df[geo.check_coordinates(df, lat, lon)["valid"]]
    except ValueError as e:
        ui.note(str(e), "high")
        return
    tabs = st.tabs(["Check and map", "Distances", "Zones", "Spatial pattern", "Grid"])
    with tabs[0]:
        _check(df, lat, lon)
    if usable.empty:
        return
    with tabs[1]:
        _distances(usable, lat, lon)
    with tabs[2]:
        _zones(usable, lat, lon)
    with tabs[3]:
        _pattern(usable, lat, lon)
    with tabs[4]:
        _grid(usable, lat, lon)
