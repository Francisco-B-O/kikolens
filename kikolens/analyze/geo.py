"""Análisis geográfico local, sin mapas de fondo que descargar: nada sale de este ordenador.

  * `detect_coordinates` / `check_coordinates`: encuentra las columnas de latitud y longitud y avisa de lo que huele mal (intercambiadas, fuera de rango,
    el punto 0,0 de los GPS sin señal, filas repetidas).
  * `distances`: kilómetros hasta un punto, o hasta el centro más cercano de una lista (tiendas, almacenes), con el reparto por área de influencia.
  * `hotspots`: agrupaciones de puntos próximos (DBSCAN con distancia sobre la esfera).
  * `autocorrelation`: ¿se parecen entre sí los valores cercanos? (I de Moran con prueba por permutaciones) y dónde están los focos calientes y fríos
    (Getis-Ord Gi*, con corrección por contrastes múltiples).
  * `grid_summary`: resumen por celdas de unos pocos kilómetros.
Las distancias son sobre una esfera (haversine), lo bastante buenas para decisiones de negocio y no para topografía."""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import sparse, stats
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0088
MAX_POINTS = 200_000
LAT_NAMES = re.compile(r"(^|_)(lat|latitude|latitud)($|_)", re.IGNORECASE)
LON_NAMES = re.compile(r"(^|_)(lon|lng|long|longitude|longitud)($|_)", re.IGNORECASE)


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distancia en km sobre la esfera entre pares de puntos (admite arrays)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── Coordenadas ──────────────────────────────────────────────────────────────
def detect_coordinates(df: pd.DataFrame) -> list:
    """Pares (latitud, longitud) probables: por el nombre de las columnas y porque los valores caben en -90..90 y -180..180."""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    lats = [c for c in numeric if LAT_NAMES.search(str(c))]
    lons = [c for c in numeric if LON_NAMES.search(str(c))]
    pairs = []
    for lat in lats:
        for lon in lons:
            a, b = df[lat].dropna(), df[lon].dropna()
            if len(a) and len(b) and a.abs().max() <= 90.0001 and b.abs().max() <= 180.0001:
                pairs.append((lat, lon))
            elif len(a) and len(b) and b.abs().max() <= 90.0001 and a.abs().max() <= 180.0001:
                pairs.append((lat, lon))                              # parecen intercambiadas: se propone igual y `check_coordinates` lo dice
    return pairs


def _points(df: pd.DataFrame, lat: str, lon: str, value: str | None = None) -> pd.DataFrame:
    for c in (lat, lon) + ((value,) if value else ()):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"'{c}' must be numeric.")
    frame = df[[lat, lon] + ([value] if value else [])].copy()
    frame.columns = ["lat", "lon"] + (["value"] if value else [])
    return frame


def check_coordinates(df: pd.DataFrame, lat: str, lon: str) -> dict:
    """Qué filas tienen coordenadas utilizables y qué problemas hay en el resto."""
    pts = _points(df, lat, lon)
    missing = pts["lat"].isna() | pts["lon"].isna()
    out_of_range = ~missing & ((pts["lat"].abs() > 90) | (pts["lon"].abs() > 180))
    null_island = ~missing & (pts["lat"].abs() < 1e-9) & (pts["lon"].abs() < 1e-9)
    valid = ~(missing | out_of_range | null_island)
    notes = []
    swapped = int((out_of_range & (pts["lat"].abs() <= 180) & (pts["lon"].abs() <= 90)).sum())
    if swapped:
        notes.append(f"{swapped:,} row(s) would be valid with latitude and longitude swapped: the columns '{lat}' and '{lon}' may be the other way round.")
    if null_island.any():
        notes.append(f"{int(null_island.sum()):,} row(s) are at 0, 0 (a GPS with no signal, not a place): they were left out.")
    if missing.any():
        notes.append(f"{int(missing.sum()):,} row(s) have no coordinates.")
    other = int(out_of_range.sum()) - swapped
    if other > 0:
        notes.append(f"{other:,} row(s) are outside the valid range (latitude -90 to 90, longitude -180 to 180).")
    good = pts[valid]
    repeated = int(good.duplicated().sum())
    if len(good) and repeated > 0.5 * len(good):
        notes.append(f"{repeated / len(good):.0%} of the usable points repeat another's exact position: coordinates may have been rounded or taken from a postal code or city centre.")
    if len(good) > 1 and (good["lat"].round(1).nunique() == 1 and good["lon"].round(1).nunique() == 1):
        notes.append("All points are within about 10 km of each other.")
    return {"valid": valid, "usable": int(valid.sum()), "rows": int(len(pts)), "missing": int(missing.sum()), "out_of_range": int(out_of_range.sum()), "null_island": int(null_island.sum()),
            "swapped": swapped, "repeated": repeated, "notes": notes,
            "extent": (float(good["lat"].min()), float(good["lat"].max()), float(good["lon"].min()), float(good["lon"].max())) if len(good) else None,
            "summary": f"{int(valid.sum()):,} of {len(pts):,} rows have usable coordinates."}


def _usable(df: pd.DataFrame, lat: str, lon: str, value: str | None = None, minimum: int = 3) -> pd.DataFrame:
    pts = _points(df, lat, lon, value)
    ok = check_coordinates(df, lat, lon)["valid"]
    pts = pts[ok]
    if value:
        pts = pts.dropna(subset=["value"])
    if len(pts) < minimum:
        raise ValueError(f"At least {minimum} rows with usable coordinates are needed (found {len(pts)}).")
    if len(pts) > MAX_POINTS:
        raise ValueError(f"At most {MAX_POINTS:,} points; summarise them on a grid first (grid_summary).")
    return pts


# ── Distancias y áreas de influencia ─────────────────────────────────────────
def distances(df: pd.DataFrame, lat: str, lon: str, centers: pd.DataFrame | tuple, center_lat: str | None = None, center_lon: str | None = None,
              center_name: str | None = None, reach_km: float | None = None) -> dict:
    """Kilómetros hasta un punto `(lat, lon)` o hasta el más cercano de una lista de centros, y cuántos puntos tiene cada centro como más cercano."""
    pts = _usable(df, lat, lon, None, 1)
    if isinstance(centers, tuple):
        table = pd.DataFrame({"name": ["point"], "lat": [centers[0]], "lon": [centers[1]]})
    else:
        if not (center_lat and center_lon):
            raise ValueError("Give the latitude and longitude columns of the centres.")
        c = _points(centers, center_lat, center_lon)
        c["name"] = centers[center_name].astype(str).to_numpy() if center_name else [f"centre {i + 1}" for i in range(len(c))]
        table = c.dropna(subset=["lat", "lon"])
        table = table[(table["lat"].abs() <= 90) & (table["lon"].abs() <= 180)]
    if table.empty:
        raise ValueError("There are no centres with valid coordinates.")
    tree = BallTree(np.radians(table[["lat", "lon"]].to_numpy()), metric="haversine")
    dist, idx = tree.query(np.radians(pts[["lat", "lon"]].to_numpy()), k=1)
    km = dist.ravel() * EARTH_KM
    nearest = table["name"].to_numpy()[idx.ravel()]
    result = pts.assign(km=km, nearest=nearest)
    share = pd.DataFrame({"centre": table["name"], "points": [int((nearest == n).sum()) for n in table["name"]]})
    share["share"] = share["points"] / len(result)
    share["mean km"] = [float(km[nearest == n].mean()) if (nearest == n).any() else np.nan for n in table["name"]]
    share["farthest km"] = [float(km[nearest == n].max()) if (nearest == n).any() else np.nan for n in table["name"]]
    share = share.sort_values("points", ascending=False, ignore_index=True)
    notes = []
    covered = None
    if reach_km is not None:
        covered = float((km <= reach_km).mean())
        if covered < 0.9:
            notes.append(f"{1 - covered:.0%} of the points are more than {reach_km:g} km from any centre.")
    if len(table) > 1 and share["share"].iloc[0] > 0.5:
        notes.append(f"'{share['centre'].iloc[0]}' is the closest centre for {share['share'].iloc[0]:.0%} of the points: the others cover little.")
    return {"points": result, "table": share, "median_km": float(np.median(km)), "p90_km": float(np.percentile(km, 90)), "covered": covered, "notes": notes,
            "summary": (f"The median point is {np.median(km):,.1f} km from its nearest centre and 90% are within {np.percentile(km, 90):,.1f} km"
                        + (f"; {covered:.0%} are within {reach_km:g} km." if covered is not None else "."))}


# ── Agrupaciones ─────────────────────────────────────────────────────────────
def hotspots(df: pd.DataFrame, lat: str, lon: str, value: str | None = None, eps_km: float = 5.0, min_points: int = 10) -> dict:
    """Zonas con muchos puntos próximos (DBSCAN): cada grupo con su centro, su tamaño y, si hay `value`, su media frente al resto."""
    if eps_km <= 0 or min_points < 2:
        raise ValueError("eps_km must be positive and min_points at least 2.")
    pts = _usable(df, lat, lon, value, min_points)
    labels = DBSCAN(eps=eps_km / EARTH_KM, min_samples=min_points, metric="haversine", algorithm="ball_tree").fit_predict(np.radians(pts[["lat", "lon"]].to_numpy()))
    pts = pts.assign(cluster=labels)
    rows = []
    for c in sorted(set(labels) - {-1}):
        part = pts[pts["cluster"] == c]
        rows.append({"cluster": c + 1, "points": len(part), "share": len(part) / len(pts), "centre lat": part["lat"].mean(), "centre lon": part["lon"].mean(),
                     "radius km": float(haversine(part["lat"].mean(), part["lon"].mean(), part["lat"].to_numpy(), part["lon"].to_numpy()).max()),
                     **({"mean value": part["value"].mean(), "vs the rest": part["value"].mean() - pts.loc[pts["cluster"] != c, "value"].mean()} if value else {})})
    table = pd.DataFrame(rows).sort_values("points", ascending=False, ignore_index=True) if rows else pd.DataFrame(columns=["cluster", "points", "share", "centre lat", "centre lon", "radius km"])
    noise = float((labels == -1).mean())
    notes = []
    if not rows:
        notes.append(f"No zone has {min_points} or more points within {eps_km:g} km of each other: try a larger radius or fewer points.")
    elif noise > 0.5:
        notes.append(f"{noise:.0%} of the points are outside every zone: the points are spread out; a larger radius would group more of them.")
    if len(rows) == 1 and noise < 0.05:
        notes.append("Everything falls in one zone: the radius may be too large to tell areas apart.")
    pts["cluster"] = np.where(labels == -1, 0, labels + 1)
    return {"table": table, "points": pts, "zones": len(rows), "outside": noise, "notes": notes, "figure": points_figure(pts, "cluster" if rows else None),
            "summary": (f"{len(rows)} zone(s) found; {1 - noise:.0%} of the points belong to one." if rows else "No dense zones found.")}


# ── Autocorrelación espacial ─────────────────────────────────────────────────
def _knn_weights(pts: pd.DataFrame, k: int) -> tuple:
    tree = BallTree(np.radians(pts[["lat", "lon"]].to_numpy()), metric="haversine")
    dist, idx = tree.query(np.radians(pts[["lat", "lon"]].to_numpy()), k=k + 1)
    n = len(pts)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    return sparse.csr_matrix((np.ones(n * k), (rows, cols)), shape=(n, n)), dist[:, 1:] * EARTH_KM


def autocorrelation(df: pd.DataFrame, lat: str, lon: str, value: str, k: int = 8, permutations: int = 999, seed: int = 0, alpha: float = 0.05) -> dict:
    """I de Moran global (¿valores parecidos cerca unos de otros?) y focos locales de Getis-Ord Gi* (dónde están los altos y los bajos), con FDR."""
    if not 2 <= k <= 50:
        raise ValueError("k must be between 2 and 50.")
    if not 99 <= permutations <= 9999:
        raise ValueError("permutations must be between 99 and 9999.")
    pts = _usable(df, lat, lon, value, max(k + 2, 10)).reset_index(drop=True)
    x = pts["value"].to_numpy(float)
    if np.ptp(x) == 0:
        raise ValueError("The value is the same everywhere.")
    n = len(pts)
    k = min(k, n - 1)
    W, neighbour_km = _knn_weights(pts, k)
    rowsum = np.asarray(W.sum(axis=1)).ravel()
    Wr = sparse.diags(1.0 / rowsum) @ W
    z = x - x.mean()
    denom = float(z @ z)
    moran = float(z @ (Wr @ z) / denom)
    rng = np.random.default_rng(seed)
    sims = []
    for start in range(0, permutations, 100):
        batch = min(100, permutations - start)
        shuffled = np.column_stack([rng.permutation(z) for _ in range(batch)])
        sims.append(np.einsum("ij,ij->j", shuffled, Wr @ shuffled) / denom)
    sims = np.concatenate(sims)
    p = float((1 + np.sum(np.abs(sims - sims.mean()) >= abs(moran - sims.mean()))) / (permutations + 1))
    # Gi*: vecinos más el propio punto, pesos binarios
    Wstar = (W + sparse.identity(n)).tocsr()
    wsum = np.asarray(Wstar.sum(axis=1)).ravel()
    s = x.std()
    gi_z = (np.asarray(Wstar @ x).ravel() - x.mean() * wsum) / (s * np.sqrt((n * wsum - wsum ** 2) / (n - 1)))
    p_local = 2 * stats.norm.sf(np.abs(gi_z))
    order = np.argsort(p_local)
    adjusted = np.empty(n)
    adjusted[order] = np.minimum.accumulate((p_local[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    klass = np.where(adjusted < alpha, np.where(gi_z > 0, "hot spot", "cold spot"), "not significant")
    local = pts.assign(z=gi_z, p_adjusted=adjusted, kind=klass)
    expected = -1 / (n - 1)
    verdict = ("clustered: similar values sit close together" if moran > expected and p < alpha else
               "dispersed: neighbours tend to differ" if moran < expected and p < alpha else "no clear spatial pattern")
    notes = []
    if (klass == "not significant").all():
        notes.append("No local hot or cold spots stand out once the many comparisons are accounted for.")
    if n < 30:
        notes.append("Fewer than 30 points: the tests have little power.")
    if np.median(neighbour_km) > 200:
        notes.append(f"The typical neighbour is {np.median(neighbour_km):,.0f} km away: the points are very sparse, so 'neighbour' means something loose.")
    hot, cold = int((klass == "hot spot").sum()), int((klass == "cold spot").sum())
    return {"moran": moran, "expected": float(expected), "p_value": p, "verdict": verdict, "local": local, "hot": hot, "cold": cold, "k": k, "permutations": permutations,
            "table": local[local["kind"] != "not significant"].sort_values("z", key=np.abs, ascending=False, ignore_index=True), "notes": notes,
            "figure": points_figure(local, "kind", title=f"Hot and cold spots of {value}"),
            "summary": f"Moran's I = {moran:.3f} (expected {expected:.3f} with no pattern, p = {p:.3f}): {verdict}. {hot} hot spot(s) and {cold} cold spot(s) after correcting for multiple tests."}


# ── Rejilla ──────────────────────────────────────────────────────────────────
def grid_summary(df: pd.DataFrame, lat: str, lon: str, value: str | None = None, cell_km: float = 10.0, agg: str = "mean") -> dict:
    """Resume los puntos en celdas cuadradas de unos `cell_km` kilómetros (la anchura en longitud se corrige con la latitud)."""
    if cell_km <= 0:
        raise ValueError("cell_km must be positive.")
    if agg not in ("mean", "sum", "median", "max", "min"):
        raise ValueError("agg must be mean, sum, median, max or min.")
    pts = _usable(df, lat, lon, value, 1)
    lat_step = cell_km / 111.195
    mid = float(pts["lat"].mean())
    lon_step = cell_km / (111.195 * max(np.cos(np.radians(mid)), 0.05))
    cells = pts.assign(row=np.floor(pts["lat"] / lat_step).astype(int), col=np.floor(pts["lon"] / lon_step).astype(int))
    grouped = cells.groupby(["row", "col"])
    table = grouped.size().rename("points").reset_index()
    if value:
        table["value"] = grouped["value"].agg(agg).to_numpy()
    table["lat"] = (table["row"] + 0.5) * lat_step
    table["lon"] = (table["col"] + 0.5) * lon_step
    table = table.sort_values("points", ascending=False, ignore_index=True)
    top = table.iloc[0]
    fig = go.Figure(go.Scatter(x=table["lon"], y=table["lat"], mode="markers", text=table["points"].astype(str) + " points",
                               marker=dict(symbol="square", size=10, color=table["value"] if value else table["points"], colorscale="YlOrRd", colorbar=dict(title=value or "points"),
                                           line=dict(width=0))))
    fig.update_layout(title=f"Cells of about {cell_km:g} km", xaxis_title="Longitude", yaxis_title="Latitude", yaxis=dict(scaleanchor="x", scaleratio=1 / max(np.cos(np.radians(mid)), 0.05)))
    return {"table": table, "cells": len(table), "cell_km": cell_km, "figure": fig, "notes": [],
            "summary": f"{len(table):,} cell(s) of about {cell_km:g} km; the busiest has {int(top['points']):,} point(s) around {top['lat']:.3f}, {top['lon']:.3f}."}


def points_figure(pts: pd.DataFrame, colour: str | None = None, title: str = "Points", outlines: bool = False) -> go.Figure:
    """Puntos sobre un globo sin mapa de fondo (no se descarga nada). Con `outlines` dibuja costas y países, que Plotly carga de su servidor."""
    fig = go.Figure()
    categorical = bool(colour) and colour in pts.columns and (not pd.api.types.is_numeric_dtype(pts[colour]) or pts[colour].nunique() <= 12)
    if categorical:
        for key, part in pts.groupby(colour):
            fig.add_trace(go.Scattergeo(lat=part["lat"], lon=part["lon"], mode="markers", name=str(key), marker=dict(size=5, opacity=0.8)))
    else:
        fig.add_trace(go.Scattergeo(lat=pts["lat"], lon=pts["lon"], mode="markers", marker=dict(size=5, opacity=0.8, color=pts[colour] if colour in pts.columns else None, colorscale="Viridis",
                                                                                                  showscale=colour in pts.columns)))
    fig.update_geos(fitbounds="locations", visible=outlines, showcountries=outlines, showcoastlines=outlines, showland=outlines, projection_type="natural earth")
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0), height=480)
    return fig
