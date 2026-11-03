"""Panel: un conjunto de tarjetas (métricas, gráficos, tablas) definidas por parámetros simples, con filtros
globales. Se guarda como JSON, se recalcula sobre cualquier dataset con las mismas columnas y se exporta a HTML."""
import datetime as dt
import json
import uuid
from dataclasses import dataclass, field

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from kikolens.analyze import transform
from kikolens.report.builder import STYLE, _Charts, _esc, _table

FORMAT_VERSION = 1
TILE_TYPES = ["metric", "chart", "table"]
CHART_KINDS = ["bar", "line", "area", "scatter", "histogram", "box", "pie"]
AGGS = ["sum", "mean", "median", "min", "max", "count", "nunique"]
MAX_TILES = 24
MAX_POINTS = 5000
MAX_CATEGORIES = 30


def _need(df: pd.DataFrame, *columns) -> None:
    for col in columns:
        if col and col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")


def _numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def apply_filters(df: pd.DataFrame, filters) -> pd.DataFrame:
    for f in filters or []:
        df, _ = transform.filter_rows(df, f.get("column"), f.get("op"), f.get("value"))
    return df


# ── Métricas ─────────────────────────────────────────────────────────────────
def compute_metric(df: pd.DataFrame, tile: dict):
    agg, col = tile.get("agg", "count"), tile.get("column")
    if agg not in AGGS:
        raise ValueError(f"Aggregation must be one of {AGGS}.")
    if agg == "count" and not col:
        return int(len(df))
    _need(df, col)
    if agg in ("count", "nunique"):
        return int(df[col].count() if agg == "count" else df[col].nunique())
    if not _numeric(df[col]):
        raise ValueError(f"'{col}' must be numeric for {agg}.")
    value = getattr(df[col], agg)()
    return None if pd.isna(value) else float(value)


# ── Gráficos ─────────────────────────────────────────────────────────────────
def _period(dates: pd.Series) -> str:
    span = (dates.max() - dates.min()).days if dates.notna().any() else 0
    return "D" if span <= 120 else "W" if span <= 900 else "MS"


def _grouped(df: pd.DataFrame, x: str, y: str | None, agg: str, top: int) -> pd.DataFrame:
    if agg not in AGGS:
        raise ValueError(f"Aggregation must be one of {AGGS}.")
    if y and agg not in ("count", "nunique") and not _numeric(df[y]):
        raise ValueError(f"'{y}' must be numeric for {agg}.")
    data = df
    if pd.api.types.is_datetime64_any_dtype(df[x]):
        data = df.assign(**{x: df[x].dt.to_period(_period(df[x])).dt.start_time})
    if y:
        out = data.groupby(x, dropna=True)[y].agg(agg).reset_index()
    else:
        out = data.groupby(x, dropna=True).size().reset_index(name="rows")
    return out


def build_figure(df: pd.DataFrame, tile: dict) -> go.Figure:
    kind = tile.get("kind", "bar")
    if kind not in CHART_KINDS:
        raise ValueError(f"Chart kind must be one of {CHART_KINDS}.")
    x, y, color = tile.get("x"), tile.get("y"), tile.get("color")
    agg, top = tile.get("agg", "sum"), int(tile.get("top", 15))
    _need(df, x, y, color)
    if not x:
        raise ValueError("Choose a column for the horizontal axis.")
    title = tile.get("title") or ""
    if kind == "histogram":
        if not _numeric(df[x]):
            raise ValueError(f"'{x}' must be numeric for a histogram.")
        fig = px.histogram(df, x=x, color=color if color and df[color].nunique() <= 8 else None, nbins=40)
    elif kind == "scatter":
        if not y:
            raise ValueError("A scatter chart needs a vertical column.")
        data = df if len(df) <= MAX_POINTS else df.sample(MAX_POINTS, random_state=0)
        fig = px.scatter(data, x=x, y=y, color=color if color and data[color].nunique() <= 12 else None, opacity=0.65)
    elif kind == "box":
        if not y or not _numeric(df[y]):
            raise ValueError("A box plot needs a numeric vertical column.")
        if df[x].nunique() > MAX_CATEGORIES:
            raise ValueError(f"'{x}' has too many groups for a box plot (limit {MAX_CATEGORIES}).")
        fig = px.box(df, x=x, y=y)
    else:
        data = _grouped(df, x, y, agg if y else "count", top)
        value = y or "rows"
        if kind == "pie":
            data = data.nlargest(min(top, 12), value)
            fig = px.pie(data, names=x, values=value, hole=0.45)
        elif kind == "bar":
            data = data.nlargest(top, value).sort_values(value)
            fig = px.bar(data, x=value, y=x, orientation="h")
        else:
            data = data.sort_values(x)
            fig = (px.line if kind == "line" else px.area)(data, x=x, y=value)
    if title:
        fig.update_layout(title=title)
    return fig


def build_table(df: pd.DataFrame, tile: dict) -> pd.DataFrame:
    group, columns = tile.get("group_by"), tile.get("columns") or []
    _need(df, *([group] if group else []), *columns, tile.get("sort"))
    n = max(1, min(int(tile.get("n", 10)), 500))
    if group:
        agg, value = tile.get("agg", "sum"), tile.get("value")
        if value:
            _need(df, value)
        out = _grouped(df, group, value, agg if value else "count", n)
    else:
        out = df[columns] if columns else df
    if tile.get("sort"):
        if tile["sort"] not in out.columns:
            raise ValueError(f"Cannot sort by '{tile['sort']}' here.")
        out = out.sort_values(tile["sort"], ascending=bool(tile.get("ascending", False)))
    return out.head(n).reset_index(drop=True)


# ── Panel ────────────────────────────────────────────────────────────────────
@dataclass
class Panel:
    title: str = "Panel"
    filters: list = field(default_factory=list)
    tiles: list = field(default_factory=list)

    def add(self, tile: dict) -> str:
        if tile.get("type") not in TILE_TYPES:
            raise ValueError(f"Tile type must be one of {TILE_TYPES}.")
        if len(self.tiles) >= MAX_TILES:
            raise ValueError(f"A panel can hold at most {MAX_TILES} tiles.")
        tile = {**tile, "id": tile.get("id") or uuid.uuid4().hex[:8]}
        self.tiles.append(tile)
        return tile["id"]

    def remove(self, tile_id: str) -> None:
        self.tiles = [t for t in self.tiles if t["id"] != tile_id]

    def move(self, tile_id: str, offset: int) -> None:
        ids = [t["id"] for t in self.tiles]
        if tile_id not in ids:
            return
        i, j = ids.index(tile_id), max(0, min(len(ids) - 1, ids.index(tile_id) + offset))
        self.tiles.insert(j, self.tiles.pop(i))

    def evaluate(self, df: pd.DataFrame):
        """Calcula cada tarjeta. Devuelve (datos filtrados, [(tarjeta, valor o None, error o None)])."""
        data = apply_filters(df, self.filters)
        results = []
        for tile in self.tiles:
            try:
                if tile["type"] == "metric":
                    value = compute_metric(data, tile)
                elif tile["type"] == "chart":
                    value = build_figure(data, tile)
                else:
                    value = build_table(data, tile)
                results.append((tile, value, None))
            except (ValueError, KeyError, TypeError) as e:
                results.append((tile, None, str(e)))
        return data, results

    def to_json(self) -> str:
        return json.dumps({"kikolens_panel": FORMAT_VERSION, "title": self.title, "filters": self.filters,
                           "tiles": self.tiles}, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, text: str) -> "Panel":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"The panel is not valid JSON: {e.msg}.") from None
        if not isinstance(raw, dict) or "tiles" not in raw:
            raise ValueError("This file is not a KikoLens panel.")
        if raw.get("kikolens_panel", 0) > FORMAT_VERSION:
            raise ValueError("This panel was made by a newer version of KikoLens.")
        panel = cls(str(raw.get("title") or "Panel"), list(raw.get("filters") or []))
        for tile in raw["tiles"]:
            if not isinstance(tile, dict):
                raise ValueError("Malformed tile in the panel.")
            panel.add(tile)
        return panel


def suggest_panel(df: pd.DataFrame, title: str = "Overview") -> Panel:
    """Panel inicial a partir de la forma de los datos: un conteo, la media de lo principal y unos gráficos."""
    panel = Panel(title)
    nums = [c for c in df.select_dtypes("number").columns if not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() > 5]
    cats = [c for c in df.columns if not _numeric(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])
            and 2 <= df[c].nunique() <= MAX_CATEGORIES]
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    panel.add({"type": "metric", "title": "Rows", "agg": "count"})
    for col in nums[:2]:
        panel.add({"type": "metric", "title": f"Average {col}", "agg": "mean", "column": col})
        panel.add({"type": "metric", "title": f"Total {col}", "agg": "sum", "column": col})
    if dates and nums:
        panel.add({"type": "chart", "kind": "line", "title": f"{nums[0]} over time", "x": dates[0], "y": nums[0], "agg": "sum"})
    if cats and nums:
        panel.add({"type": "chart", "kind": "bar", "title": f"{nums[0]} by {cats[0]}", "x": cats[0], "y": nums[0], "agg": "sum"})
    if len(cats) > 1:
        panel.add({"type": "chart", "kind": "pie", "title": f"Rows by {cats[1]}", "x": cats[1]})
    if nums:
        panel.add({"type": "chart", "kind": "histogram", "title": f"Distribution of {nums[-1]}", "x": nums[-1]})
    panel.add({"type": "table", "title": "First rows", "n": 8})
    return panel


def format_metric(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.4g}" if abs(value) < 1000 else f"{value:,.0f}"


def panel_html(df: pd.DataFrame, panel: Panel, offline: bool = True) -> str:
    """Página HTML autocontenida con las tarjetas calculadas (los datos no se incluyen, solo lo que se ve)."""
    data, results = panel.evaluate(df)
    charts = _Charts(offline)
    kpis, blocks = [], []
    for tile, value, error in results:
        title = _esc(tile.get("title") or tile["type"].title())
        if error:
            blocks.append(f'<div class="issue high"><b>{title}</b>: {_esc(error)}</div>')
        elif tile["type"] == "metric":
            kpis.append(f'<div class="kpi"><b>{_esc(format_metric(value))}</b><span>{title}</span></div>')
        elif tile["type"] == "chart":
            blocks.append(f'<div class="tile">{charts.render(value)}</div>')
        else:
            blocks.append(f'<div class="tile"><h3>{title}</h3>{_table(value.round(4), index=False)}</div>')
    filters = "".join(f"<li>{_esc(f['column'])} {_esc(f['op'])} {_esc(f.get('value') or '')}</li>" for f in panel.filters)
    extra = ".tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}.tile{min-width:0}main{max-width:1180px}"
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_esc(panel.title)} · KikoLens panel</title>'
            f"<style>{STYLE}{extra}</style></head><body><main><h1>{_esc(panel.title)}</h1>"
            f'<div class="meta">{len(data):,} of {len(df):,} rows · {dt.datetime.now():%Y-%m-%d %H:%M} · KikoLens</div>'
            + (f'<div class="meta">Filters:<ul>{filters}</ul></div>' if filters else "")
            + f'<div class="kpis">{"".join(kpis)}</div><div class="tiles">{"".join(blocks)}</div>'
            "<footer>Produced locally. No data left this machine.</footer></main></body></html>")
