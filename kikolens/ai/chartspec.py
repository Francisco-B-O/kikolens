"""Gráficos a partir de una descripción en lenguaje natural. El modelo no escribe código: devuelve una *especificación* en JSON
(tipo de gráfico, columnas, agregación, filtros) que aquí se valida contra los datos reales y se dibuja con funciones propias.
Si algo no cuadra (una columna que no existe, un filtro imposible) se explica el error con las columnas válidas, y esa frase es la
que se devuelve al modelo para que lo corrija."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

KINDS = ("bar", "line", "area", "scatter", "histogram", "box", "pie", "heatmap")
AGGS = ("sum", "mean", "median", "min", "max", "count", "nunique")
GRAINS = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
OPS = ("==", "!=", ">", ">=", "<", "<=", "in", "contains", "between")
MAX_FILTERS = 6
MAX_POINTS = 3000


class SpecError(ValueError):
    """La especificación no se puede dibujar con estos datos; el mensaje dice por qué y qué sí es válido."""


def _is_number(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _is_date(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def _column(df: pd.DataFrame, spec: dict, key: str, required: bool = False):
    value = spec.get(key)
    if value in (None, "", "none", "null"):
        if required:
            raise SpecError(f"'{key}' is required for this kind of chart. Columns: {', '.join(map(str, df.columns))}.")
        return None
    if value not in df.columns:
        close = [c for c in df.columns if str(c).lower() == str(value).lower()]
        if close:
            return close[0]
        raise SpecError(f"Column '{value}' does not exist. Columns: {', '.join(map(str, df.columns))}.")
    return value


def validate_spec(df: pd.DataFrame, spec) -> dict:
    """La especificación limpia (con valores por defecto) o SpecError."""
    if not isinstance(spec, dict):
        raise SpecError("The chart specification must be a JSON object.")
    kind = str(spec.get("kind", "")).lower()
    if kind not in KINDS:
        raise SpecError(f"'kind' must be one of {', '.join(KINDS)}.")
    clean = {"kind": kind, "title": str(spec.get("title") or "")[:120]}
    clean["x"] = _column(df, spec, "x", required=kind != "box")
    clean["y"] = _column(df, spec, "y", required=kind in ("scatter", "box", "heatmap"))
    clean["color"] = _column(df, spec, "color")
    clean["value"] = _column(df, spec, "value")
    agg = str(spec.get("agg") or ("count" if clean["y"] is None and clean["value"] is None else "sum")).lower()
    if agg not in AGGS:
        raise SpecError(f"'agg' must be one of {', '.join(AGGS)}.")
    clean["agg"] = agg
    grain = spec.get("grain")
    if grain not in (None, "", "none"):
        if str(grain).lower() not in GRAINS:
            raise SpecError(f"'grain' must be one of {', '.join(GRAINS)}.")
        clean["grain"] = str(grain).lower()
    if kind == "scatter" and not (_is_number(df[clean["x"]]) and _is_number(df[clean["y"]])):
        raise SpecError("A scatter plot needs two numeric columns for x and y.")
    if kind == "histogram" and not _is_number(df[clean["x"]]):
        raise SpecError("A histogram needs a numeric column for x.")
    if kind == "box" and not _is_number(df[clean["y"]]):
        raise SpecError("A box plot needs a numeric column for y (and, optionally, a category for x).")
    if kind in ("bar", "line", "area", "pie") and clean["y"] and agg != "count" and agg != "nunique" and not _is_number(df[clean["y"]]):
        raise SpecError(f"'{clean['y']}' is not numeric, so it cannot be summed or averaged. Use agg 'count' or 'nunique', or another column.")
    if kind == "heatmap":
        if clean["value"] is None and agg != "count":
            raise SpecError("A heatmap needs a numeric 'value' column, or agg 'count'.")
        if clean["value"] is not None and not _is_number(df[clean["value"]]):
            raise SpecError(f"'{clean['value']}' is not numeric.")
    top = spec.get("top")
    clean["top"] = None
    if top not in (None, ""):
        try:
            clean["top"] = max(1, min(int(top), 50))
        except (TypeError, ValueError):
            raise SpecError("'top' must be a whole number.") from None
    sort = str(spec.get("sort") or "").lower()
    if sort not in ("", "asc", "desc", "none"):
        raise SpecError("'sort' must be asc, desc or none.")
    clean["sort"] = sort or None
    filters = spec.get("filters") or []
    if not isinstance(filters, list) or len(filters) > MAX_FILTERS:
        raise SpecError(f"'filters' must be a list of at most {MAX_FILTERS} conditions.")
    clean["filters"] = [_filter(df, f) for f in filters]
    return clean


def _coerce(series: pd.Series, value):
    if _is_date(series):
        out = pd.to_datetime(value, errors="coerce")
        if pd.isna(out):
            raise SpecError(f"'{value}' is not a valid date.")
        return out
    if _is_number(series):
        try:
            return float(value)
        except (TypeError, ValueError):
            raise SpecError(f"'{value}' is not a number, but the column is numeric.") from None
    return str(value)


def _filter(df: pd.DataFrame, f) -> dict:
    if not isinstance(f, dict):
        raise SpecError("Each filter must be an object with column, op and value.")
    col = _column(df, f, "column", required=True)
    op = str(f.get("op", "")).strip()
    op = {"=": "==", "equals": "==", "is": "==", "not": "!=", "contains": "contains", "in": "in", "between": "between"}.get(op.lower(), op)
    if op not in OPS:
        raise SpecError(f"Filter operator must be one of {', '.join(OPS)}.")
    value = f.get("value")
    series = df[col]
    if op in ("in", "between"):
        if not isinstance(value, list) or not value or (op == "between" and len(value) != 2):
            raise SpecError(f"The value of an '{op}' filter must be a list" + (" of exactly two values." if op == "between" else "."))
        value = [_coerce(series, v) for v in value]
    elif op == "contains":
        value = str(value)
    else:
        value = _coerce(series, value)
    return {"column": col, "op": op, "value": value}


def apply_filters(df: pd.DataFrame, filters: list) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for f in filters:
        s, v = df[f["column"]], f["value"]
        if f["op"] == "==":
            mask &= s.astype(str).str.lower().eq(str(v).lower()) if isinstance(v, str) else s.eq(v)
        elif f["op"] == "!=":
            mask &= ~s.astype(str).str.lower().eq(str(v).lower()) if isinstance(v, str) else s.ne(v)
        elif f["op"] == ">":
            mask &= s.gt(v)
        elif f["op"] == ">=":
            mask &= s.ge(v)
        elif f["op"] == "<":
            mask &= s.lt(v)
        elif f["op"] == "<=":
            mask &= s.le(v)
        elif f["op"] == "in":
            mask &= s.astype(str).str.lower().isin([str(x).lower() for x in v]) if isinstance(v[0], str) else s.isin(v)
        elif f["op"] == "between":
            mask &= s.between(min(v), max(v))
        else:
            mask &= s.astype(str).str.contains(str(v), case=False, regex=False, na=False)
    out = df[mask]
    if out.empty:
        described = "; ".join(f"{f['column']} {f['op']} {f['value']}" for f in filters)
        raise SpecError(f"The filters leave no rows ({described}).")
    return out


def _aggregate(frame: pd.DataFrame, keys: list, value, agg: str) -> pd.DataFrame:
    if value is None or agg == "count":
        return frame.groupby(keys, dropna=False).size().reset_index(name="rows")
    out = frame.groupby(keys, dropna=False)[value].agg(agg).reset_index()
    return out.rename(columns={value: f"{agg} of {value}"})


def _with_grain(frame: pd.DataFrame, x: str, grain) -> pd.DataFrame:
    if grain and _is_date(frame[x]):
        frame = frame.copy()
        frame[x] = frame[x].dt.to_period(GRAINS[grain]).dt.start_time
    return frame


def render(df: pd.DataFrame, spec: dict) -> tuple:
    """(figura de Plotly, tabla con los datos dibujados, especificación limpia). Lanza SpecError si no se puede."""
    spec = validate_spec(df, spec)
    frame = apply_filters(df, spec["filters"]) if spec["filters"] else df
    kind, x, y, color, agg = spec["kind"], spec["x"], spec["y"], spec["color"], spec["agg"]
    title = spec["title"]
    fig = go.Figure()
    if kind == "scatter":
        data = frame[[c for c in dict.fromkeys([x, y, color]) if c]].dropna(subset=[x, y])
        data = data.sample(min(len(data), MAX_POINTS), random_state=0)
        _add_groups(fig, data, color, lambda part, name: go.Scatter(x=part[x], y=part[y], mode="markers", name=name, opacity=0.65, marker=dict(size=6)))
        fig.update_layout(xaxis_title=x, yaxis_title=y)
        table = data
    elif kind == "histogram":
        data = frame[[c for c in dict.fromkeys([x, color]) if c]].dropna(subset=[x])
        _add_groups(fig, data, color, lambda part, name: go.Histogram(x=part[x], name=name, opacity=0.7))
        fig.update_layout(barmode="overlay", xaxis_title=x, yaxis_title="rows")
        table = data.describe().T.reset_index() if color is None else data
    elif kind == "box":
        data = frame[[c for c in dict.fromkeys([x, y]) if c]].dropna(subset=[y])
        if x:
            levels = data[x].astype(str).value_counts().head(spec["top"] or 12).index
            data = data[data[x].astype(str).isin(levels)]
            for level in levels:
                fig.add_trace(go.Box(y=data.loc[data[x].astype(str) == level, y], name=str(level)))
        else:
            fig.add_trace(go.Box(y=data[y], name=y))
        fig.update_layout(showlegend=False, yaxis_title=y)
        table = data.groupby(x)[y].describe().reset_index() if x else data[y].describe().to_frame().T
    elif kind == "heatmap":
        value = spec["value"]
        pivot = frame.pivot_table(index=y, columns=x, values=value, aggfunc="count" if value is None or agg == "count" else agg) if value else \
            frame.groupby([y, x]).size().unstack(fill_value=0)
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index[: spec["top"] or 15], pivot.sum().sort_values(ascending=False).index[: spec["top"] or 15]]
        fig.add_trace(go.Heatmap(z=pivot.to_numpy(), x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index], colorscale="Oranges"))
        fig.update_layout(xaxis_title=x, yaxis_title=y)
        table = pivot.reset_index()
    else:                                                          # bar, line, area, pie
        data = _with_grain(frame, x, spec.get("grain"))
        keys = [x] + ([color] if color and color != x else [])
        value = y if y else None
        table = _aggregate(data, keys, value, agg)
        measure = table.columns[-1]
        ordering = spec["sort"] or ("asc" if kind in ("line", "area") else "desc")
        if kind in ("line", "area") or _is_date(table[x]):
            table = table.sort_values(keys)
        elif ordering != "none":
            table = table.sort_values(measure, ascending=ordering == "asc")
        if spec["top"] and kind in ("bar", "pie") and not _is_date(table[x]):
            leaders = table.groupby(x)[measure].sum().sort_values(ascending=False).head(spec["top"]).index
            table = table[table[x].isin(leaders)]
        if kind == "pie":
            shares = table.groupby(x)[measure].sum().sort_values(ascending=False)
            head, rest = shares.head(8), shares.iloc[8:].sum()
            labels = [str(i) for i in head.index] + (["other"] if rest > 0 else [])
            fig.add_trace(go.Pie(labels=labels, values=list(head.values) + ([rest] if rest > 0 else []), hole=0.35))
        else:
            maker = {"bar": lambda p, n: go.Bar(x=p[x].astype(str) if not _is_date(p[x]) else p[x], y=p[measure], name=n),
                     "line": lambda p, n: go.Scatter(x=p[x], y=p[measure], mode="lines+markers", name=n),
                     "area": lambda p, n: go.Scatter(x=p[x], y=p[measure], mode="lines", name=n, stackgroup="one")}[kind]
            _add_groups(fig, table, color if color != x else None, maker)
            fig.update_layout(xaxis_title=x, yaxis_title=measure, barmode="stack" if color else "group")
    fig.update_layout(title=title or _default_title(spec), showlegend=bool(color) or kind == "pie")
    return fig, table.reset_index(drop=True), spec


def _add_groups(fig: go.Figure, data: pd.DataFrame, color, maker) -> None:
    if color and color in data.columns:
        levels = data[color].astype(str).value_counts().head(10).index
        for level in levels:
            fig.add_trace(maker(data[data[color].astype(str) == level], str(level)))
    else:
        fig.add_trace(maker(data, ""))


def _default_title(spec: dict) -> str:
    parts = [spec["kind"].title()]
    if spec["y"] or spec["x"]:
        parts.append(" of " + " by ".join(str(p) for p in (spec["y"], spec["x"]) if p) if spec["y"] else f" of {spec['x']}")
    return "".join(parts)


def example_spec() -> dict:
    return {"kind": "bar", "x": "region", "y": "revenue", "agg": "sum", "top": 10, "filters": [{"column": "year", "op": ">=", "value": 2023}],
            "title": "Revenue by region"}


def numeric_summary(fig: go.Figure) -> str:
    """Un resumen en texto de lo que dibuja una figura (máximos, mínimos, tendencia), para pedirle al modelo que la explique sin enviarle datos en bruto."""
    lines = []
    for trace in fig.data[:6]:
        y = getattr(trace, "y", None)
        x = getattr(trace, "x", None)
        values = getattr(trace, "values", None)
        if values is not None:
            labels = list(getattr(trace, "labels", []))
            order = np.argsort(-np.asarray(values, float))[:3]
            lines.append("Shares: " + ", ".join(f"{labels[i]} {float(values[i]) / float(np.sum(values)):.0%}" for i in order))
        elif y is not None and len(y):
            arr = np.asarray(y, dtype=float) if np.issubdtype(np.asarray(y).dtype, np.number) else None
            if arr is None:
                continue
            name = trace.name or "series"
            lines.append(f"{name}: {len(arr)} points, from {arr[0]:.4g} to {arr[-1]:.4g}, minimum {arr.min():.4g} at {_at(x, arr.argmin())}, "
                         f"maximum {arr.max():.4g} at {_at(x, arr.argmax())}, mean {arr.mean():.4g}")
    return "\n".join(lines) or "The chart has no numeric series."


def _at(x, i):
    try:
        return str(list(x)[int(i)])
    except Exception:
        return "?"
