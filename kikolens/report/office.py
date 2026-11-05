"""Exportación de un panel a PowerPoint y a Word. Las diapositivas llevan gráficos nativos (se pueden editar en PowerPoint) y el documento de Word,
imágenes. Se hace todo en este ordenador y los datos de origen no se incluyen: solo lo que se ve en el panel. Necesita `pip install kikolens[office]`."""
import datetime as dt
import io

import numpy as np
import pandas as pd

from kikolens.analyze import panel as pn

MAX_TABLE_ROWS = 14
MAX_TABLE_COLUMNS = 8
ACCENT = (0x2F, 0x6F, 0xB3)
INK = (0x1F, 0x29, 0x37)
MUTED = (0x6B, 0x72, 0x80)


def _need(module: str, package: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError:
        raise ImportError(f"Exporting to Office needs {package}: install it with  pip install kikolens[office]") from None


# ── Datos de cada gráfico, en una forma común para PowerPoint y Word ─────────
def chart_data(figure, tile: dict) -> dict:
    """Extrae del gráfico del panel lo necesario para redibujarlo: categorías y valores, puntos o una tabla."""
    kind = tile.get("kind", "bar")
    trace = figure.data[0]
    title = tile.get("title") or ""
    if kind == "pie":
        return {"shape": "categories", "kind": "pie", "categories": [str(v) for v in trace.labels], "values": [float(v) for v in trace.values], "title": title}
    if kind == "bar":
        return {"shape": "categories", "kind": "bar", "categories": [str(v) for v in trace.y], "values": [float(v) for v in trace.x], "title": title,
                "value_label": tile.get("y") or "rows"}
    if kind in ("line", "area"):
        x = list(trace.x)
        labels = [v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v) for v in x]
        return {"shape": "categories", "kind": kind, "categories": labels, "values": [float(v) for v in trace.y], "title": title, "value_label": tile.get("y") or "rows"}
    if kind == "histogram":
        values = np.asarray([v for t in figure.data for v in t.x], dtype=float)
        counts, edges = np.histogram(values, bins=min(20, max(5, int(np.sqrt(len(values))))))
        return {"shape": "categories", "kind": "histogram", "categories": [f"{(a + b) / 2:,.4g}" for a, b in zip(edges[:-1], edges[1:])], "values": counts.astype(float).tolist(),
                "title": title, "value_label": "rows"}
    if kind == "scatter":
        x = np.asarray(trace.x, dtype=float)
        y = np.asarray(trace.y, dtype=float)
        keep = ~(np.isnan(x) | np.isnan(y))
        return {"shape": "points", "kind": "scatter", "x": x[keep][:500].tolist(), "y": y[keep][:500].tolist(), "x_label": tile.get("x"), "y_label": tile.get("y"), "title": title}
    groups = {}                                                       # box: se resume con cuartiles
    for t in figure.data:
        for g, v in zip(t.x, t.y):
            groups.setdefault(str(g), []).append(float(v))
    rows = [{tile.get("x", "group"): g, "min": np.min(v), "25%": np.percentile(v, 25), "median": np.median(v), "75%": np.percentile(v, 75), "max": np.max(v), "rows": len(v)}
            for g, v in groups.items()]
    return {"shape": "table", "kind": "box", "table": pd.DataFrame(rows), "title": title}


def _evaluate(df: pd.DataFrame, panel: pn.Panel, cross: dict | None):
    data, results = panel.evaluate(df, cross)
    metrics, charts, tables, failed = [], [], [], []
    for tile, value, error in results:
        title = tile.get("title") or tile["type"].title()
        if error:
            failed.append((title, error))
        elif tile["type"] == "metric":
            metrics.append((title, pn.format_metric(value)))
        elif tile["type"] == "chart":
            charts.append((title, chart_data(value, tile)))
        else:
            tables.append((title, value))
    return data, metrics, charts, tables, failed


def _selection_text(panel: pn.Panel, cross: dict | None) -> list:
    lines = [f"{f['column']} {f['op']} {', '.join(map(str, f['value'])) if isinstance(f['value'], list) else f.get('value') or ''}".strip() for f in panel.filters]
    lines += [f"Selected in a chart: {pn.describe_filters(fs)}" for fs in (cross or {}).values() if fs]
    return lines


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.iloc[:MAX_TABLE_ROWS, :MAX_TABLE_COLUMNS].copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:,.4g}" if abs(v) < 1000 else f"{v:,.0f}")
    return out.astype(str)


# ── PowerPoint ───────────────────────────────────────────────────────────────
def panel_pptx(df: pd.DataFrame, panel: pn.Panel, cross: dict | None = None) -> bytes:
    """Una presentación: portada, cifras clave, una diapositiva por gráfico (nativo) y por tabla, y una última con lo que no se pudo calcular."""
    _need("pptx", "python-pptx")
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData, XyChartData
    from pptx.dml.color import RGBColor
    from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
    from pptx.util import Inches, Pt

    data, metrics, charts, tables, failed = _evaluate(df, panel, cross)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    colour = lambda c: RGBColor(*c)                                              # noqa: E731

    def text(slide, content, left, top, width, height, size=18, bold=False, color=INK):
        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        frame = box.text_frame
        frame.word_wrap = True
        for i, line in enumerate(content if isinstance(content, list) else [content]):
            para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            para.text = str(line)
            para.font.size, para.font.bold, para.font.color.rgb = Pt(size), bold, colour(color)
        return box

    def titled(title):
        slide = prs.slides.add_slide(blank)
        bar = slide.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.12))
        bar.fill.solid()
        bar.fill.fore_color.rgb = colour(ACCENT)
        bar.line.fill.background()
        text(slide, title, 0.6, 0.35, 12.1, 0.8, size=28, bold=True)
        text(slide, "Produced locally with KikoLens", 0.6, 7.0, 6, 0.3, size=10, color=MUTED)
        return slide

    cover = prs.slides.add_slide(blank)
    side = cover.shapes.add_shape(1, 0, 0, Inches(0.4), prs.slide_height)
    side.fill.solid()
    side.fill.fore_color.rgb = colour(ACCENT)
    side.line.fill.background()
    text(cover, panel.title, 1.0, 2.3, 11.3, 1.4, size=44, bold=True)
    text(cover, f"{len(data):,} of {len(df):,} rows  |  {dt.datetime.now():%Y-%m-%d}", 1.0, 3.9, 11.3, 0.5, size=18, color=MUTED)
    lines = _selection_text(panel, cross)
    if lines:
        text(cover, ["Filters:"] + lines[:6], 1.0, 4.7, 11.3, 1.8, size=14, color=MUTED)

    if metrics:
        slide = titled("Key figures")
        cols = 4 if len(metrics) > 6 else 3 if len(metrics) > 2 else len(metrics)
        for i, (label, value) in enumerate(metrics[:12]):
            row, col = divmod(i, cols)
            left, top = 0.6 + col * (12.1 / cols), 1.7 + row * 1.9
            card = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(12.1 / cols - 0.25), Inches(1.6))
            card.fill.solid()
            card.fill.fore_color.rgb = RGBColor(0xF3, 0xF6, 0xFA)
            card.line.color.rgb = RGBColor(0xD9, 0xE2, 0xEC)
            text(slide, value, left + 0.15, top + 0.2, 12.1 / cols - 0.5, 0.8, size=34, bold=True, color=ACCENT)
            text(slide, label, left + 0.15, top + 1.0, 12.1 / cols - 0.5, 0.5, size=14, color=MUTED)

    kinds = {"bar": XL_CHART_TYPE.BAR_CLUSTERED, "pie": XL_CHART_TYPE.PIE, "line": XL_CHART_TYPE.LINE, "area": XL_CHART_TYPE.AREA, "histogram": XL_CHART_TYPE.COLUMN_CLUSTERED}
    for title, spec in charts:
        slide = titled(title)
        if spec["shape"] == "table":
            _pptx_table(slide, spec["table"].round(4), Inches, Pt)
            continue
        if spec["shape"] == "points":
            chart_data_ = XyChartData()
            series = chart_data_.add_series(f"{spec['y_label']} against {spec['x_label']}")
            for a, b in zip(spec["x"], spec["y"]):
                series.add_data_point(a, b)
            graphic = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.4), chart_data_)
        else:
            chart_data_ = CategoryChartData()
            chart_data_.categories = spec["categories"]
            chart_data_.add_series(spec.get("value_label") or "value", spec["values"])
            graphic = slide.shapes.add_chart(kinds[spec["kind"]], Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.4), chart_data_)
        chart = graphic.chart
        chart.has_legend = spec["kind"] == "pie"
        if chart.has_legend:
            chart.legend.position, chart.legend.include_in_layout = XL_LEGEND_POSITION.RIGHT, False
        chart.font.size = Pt(12)
        if spec["kind"] != "pie":
            chart.plots[0].series[0].format.fill.solid()
            chart.plots[0].series[0].format.fill.fore_color.rgb = colour(ACCENT)
            if spec["kind"] in ("line", "scatter"):
                chart.plots[0].series[0].format.line.color.rgb = colour(ACCENT)
        if spec["kind"] == "bar":
            chart.category_axis.reverse_order = True                  # la barra más grande arriba, como en el panel

    for title, table in tables:
        slide = titled(title)
        _pptx_table(slide, table.round(4), Inches, Pt)

    if failed:
        slide = titled("Could not be calculated")
        text(slide, [f"{title}: {error}" for title, error in failed[:12]], 0.8, 1.6, 11.7, 5, size=16)
    buffer = io.BytesIO()
    prs.save(buffer)
    return buffer.getvalue()


def _pptx_table(slide, table: pd.DataFrame, Inches, Pt) -> None:
    from pptx.dml.color import RGBColor
    shown = _frame(table)
    if shown.empty:
        return
    rows, cols = len(shown) + 1, len(shown.columns)
    graphic = slide.shapes.add_table(rows, cols, Inches(0.8), Inches(1.5), Inches(11.7), Inches(0.4 * rows))
    grid = graphic.table
    for j, name in enumerate(shown.columns):
        grid.cell(0, j).text = str(name)
    for i, row in enumerate(shown.itertuples(index=False), start=1):
        for j, value in enumerate(row):
            grid.cell(i, j).text = str(value)
    for r in range(rows):
        for c in range(cols):
            for para in grid.cell(r, c).text_frame.paragraphs:
                para.font.size = Pt(12)
                if r == 0:
                    para.font.bold = True
                    para.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)


# ── Word ─────────────────────────────────────────────────────────────────────
def _png(spec: dict) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    colour = "#2F6FB3"
    kind = spec["kind"]
    if kind == "pie":
        ax.pie(spec["values"], labels=spec["categories"], autopct="%1.0f%%", startangle=90, wedgeprops={"width": 0.5})
    elif kind == "bar":
        ax.barh(spec["categories"][::-1], spec["values"][::-1], color=colour)
        ax.set_xlabel(spec.get("value_label") or "")
    elif kind in ("line", "area"):
        xs = range(len(spec["categories"]))
        ax.plot(xs, spec["values"], color=colour)
        if kind == "area":
            ax.fill_between(xs, spec["values"], alpha=0.25, color=colour)
        step = max(1, len(xs) // 8)
        ax.set_xticks(list(xs)[::step], [spec["categories"][i] for i in list(xs)[::step]], rotation=30, ha="right")
        ax.set_ylabel(spec.get("value_label") or "")
    elif kind == "histogram":
        ax.bar(range(len(spec["values"])), spec["values"], color=colour, width=0.9)
        step = max(1, len(spec["values"]) // 8)
        ax.set_xticks(range(0, len(spec["values"]), step), spec["categories"][::step], rotation=30, ha="right")
        ax.set_ylabel("rows")
    else:
        ax.scatter(spec["x"], spec["y"], s=10, alpha=0.6, color=colour)
        ax.set_xlabel(spec["x_label"] or "")
        ax.set_ylabel(spec["y_label"] or "")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def panel_docx(df: pd.DataFrame, panel: pn.Panel, cross: dict | None = None) -> bytes:
    """Un documento: título, filtros aplicados, cifras clave, cada gráfico como imagen con su título y cada tabla."""
    _need("docx", "python-docx")
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor

    data, metrics, charts, tables, failed = _evaluate(df, panel, cross)
    doc = Document()
    doc.core_properties.title = panel.title
    doc.core_properties.author = "KikoLens"
    doc.add_heading(panel.title, level=0)
    meta = doc.add_paragraph(f"{len(data):,} of {len(df):,} rows  |  {dt.datetime.now():%Y-%m-%d}")
    meta.runs[0].font.color.rgb = RGBColor(*MUTED)
    lines = _selection_text(panel, cross)
    if lines:
        doc.add_paragraph("Filters applied: " + "; ".join(lines))
    if metrics:
        doc.add_heading("Key figures", level=1)
        grid = doc.add_table(rows=0, cols=2)
        grid.style = "Light List Accent 1"
        for label, value in metrics:
            row = grid.add_row().cells
            row[0].text, row[1].text = label, value
    for title, spec in charts:
        doc.add_heading(title, level=1)
        if spec["shape"] == "table":
            _docx_table(doc, spec["table"].round(4))
        else:
            doc.add_picture(io.BytesIO(_png(spec)), width=Inches(6.2))
    for title, table in tables:
        doc.add_heading(title, level=1)
        _docx_table(doc, table.round(4))
    if failed:
        doc.add_heading("Could not be calculated", level=1)
        for title, error in failed:
            doc.add_paragraph(f"{title}: {error}", style="List Bullet")
    foot = doc.add_paragraph("Produced locally with KikoLens.")
    foot.runs[0].font.size = Pt(9)
    foot.runs[0].font.color.rgb = RGBColor(*MUTED)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _docx_table(doc, table: pd.DataFrame) -> None:
    shown = _frame(table)
    if shown.empty:
        return
    grid = doc.add_table(rows=1, cols=len(shown.columns))
    grid.style = "Light Grid Accent 1"
    for j, name in enumerate(shown.columns):
        grid.rows[0].cells[j].text = str(name)
    for row in shown.itertuples(index=False):
        cells = grid.add_row().cells
        for j, value in enumerate(row):
            cells[j].text = str(value)
