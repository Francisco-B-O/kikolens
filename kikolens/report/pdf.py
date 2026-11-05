"""Informe en PDF generado directamente con matplotlib (sin navegador ni dependencias extra). Páginas: portada,
calidad, distribuciones, correlaciones y, si se pasan, resultados del modelo y de la previsión."""
import io
import re
import textwrap
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from kikolens.analyze.quality import profile_quality  # noqa: E402
from kikolens.utils.dtypes import categorical_columns  # noqa: E402

ACCENT, INK, MUTED, PAPER = "#9b3412", "#1c1a17", "#6a655c", "#f9f7f2"
PAGE = (8.27, 11.69)  # A4 en pulgadas


def _page(title: str, subtitle: str = ""):
    fig = plt.figure(figsize=PAGE, facecolor=PAPER)
    fig.text(0.08, 0.95, title, fontsize=20, family="serif", weight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.08, 0.915, subtitle, fontsize=10, color=MUTED, va="top")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.9, 0.9], color="#ddd8cb", lw=0.8))
    return fig


def _plain(text: str) -> str:
    """Quita la marca de Markdown (títulos y negritas) que traen los textos de los modelos."""
    return re.sub(r"^#+\s*|\*\*", "", text, flags=re.MULTILINE)


def _paragraphs(fig, blocks, top=0.87, width=88, size=10.5):
    y = top
    for block in map(_plain, blocks):
        for line in textwrap.wrap(block, width) or [""]:
            fig.text(0.08, y, line, fontsize=size, color=INK, va="top")
            y -= 0.022
        y -= 0.014
    return y


def _table(fig, frame: pd.DataFrame, rect, font=7.5, max_rows=22):
    ax = fig.add_axes(rect)
    ax.axis("off")
    shown = frame.head(max_rows)
    cells = [[_fmt(v) for v in row] for row in shown.itertuples(index=False, name=None)]
    if not cells:
        return
    n = len(shown.columns)
    table = ax.table(cellText=cells, colLabels=[str(c) for c in shown.columns], loc="upper center", cellLoc="left",
                     colWidths=[1 / n] * n)
    table.auto_set_font_size(False)
    table.set_fontsize(font)
    table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#ddd8cb")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#f0ede4")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor("white")


def _fmt(value) -> str:
    if isinstance(value, (float, np.floating)):
        return "" if pd.isna(value) else f"{value:,.4g}"
    text = str(value)
    return text if len(text) <= 28 else text[:27] + "..."


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b3a5")
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_facecolor(PAPER)


def _cover(pdf, df, name, quality, narrative):
    fig = _page("KikoLens report", f"{name}  -  {date.today():%d %B %Y}")
    stats = [("Rows", f"{len(df):,}"), ("Columns", f"{df.shape[1]}"), ("Quality", f"{quality['score']:.0f}/100"),
             ("Missing", f"{quality['missing_pct']:.1f}%")]
    for i, (label, value) in enumerate(stats):
        x = 0.08 + i * 0.22
        fig.text(x, 0.86, label.upper(), fontsize=8, color=MUTED, va="top")
        fig.text(x, 0.83, value, fontsize=22, family="serif", weight="bold", color=ACCENT, va="top")
    if narrative:
        _paragraphs(fig, [narrative], top=0.75, size=11)
    pdf.savefig(fig)
    plt.close(fig)


def _quality_page(pdf, quality):
    fig = _page("Data quality", f"Score {quality['score']:.0f} out of 100")
    issues = [f"[{i['severity']}] {i['message']}" for i in quality["issues"][:14]] or ["No problems were detected."]
    y = _paragraphs(fig, issues, top=0.87, size=9.5)
    columns = quality["columns"]
    keep = [c for c in ("column", "dtype", "missing_pct", "unique", "outliers") if c in columns.columns]
    _table(fig, columns[keep], (0.08, 0.05, 0.84, max(y - 0.08, 0.3)), max_rows=24)
    pdf.savefig(fig)
    plt.close(fig)


def _distributions(pdf, df):
    numeric = list(df.select_dtypes(include=[np.number]).columns)[:6]
    cats = [c for c in categorical_columns(df) if 1 < df[c].nunique() <= 15][:4]
    if not numeric and not cats:
        return
    fig = _page("Distributions", "Numeric columns as histograms, categories as counts")
    panels = [("num", c) for c in numeric] + [("cat", c) for c in cats]
    cols, rows = 2, 5
    for i, (kind, col) in enumerate(panels[:cols * rows]):
        ax = fig.add_axes([0.1 + (i % cols) * 0.44, 0.74 - (i // cols) * 0.17, 0.36, 0.11])
        _style(ax)
        if kind == "num":
            ax.hist(df[col].dropna(), bins=24, color=ACCENT, edgecolor=PAPER, linewidth=0.4)
        else:
            counts = df[col].value_counts().head(10)[::-1]
            ax.barh([str(k)[:18] for k in counts.index], counts.values, color=ACCENT)
        ax.set_title(str(col), fontsize=9, family="serif", loc="left", color=INK)
    pdf.savefig(fig)
    plt.close(fig)


def _correlations(pdf, df):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return
    numeric = numeric.loc[:, numeric.nunique() > 1].iloc[:, :14]
    if numeric.shape[1] < 2:
        return
    corr = numeric.corr()
    fig = _page("Correlations", "Pearson correlation between numeric columns")
    ax = fig.add_axes([0.22, 0.28, 0.62, 0.55])
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), [str(c)[:14] for c in corr.columns], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr)), [str(c)[:14] for c in corr.index], fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(corr.iat[i, j]) > 0.6 else INK)
    fig.colorbar(image, ax=ax, fraction=0.04)
    pdf.savefig(fig)
    plt.close(fig)


def _model_page(pdf, model):
    fig = _page("Model", f"Predicting {model['target']} ({model['task_type']})")
    y = _paragraphs(fig, [model.get("narrative", "")], top=0.87, size=9.5)
    board = model["leaderboard"]
    keep = [c for c in board.columns if not c.endswith("std") and "time" not in c and c != "status"][:6]
    _table(fig, board[keep], (0.08, max(y - 0.34, 0.3), 0.84, 0.3), max_rows=12)
    importance = list(model.get("feature_importance", {}).items())[:10][::-1]
    if importance:
        ax = fig.add_axes([0.28, 0.07, 0.62, 0.2])
        _style(ax)
        ax.barh([str(k)[:24] for k, _ in importance], [v for _, v in importance], color=ACCENT)
        ax.set_title("Permutation importance (top 10)", fontsize=9, family="serif", loc="left", color=INK)
    pdf.savefig(fig)
    plt.close(fig)


def _forecast_page(pdf, forecast):
    result = forecast["result"]
    fig = _page("Forecast", f"{forecast['value']} by {forecast['date']}")
    _paragraphs(fig, [result["summary"]], top=0.87, size=10)
    series, frame = forecast["series"], result["forecast"]
    ax = fig.add_axes([0.1, 0.55, 0.82, 0.25])
    _style(ax)
    recent = series.iloc[-120:]
    ax.plot(recent.index, recent.values, color=INK, lw=1.2, label="Observed")
    ax.plot(frame.index, frame["forecast"], color=ACCENT, lw=1.6, label="Forecast")
    ax.fill_between(frame.index, frame["lower"], frame["upper"], color=ACCENT, alpha=0.18, label="Range")
    ax.legend(fontsize=7, frameon=False)
    board = result["leaderboard"]
    _table(fig, board[[c for c in ("model", "MAE", "RMSE", "MAPE") if c in board.columns]], (0.08, 0.1, 0.84, 0.35))
    pdf.savefig(fig)
    plt.close(fig)


def build_pdf(df: pd.DataFrame, name: str = "dataset", autopilot: dict | None = None, model: dict | None = None) -> bytes:
    """El PDF como bytes. `autopilot` es el resultado de `run_autopilot`; `model` el de `run_model_lab` (si no, se usa el
    del piloto)."""
    if df is None or df.empty:
        raise ValueError("There is nothing to put in the report.")
    quality = profile_quality(df)
    model = model or (autopilot or {}).get("model")
    forecast = (autopilot or {}).get("forecast")
    buffer = io.BytesIO()
    with PdfPages(buffer, metadata={"Title": f"KikoLens report - {name}", "Creator": "KikoLens"}) as pdf:
        _cover(pdf, df, name, quality, (autopilot or {}).get("narrative", ""))
        _quality_page(pdf, quality)
        _distributions(pdf, df)
        _correlations(pdf, df)
        if model:
            _model_page(pdf, model)
        if forecast:
            _forecast_page(pdf, forecast)
    return buffer.getvalue()
