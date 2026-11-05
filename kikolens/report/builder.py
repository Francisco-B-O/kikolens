"""Informe HTML autocontenido (se puede imprimir a PDF desde el navegador)."""
import datetime as dt
import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from kikolens.analyze.discovery import run_discovery_engine
from kikolens.analyze.quality import profile_quality
from kikolens.analyze.stats import basic_statistics, correlation_matrix

STYLE = """
:root { --ink:#1c1b19; --muted:#6b675f; --line:#dedad0; --bg:#faf8f3; --accent:#9a3412; --card:#fff; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 Georgia,'Times New Roman',serif; }
main { max-width:920px; margin:0 auto; padding:56px 28px 80px; }
h1 { font-size:2.2rem; margin:0 0 4px; letter-spacing:-.01em; }
h2 { font-size:1.35rem; margin:3rem 0 .75rem; padding-top:1rem; border-top:1px solid var(--line); }
h3 { font-size:1rem; margin:1.5rem 0 .4rem; }
.meta, .muted { color:var(--muted); font-family:system-ui,sans-serif; font-size:.85rem; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:1.5rem 0; }
.kpi { background:var(--card); border:1px solid var(--line); padding:14px 16px; }
.kpi b { display:block; font-size:1.7rem; font-weight:600; }
.kpi span { color:var(--muted); font:.78rem system-ui,sans-serif; text-transform:uppercase; letter-spacing:.06em; }
table { border-collapse:collapse; width:100%; font:.84rem system-ui,sans-serif; margin:.5rem 0 1rem; background:var(--card); }
th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; background:#f3f0e8; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.issue { padding:8px 12px; margin:6px 0; background:var(--card); border-left:3px solid var(--line); font-family:system-ui,sans-serif; font-size:.88rem; }
.issue.high { border-left-color:#b91c1c; } .issue.medium { border-left-color:#c2791c; } .issue.low { border-left-color:#6b675f; }
.ai { white-space:pre-wrap; background:var(--card); border:1px solid var(--line); padding:14px 18px; }
footer { margin-top:4rem; color:var(--muted); font:.8rem system-ui,sans-serif; }
@media print { body { background:#fff; } main { padding:0; } h2 { break-after:avoid; } .chart, table { break-inside:avoid; } }
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _table(df: pd.DataFrame, index=True, max_rows=60) -> str:
    if df.empty:
        return '<p class="muted">No data.</p>'
    shown = df.head(max_rows)
    head = "".join(f"<th>{_esc(c)}</th>" for c in ([shown.index.name or ""] if index else []) + list(shown.columns))
    rows = []
    for idx, row in shown.iterrows():
        cells = [f"<td>{_esc(idx)}</td>"] if index else []
        for v in row:
            if isinstance(v, (int, float)) and not isinstance(v, bool) and pd.notna(v):
                cells.append(f'<td class="num">{v:,.4g}</td>')
            else:
                cells.append(f"<td>{'' if pd.isna(v) else _esc(v)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    note = f'<p class="muted">Showing {max_rows} of {len(df)} rows.</p>' if len(df) > max_rows else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>{note}"


class _Charts:
    """Inserta plotly.js una sola vez (inline para funcionar sin conexión) y el resto como divs."""

    def __init__(self, offline: bool):
        self.offline = offline
        self.first = True

    def render(self, fig: go.Figure) -> str:
        fig.update_layout(template="simple_white", font=dict(family="system-ui, sans-serif", size=12),
                          margin=dict(l=40, r=20, t=50, b=40), paper_bgcolor="#ffffff")
        include = (True if self.offline else "cdn") if self.first else False
        self.first = False
        return f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=include)}</div>'


def _model_section(result: dict, title: str, charts: _Charts) -> str:
    metrics = "".join(
        f'<div class="kpi"><b>{v:.4g}</b><span>{_esc(k)}</span></div>' for k, v in result["metrics"].items()
        if isinstance(v, (int, float)))
    importance = list(result.get("feature_importance", {}).items())[:10]
    imp_df = pd.DataFrame(importance, columns=["feature", "importance"]).set_index("feature")
    figs = "".join(charts.render(f) for key, f in result.get("plots", {}).items() if key in ("performance", "training"))
    narrative = _esc(result.get("narrative", "")).replace("###", "").replace("**", "").replace("`", "")
    board = result.get("leaderboard")
    leaderboard = ""
    if board is not None:
        cols = [c for c in board.columns if not c.endswith("std") and c != "total time (s)"]
        leaderboard = f"<h3>Model comparison (cross-validated)</h3>{_table(board[cols].round(4), index=False)}"
    return (f"<h2>{_esc(title)}</h2><p class='muted'>Target: {_esc(result.get('target', ''))} · "
            f"Task: {_esc(result.get('task_type', ''))}</p><div class='kpis'>{metrics}</div>"
            f"<div class='ai'>{narrative.strip()}</div>{leaderboard}<h3>Most influential features</h3>{_table(imp_df)}{figs}")


def build_report(df: pd.DataFrame, name: str = "dataset", models: dict | None = None,
                 ai_text: str | None = None, offline: bool = True) -> str:
    """Genera el informe completo. `models` mapea título -> resultado de run_automl_analysis/run_neural_network."""
    charts = _Charts(offline)
    quality = profile_quality(df)
    insights = run_discovery_engine(df) if len(df) else []
    corr = correlation_matrix(df)

    kpis = (
        f'<div class="kpi"><b>{len(df):,}</b><span>Rows</span></div>'
        f'<div class="kpi"><b>{len(df.columns)}</b><span>Columns</span></div>'
        f'<div class="kpi"><b>{quality["score"]:.0f}/100</b><span>Quality score</span></div>'
        f'<div class="kpi"><b>{quality["missing_pct"]:.1f}%</b><span>Missing cells</span></div>'
        f'<div class="kpi"><b>{quality["duplicate_rows"]:,}</b><span>Duplicate rows</span></div>'
    )
    issues = "".join(f'<div class="issue {_esc(i["severity"])}">{_esc(i["message"])}</div>' for i in quality["issues"]) \
        or '<p class="muted">No quality issues detected.</p>'

    parts = [
        f"<h1>{_esc(name)}</h1>",
        f'<div class="meta">Generated {dt.datetime.now():%Y-%m-%d %H:%M} · KikoLens</div>',
        f'<div class="kpis">{kpis}</div>',
        "<h2>Data quality</h2>", issues,
        "<h3>Column profile</h3>", _table(quality["columns"].set_index("column")),
        "<h2>Descriptive statistics</h2>", _table(basic_statistics(df).round(3)) if len(df.columns) else "",
    ]
    if not corr.empty and len(corr) > 1:
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
                                   colorscale="RdBu_r", text=corr.round(2).values, texttemplate="%{text}"))
        fig.update_layout(title="Correlation matrix", height=520)
        parts += ["<h2>Correlations</h2>", charts.render(fig)]
    if insights:
        parts += ["<h2>Patterns detected</h2>"] + [
            f'<div class="issue {_esc(i["severity"])}"><b>{_esc(i["type"])}</b> {_esc(i["message"])}<br>'
            f'<span class="muted">{_esc(i["detail"])}</span></div>' for i in insights]
    for title, result in (models or {}).items():
        parts.append(_model_section(result, title, charts))
    if ai_text:
        parts += ["<h2>AI commentary</h2>", f'<div class="ai">{_esc(ai_text)}</div>']
    parts.append("<footer>Produced locally. No data left this machine.</footer>")

    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{_esc(name)} · KikoLens report</title><style>{STYLE}</style></head>"
            f"<body><main>{''.join(parts)}</main></body></html>")


def save_report(df: pd.DataFrame, path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(df, name=kwargs.pop("name", path.stem), **kwargs), encoding="utf-8")
    return path
