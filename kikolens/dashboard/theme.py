"""Tema editorial: papel cálido, tinta oscura y un único acento. Sigue el tema claro/oscuro de Streamlit."""
import plotly.express as px
import streamlit as st

PALETTES = {
    "light": {
        "bg": "#faf8f3", "surface": "#ffffff", "ink": "#1c1b19", "muted": "#6b675f", "line": "#dedad0",
        "accent": "#9a3412", "on_accent": "#ffffff", "accent_soft": "#f6e7de", "high": "#b91c1c", "medium": "#c2791c", "low": "#8a857a",
        "ok": "#3f6212",
        "colorway": ["#9a3412", "#2f5d62", "#b8860b", "#5b6b8c", "#6b7a3d", "#8c5b6b"],
    },
    "dark": {
        "bg": "#171614", "surface": "#1f1e1b", "ink": "#ece9e1", "muted": "#a19c91", "line": "#36342f",
        "accent": "#e0865a", "on_accent": "#1a1109", "accent_soft": "#2c211b", "high": "#ef6b6b", "medium": "#e0a23a", "low": "#8f8a7f",
        "ok": "#a3c26b",
        "colorway": ["#e0865a", "#6fa8ad", "#d9b44a", "#8ea0c4", "#a3b46a", "#c48a9c"],
    },
}


def current_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # fuera de una sesión o en versiones antiguas
        return "light"


def palette() -> dict:
    return PALETTES[current_mode()]


def _css(p: dict) -> str:
    return f"""
:root {{
  --bg:{p['bg']}; --surface:{p['surface']}; --ink:{p['ink']}; --muted:{p['muted']}; --line:{p['line']};
  --accent:{p['accent']}; --on-accent:{p['on_accent']}; --accent-soft:{p['accent_soft']}; --high:{p['high']}; --medium:{p['medium']};
  --low:{p['low']}; --ok:{p['ok']};
  --serif: Georgia, 'Times New Roman', serif;
  --sans: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}}
.stApp {{ background:var(--bg); color:var(--ink); }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stSidebar"] {{ background:var(--surface); border-right:1px solid var(--line); }}
.block-container {{ max-width:1180px; padding-top:2.2rem; padding-bottom:5rem; }}
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4, .kl-head h1, .kl-empty h1 {{
  font-family:var(--serif) !important; font-weight:600; letter-spacing:-.01em; color:var(--ink) !important; }}
h4 {{ font-size:1.05rem; margin-top:1.4rem; }}
p, li, label, .stMarkdown {{ font-family:var(--sans); }}
[data-testid="stCaptionContainer"] {{ color:var(--muted); }}
hr {{ border-color:var(--line); }}

.kl-brand {{ font-family:var(--serif); font-size:1.45rem; font-weight:600; padding:.25rem 0 .1rem; }}
.kl-brand small {{ display:block; font:400 .74rem var(--sans); color:var(--muted); letter-spacing:.02em; margin-top:2px; }}
.kl-label {{ display:block; margin:1.4rem 0 .4rem; font:600 .7rem var(--sans); letter-spacing:.09em;
  text-transform:uppercase; color:var(--muted); }}
.kl-file {{ font:.82rem var(--sans); color:var(--muted); padding:6px 0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}

.kl-head {{ margin:0 0 1.6rem; padding-bottom:1rem; border-bottom:1px solid var(--line); }}
.kl-head h1 {{ font-size:2rem; margin:0; padding:0; }}
.kl-head p {{ margin:.25rem 0 0; color:var(--muted); font-size:.95rem; }}

.kl-stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:0; margin:0 0 1.5rem;
  border:1px solid var(--line); background:var(--surface); }}
.kl-stat {{ padding:14px 18px; border-right:1px solid var(--line); }}
.kl-stat:last-child {{ border-right:0; }}
.kl-stat b {{ display:block; font:600 1.65rem var(--serif); font-variant-numeric:tabular-nums; }}
.kl-stat span {{ font:600 .68rem var(--sans); letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }}
.kl-stat i {{ display:block; font:normal .78rem var(--sans); color:var(--muted); margin-top:2px; }}

.kl-note {{ padding:10px 14px; margin:.5rem 0; background:var(--surface); border:1px solid var(--line);
  border-left:3px solid var(--line); font:.9rem/1.5 var(--sans); }}
.kl-note.high {{ border-left-color:var(--high); }} .kl-note.medium {{ border-left-color:var(--medium); }}
.kl-note.low {{ border-left-color:var(--low); }} .kl-note.ok {{ border-left-color:var(--ok); }}
.kl-note.accent {{ border-left-color:var(--accent); }}
.kl-note b {{ font-weight:600; }} .kl-note small {{ display:block; color:var(--muted); margin-top:3px; }}

.kl-empty {{ padding:4rem 1rem; text-align:center; color:var(--muted); font-family:var(--sans); }}
.kl-empty h1 {{ font-size:2.8rem; margin-bottom:.4rem; padding:0; }}
.kl-steps {{ display:inline-block; text-align:left; margin-top:1.2rem; }}
.kl-steps li {{ margin:.35rem 0; }}

.stButton > button, .stDownloadButton > button {{ border-radius:2px; border:1px solid var(--line); font-family:var(--sans); }}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  background:var(--accent); border-color:var(--accent); color:var(--on-accent); }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color:var(--accent); color:var(--accent); }}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{ color:var(--on-accent); filter:brightness(1.08); }}
[data-baseweb="tab-list"] {{ gap:1.2rem; border-bottom:1px solid var(--line); }}
[data-baseweb="tab"] {{ font-family:var(--sans); padding-left:0; padding-right:0; }}
[data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--line); padding:10px 14px; }}
[data-testid="stExpander"] {{ border:1px solid var(--line); border-radius:2px; background:var(--surface); }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding:3px 0; }}

.kl-hero {{ display:flex; gap:28px; align-items:center; justify-content:space-between; flex-wrap:wrap; padding:26px 30px; margin:0 0 1.4rem;
  background:linear-gradient(135deg,var(--accent-soft),var(--surface) 70%); border:1px solid var(--line); border-radius:6px; }}
.kl-hero-main h2 {{ font:600 1.9rem var(--serif) !important; margin:0 0 4px !important; padding:0 !important; }}
.kl-hero-main p {{ margin:0 0 14px; color:var(--muted); max-width:46rem; }}
.kl-gauge {{ text-align:center; font-family:var(--sans); }}
.kl-gauge-label {{ font:600 .68rem var(--sans); letter-spacing:.08em; text-transform:uppercase; color:var(--muted); margin-top:2px; }}
.kl-gauge-cap {{ font-size:.78rem; color:var(--muted); }}
.kl-tiles {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:14px; margin:0 0 1.5rem; }}
.kl-tile {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:14px 16px; transition:border-color .15s, transform .15s; }}
.kl-tile:hover {{ border-color:var(--accent); transform:translateY(-1px); }}
.kl-tile span {{ font:600 .68rem var(--sans); letter-spacing:.08em; text-transform:uppercase; color:var(--muted); display:block;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.kl-tile b {{ display:block; font:600 1.5rem var(--serif); margin:2px 0; font-variant-numeric:tabular-nums; }}
.kl-tile i {{ display:block; font:normal .76rem var(--sans); color:var(--muted); }}
.kl-tile-spark {{ margin-top:8px; line-height:0; }}
.kl-chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.kl-chip {{ font:600 .72rem var(--sans); padding:3px 10px; border-radius:99px; border:1px solid var(--line); background:var(--surface); color:var(--muted); }}
.kl-chip.accent {{ background:var(--accent-soft); border-color:var(--accent); color:var(--accent); }}
.kl-chip.ok {{ border-color:var(--ok); color:var(--ok); }} .kl-chip.high {{ border-color:var(--high); color:var(--high); }}
.kl-chip.medium {{ border-color:var(--medium); color:var(--medium); }}
.kl-strip {{ display:flex; gap:2px; height:14px; margin:6px 0 14px; }}
.kl-miss {{ flex:1; border-radius:2px; }}
.kl-chapter {{ border-left:3px solid var(--accent); padding:2px 0 2px 16px; margin:1.6rem 0 .6rem; }}
.kl-chapter small {{ font:600 .68rem var(--sans); letter-spacing:.09em; text-transform:uppercase; color:var(--muted); }}
.kl-chapter h3 {{ font:600 1.3rem var(--serif) !important; margin:2px 0 !important; padding:0 !important; }}
.kl-demo {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin-top:1.4rem; text-align:left; }}
"""


def inject() -> None:
    p = palette()
    px.defaults.color_discrete_sequence = p["colorway"]  # Plotly Express fija el color de cada traza al crearla
    st.markdown(f"<style>{_css(p)}</style>", unsafe_allow_html=True)


def style_chart(fig, height: int = 420):
    """Aplica la paleta del tema a una figura de Plotly."""
    p = palette()
    axis = dict(automargin=True, title_standoff=8, gridcolor=p["line"], zerolinecolor=p["line"], linecolor=p["line"], tickfont=dict(color=p["muted"]))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=p["ink"], family="system-ui, sans-serif", size=12),
        title=dict(text=fig.layout.title.text or "", font=dict(family="Georgia, serif", size=15), x=0, xanchor="left"),
        xaxis=axis, yaxis=axis, colorway=p["colorway"], height=height,
        margin=dict(l=8, r=8, t=48, b=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=p["muted"])),
        hoverlabel=dict(bgcolor=p["surface"], bordercolor=p["line"], font=dict(family="system-ui, sans-serif", color=p["ink"])),
    )
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    fig.for_each_yaxis(lambda a: a.update(tickfont=dict(color=p["muted"]), linecolor=p["line"]))  # incluye ejes secundarios
    return fig
