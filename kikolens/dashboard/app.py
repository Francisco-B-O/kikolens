import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
from pathlib import Path

from kikolens.analyze.stats import basic_statistics, correlation_matrix
from kikolens.sql.generator import generate_sql_from_text
from kikolens.analyze.automl import run_automl_analysis
from kikolens.ai.insights import get_ai_insights
from kikolens.utils.loaders import load_data
from kikolens.analyze.discovery import run_discovery_engine
from kikolens.sql.runner import run_sql_query
from kikolens.analyze.clustering import run_clustering_analysis

st.set_page_config(
    page_title="KikoLens",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  PALETA VERDE OSCURO — MODO OSCURO
# ─────────────────────────────────────────────────────────────────────────────
DARK_VARS = """
:root {
    --bg-app:              #050E08;
    --bg-card:             rgba(16,185,129,0.045);
    --bg-input:            rgba(16,185,129,0.06);
    --border:              rgba(16,185,129,0.11);
    --border-h:            rgba(16,185,129,0.3);
    --accent:              #10B981;
    --accent2:             #059669;
    --accent-light:        #34D399;
    --accent-dim:          rgba(16,185,129,0.14);
    --accent-glow:         rgba(52,211,153,0.35);
    --text:                #D1FAE5;
    --text-sec:            #4B9E78;
    --text-dim:            #1A4A34;
    --shadow:              rgba(0,0,0,0.55);
    --shadow-a:            rgba(16,185,129,0.18);
    --grad:                linear-gradient(135deg,#047857,#10B981);
    --grad-text:           linear-gradient(135deg,#6EE7B7 0%,#10B981 50%,#059669 100%);
    --nav-color:           #2D6E52;
    --nav-hover-bg:        rgba(16,185,129,0.09);
    --nav-hover-color:     #86EFAC;
    --nav-hover-border:    rgba(16,185,129,0.18);
    --nav-active-bg:       linear-gradient(135deg,rgba(16,185,129,0.22),rgba(5,150,105,0.13));
    --nav-active-color:    #6EE7B7;
    --nav-active-border:   rgba(16,185,129,0.32);
    --sidebar-bg:          linear-gradient(180deg,#071209 0%,#050E08 100%);
    --sidebar-border:      rgba(16,185,129,0.1);
    --tab-active:          #10B981;
    --scrollbar:           rgba(16,185,129,0.3);
    --metric-bar:          linear-gradient(90deg,#059669,#10B981);
    --chart-font:          #4B9E78;
    --chart-grid:          rgba(16,185,129,0.07);
    --toggle-label:        #4B9E78;
}
"""

# ─────────────────────────────────────────────────────────────────────────────
#  PALETA VERDE OSCURO — MODO CLARO
# ─────────────────────────────────────────────────────────────────────────────
LIGHT_VARS = """
:root {
    --bg-app:              #EDFDF3;
    --bg-card:             rgba(255,255,255,0.9);
    --bg-input:            rgba(255,255,255,0.95);
    --border:              rgba(5,150,105,0.2);
    --border-h:            rgba(5,150,105,0.45);
    --accent:              #059669;
    --accent2:             #047857;
    --accent-light:        #10B981;
    --accent-dim:          rgba(5,150,105,0.1);
    --accent-glow:         rgba(5,150,105,0.22);
    --text:                #052E16;
    --text-sec:            #14532D;
    --text-dim:            #1A6B3A;
    --shadow:              rgba(6,78,59,0.12);
    --shadow-a:            rgba(5,150,105,0.15);
    --grad:                linear-gradient(135deg,#047857,#059669);
    --grad-text:           linear-gradient(135deg,#047857 0%,#059669 50%,#10B981 100%);
    --nav-color:           #1A6B3A;
    --nav-hover-bg:        rgba(5,150,105,0.1);
    --nav-hover-color:     #052E16;
    --nav-hover-border:    rgba(5,150,105,0.25);
    --nav-active-bg:       linear-gradient(135deg,rgba(5,150,105,0.18),rgba(4,120,87,0.1));
    --nav-active-color:    #052E16;
    --nav-active-border:   rgba(5,150,105,0.35);
    --sidebar-bg:          linear-gradient(180deg,#C6F6D5 0%,#DCFCE7 100%);
    --sidebar-border:      rgba(5,150,105,0.25);
    --tab-active:          #059669;
    --scrollbar:           rgba(5,150,105,0.35);
    --metric-bar:          linear-gradient(90deg,#047857,#059669);
    --chart-font:          #14532D;
    --chart-grid:          rgba(5,150,105,0.14);
    --toggle-label:        #14532D;
}
"""

# ─────────────────────────────────────────────────────────────────────────────
#  CSS COMPARTIDO (usa variables)
# ─────────────────────────────────────────────────────────────────────────────
SHARED_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
* { box-sizing: border-box; }
#MainMenu                              { visibility: hidden; }
footer                                 { visibility: hidden; }
header                                 { visibility: hidden; }
[data-testid="stDeployButton"]         { display: none !important; }
[data-testid="stToolbar"]              { display: none !important; }
[data-testid="stDecoration"]           { display: none !important; }
[data-testid="stStatusWidget"]         { display: none !important; }
.stAppHeader                           { display: none !important; }
[data-testid="stHeader"]               { display: none !important; }

/* ── APP ── */
.stApp { background: var(--bg-app) !important; transition: background 0.45s ease; }
.main .block-container { padding: 1.75rem 2.5rem !important; max-width: 100% !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar            { width: 4px; height: 4px; }
::-webkit-scrollbar-track      { background: transparent; }
::-webkit-scrollbar-thumb      { background: var(--scrollbar); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover{ background: var(--accent); }

/* ── ANIMATIONS ── */
@keyframes fadeInUp  { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
@keyframes float     { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-7px)} }
@keyframes glowPulse { 0%,100%{filter:drop-shadow(0 0 12px var(--accent-glow))} 50%{filter:drop-shadow(0 0 26px var(--accent-glow))} }
@keyframes gradShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

/* ═══ SIDEBAR ═══ */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--sidebar-border) !important;
    transition: background 0.45s ease, border-color 0.45s ease;
}

/* Brand */
.kiko-brand {
    padding: 1.75rem 1.5rem 1.25rem;
    text-align: center;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.75rem;
}
.kiko-logo {
    font-size: 3rem; line-height: 1; margin-bottom: 0.55rem; display: block;
    animation: float 5s ease-in-out infinite, glowPulse 3s ease-in-out infinite;
}
.kiko-name {
    font-size: 1.2rem; font-weight: 900; letter-spacing: 6px; text-transform: uppercase;
    background: var(--grad-text); background-size: 200% 200%;
    animation: gradShift 4s ease infinite;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    display: block;
}
.kiko-tagline { font-size: 0.6rem; color: var(--text-dim); letter-spacing: 3px; text-transform: uppercase; margin-top: 0.3rem; display: block; }

/* Status */
.kiko-status { display:flex; align-items:center; gap:8px; padding:0.45rem 1.25rem 0; font-size:0.7rem; }
.status-dot  { width:6px; height:6px; border-radius:50%; background:var(--accent); box-shadow:0 0 7px var(--accent-glow); flex-shrink:0; }
.status-fname{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--text-sec); }

/* Nav label */
.nav-label { font-size:0.58rem; font-weight:700; letter-spacing:2.5px; text-transform:uppercase; color:var(--text-dim); padding:0 1.5rem; margin:1rem 0 0.3rem; display:block; }

/* Navigation radio → pills */
[data-testid="stSidebar"] [data-testid="stRadio"]      { padding:0 0.75rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] > div{ flex-direction:column; gap:2px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius:10px !important; padding:0.6rem 0.85rem !important;
    cursor:pointer !important; transition:all 0.18s ease !important;
    color:var(--nav-color) !important;
    font-size:0.875rem !important; font-weight:500 !important;
    border:1px solid transparent !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p { margin:0 !important; color:inherit !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background:var(--nav-hover-bg) !important;
    color:var(--nav-hover-color) !important;
    border-color:var(--nav-hover-border) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background:var(--nav-active-bg) !important;
    color:var(--nav-active-color) !important;
    border-color:var(--nav-active-border) !important;
    box-shadow:0 2px 10px var(--accent-dim) inset !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] { display:none !important; }

/* Toggle de tema */
[data-testid="stToggle"] label { color:var(--toggle-label) !important; font-size:0.8rem !important; font-weight:500 !important; }
[data-testid="stToggle"] [data-testid="stMarkdownContainer"] p { color:var(--toggle-label) !important; }

/* ═══ BUTTONS ═══ */
.stButton > button {
    background: var(--grad) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.55rem 1.25rem !important;
    font-weight: 600 !important; font-size: 0.875rem !important;
    letter-spacing: 0.2px !important; transition: all 0.2s ease !important;
    box-shadow: 0 4px 16px var(--shadow-a) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px var(--accent-glow) !important;
    filter: brightness(1.08) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ═══ INPUTS ═══ */
[data-testid="stTextInput"] > div > div > input,
[data-testid="stNumberInput"] > div > div > input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--text) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTextInput"] > div > div > input:focus,
[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
}

/* ═══ SELECTBOX / MULTISELECT ═══ */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--text) !important;
}
[data-testid="stMultiSelect"] > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ═══ TABS ═══ */
[data-testid="stTabsTabList"] { border-bottom: 1px solid var(--border) !important; gap:0 !important; }
button[data-testid="stTab"] {
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-sec) !important; font-weight: 500 !important;
    padding: 0.7rem 1.5rem !important; transition: all 0.2s ease !important; font-size: 0.875rem !important;
}
button[data-testid="stTab"]:hover            { color: var(--accent-light) !important; background: var(--accent-dim) !important; }
button[data-testid="stTab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}

/* ═══ SEGMENTED CONTROL (source mode) ═══ */
[data-testid="stSegmentedControl"] {
    background: var(--accent-dim) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 3px !important;
    width: 100% !important;
    gap: 2px !important;
}
[data-testid="stSegmentedControl"] button {
    background: transparent !important;
    border: none !important;
    border-radius: 9px !important;
    color: var(--text-sec) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 0 !important;
    flex: 1 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSegmentedControl"] button:hover {
    color: var(--accent) !important;
    background: var(--accent-dim) !important;
}
[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background: var(--accent) !important;
    color: white !important;
    box-shadow: 0 2px 8px var(--shadow-a) !important;
}

/* ═══ FILE UPLOADER ═══ */
.uploader-wrap {
    margin-top: 0.5rem;
    border-radius: 16px;
    overflow: hidden;
}
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--accent-dim) !important;
    border: 2px dashed var(--border-h) !important;
    border-radius: 16px !important;
    padding: 1.5rem 1rem !important;
    text-align: center !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: rgba(16,185,129,0.1) !important;
    box-shadow: 0 0 0 4px var(--accent-dim) !important;
}
[data-testid="stFileUploaderDropzone"] > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 0.4rem !important;
}
[data-testid="stFileUploaderDropzone"] span {
    color: var(--text-sec) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploaderDropzone"] small {
    color: var(--text-dim) !important;
    font-size: 0.68rem !important;
}
/* "Browse files" button inside uploader */
[data-testid="stFileUploaderDropzone"] button {
    background: var(--grad) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.4rem 1.1rem !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    margin-top: 0.35rem !important;
    box-shadow: 0 3px 10px var(--shadow-a) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    filter: brightness(1.1) !important;
    box-shadow: 0 5px 16px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}

/* ═══ DATAFRAME ═══ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important; overflow: hidden !important;
}

/* ═══ WIDGET LABELS ═══ */
[data-testid="stWidgetLabel"] {
    color: var(--text-sec) !important; font-size: 0.74rem !important;
    font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important;
}

/* ═══ ALERTS / INFO ═══ */
[data-testid="stAlert"]   { background: var(--accent-dim) !important; border: 1px solid var(--border-h) !important; border-radius: 12px !important; color: var(--text) !important; }
[data-testid="stSuccess"] { border-radius: 12px !important; }
[data-testid="stCode"]    { background: var(--bg-input) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
.stSpinner > div          { border-top-color: var(--accent) !important; }

/* ── TEXTO GLOBAL — overrides agresivos para ambos temas ── */
[data-testid="stCaptionContainer"] p                  { color: var(--text-sec) !important; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4               { color: var(--text) !important; }
h1,h2,h3,h4,h5,h6                                   { color: var(--text) !important; }
p, li                                                { color: var(--text) !important; }
/* Streamlit radio / checkbox labels */
[data-testid="stRadio"] label p,
[data-testid="stCheckbox"] label                     { color: var(--text-sec) !important; }
/* Info / success / error text */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span                         { color: var(--text) !important; }
/* Select / multiselect option text */
[data-testid="stSelectbox"] span,
[data-testid="stMultiSelect"] span                   { color: var(--text) !important; }

/* ── TRANSICIONES SUAVES ── */
.stApp,
[data-testid="stSidebar"],
.metric-card,
.glass-panel,
.page-hero,
.insight-card,
.welcome-feat,
.stButton > button,
[data-testid="stTextInput"] > div > div > input,
[data-testid="stNumberInput"] > div > div > input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div,
[data-testid="stFileUploader"],
[data-testid="stAlert"],
[data-testid="stDataFrame"],
button[data-testid="stTab"],
[data-testid="stTabsTabList"] {
    transition:
        background 0.45s ease,
        background-color 0.45s ease,
        border-color 0.45s ease,
        color 0.45s ease,
        box-shadow 0.45s ease !important;
}

/* ═══════════════════════════════════════════════════════════
   COMPONENTES HTML CUSTOM
═══════════════════════════════════════════════════════════ */

/* Metric grid */
.metric-grid {
    display:grid; grid-template-columns:repeat(3,1fr);
    gap:1rem; margin-bottom:1.5rem; animation:fadeInUp 0.45s ease;
}
.metric-card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:16px; padding:1.4rem 1.5rem;
    position:relative; overflow:hidden;
    transition:all 0.25s ease; cursor:default;
    box-shadow: 0 2px 12px var(--shadow);
}
.metric-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:var(--metric-bar); opacity:0; transition:opacity 0.25s ease;
}
.metric-card:hover { border-color:var(--border-h); transform:translateY(-3px); box-shadow:0 14px 40px var(--shadow); }
.metric-card:hover::before { opacity:1; }
.metric-icon  { font-size:1.5rem; margin-bottom:0.65rem; display:block; }
.metric-label { font-size:0.68rem; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:var(--text-dim); margin-bottom:0.4rem; }
.metric-value { font-size:2.1rem; font-weight:800; color:var(--text); line-height:1; letter-spacing:-1px; }
.metric-sub   { font-size:0.72rem; color:var(--text-dim); margin-top:0.35rem; }

/* Page hero */
.page-hero {
    display:flex; align-items:center; gap:1.5rem;
    margin-bottom:1.75rem; padding:1.4rem 1.75rem;
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:18px; animation:fadeInUp 0.4s ease;
    position:relative; overflow:hidden;
    box-shadow:0 2px 16px var(--shadow);
}
.page-hero::before {
    content:''; position:absolute; top:0; left:0; right:0; bottom:0;
    background:radial-gradient(ellipse at top right, var(--accent-dim) 0%, transparent 65%);
    pointer-events:none;
}
.page-hero-icon {
    width:54px; height:54px; border-radius:14px;
    background:var(--accent-dim); border:1px solid var(--border-h);
    display:flex; align-items:center; justify-content:center;
    font-size:1.7rem; flex-shrink:0;
}
.page-hero-title    { font-size:1.45rem; font-weight:700; color:var(--text); margin:0 0 0.25rem; line-height:1; }
.page-hero-subtitle { font-size:0.8rem; color:var(--text-sec); margin:0; line-height:1.5; }

/* Insight card */
.insight-card {
    background:var(--bg-card); border:1px solid var(--border); border-left:4px solid;
    border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:0.75rem;
    animation:fadeInUp 0.5s ease; transition:all 0.2s ease;
    box-shadow:0 2px 10px var(--shadow);
}
.insight-card:hover { transform:translateX(5px); border-color:var(--border-h); }
.insight-badge {
    display:inline-block; font-size:0.6rem; font-weight:700;
    letter-spacing:2px; text-transform:uppercase;
    padding:0.18rem 0.55rem; border-radius:100px;
    background:var(--accent-dim); margin-bottom:0.5rem;
}
.insight-msg    { font-size:0.95rem; font-weight:600; color:var(--text); margin-bottom:0.3rem; line-height:1.5; }
.insight-detail { font-size:0.78rem; color:var(--text-sec); line-height:1.6; }

/* Glass panel */
.glass-panel {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:16px; padding:1.4rem; margin-bottom:1rem;
    box-shadow:0 2px 12px var(--shadow);
}

/* Divider */
.kiko-divider { height:1px; background:linear-gradient(90deg,transparent,var(--border-h),transparent); margin:2rem 0; border:none; }

/* Welcome screen */
.welcome-screen {
    min-height:78vh; display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    text-align:center; padding:3rem 2rem;
}
.welcome-logo { font-size:5.5rem; line-height:1; margin-bottom:1.5rem; display:block; animation:float 4.5s ease-in-out infinite, glowPulse 3s ease-in-out infinite; }
.welcome-wordmark { font-size:5rem; font-weight:900; letter-spacing:-3px; color:var(--text); margin:0 0 0.5rem; line-height:1; }
.welcome-wordmark .g {
    background:var(--grad-text); background-size:200% 200%;
    animation:gradShift 4s ease infinite;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.welcome-tag { font-size:0.68rem; color:var(--text-dim); letter-spacing:5px; text-transform:uppercase; margin-bottom:3.5rem; }
.welcome-features { display:flex; gap:0.75rem; flex-wrap:wrap; justify-content:center; margin-bottom:3rem; max-width:720px; }
.welcome-feat {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:14px; padding:1rem 1.2rem; min-width:125px;
    transition:all 0.3s ease; cursor:default;
    box-shadow:0 2px 10px var(--shadow);
}
.welcome-feat:hover { border-color:var(--accent); background:var(--accent-dim); transform:translateY(-5px); box-shadow:0 12px 35px var(--shadow); }
.welcome-feat-icon { font-size:1.4rem; margin-bottom:0.4rem; display:block; }
.welcome-feat-name { font-size:0.68rem; font-weight:700; color:var(--text-sec); text-transform:uppercase; letter-spacing:1px; }
.welcome-hint { font-size:0.8rem; color:var(--text-sec); padding:0.65rem 1.75rem; border:1px dashed var(--border-h); border-radius:100px; background:var(--accent-dim); }
"""

# ─────────────────────────────────────────────────────────────────────────────
#  PALETA DE GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────
CHART_COLORS_DARK  = ["#10B981", "#34D399", "#059669", "#6EE7B7", "#047857", "#A7F3D0", "#D1FAE5", "#065F46"]
CHART_COLORS_LIGHT = ["#059669", "#047857", "#10B981", "#065F46", "#34D399", "#022C22", "#6EE7B7", "#166534"]

PAGE_META = {
    "Vision":    ("🔭", "#10B981"),
    "Discovery": ("🧬", "#10B981"),
    "Stats":     ("📐", "#10B981"),
    "Chat":      ("💬", "#10B981"),
    "Refinery":  ("⚗️",  "#10B981"),
    "AutoML":    ("🧠", "#10B981"),
    "Clusters":  ("✦",  "#10B981"),
}

PAGE_SUBTITLES = {
    "Vision":    "Explore and visualize your data interactively",
    "Discovery": "Automatic pattern detection & anomaly alerts",
    "Stats":     "Statistical deep-dive and correlation analysis",
    "Chat":      "Query your data in plain English using AI",
    "Refinery":  "Clean, transform and shape your dataset",
    "AutoML":    "Train predictive models in seconds",
    "Clusters":  "Discover natural groupings in your data",
}

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "df"                   not in st.session_state: st.session_state.df = None
if "filename"             not in st.session_state: st.session_state.filename = None
if "imported_model_ready" not in st.session_state: st.session_state.imported_model_ready = False
if "dark_mode"            not in st.session_state: st.session_state.dark_mode = True

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css(dark: bool):
    vars_block = DARK_VARS if dark else LIGHT_VARS
    st.markdown(f"<style>{vars_block}{SHARED_CSS}</style>", unsafe_allow_html=True)


def style_chart(fig, height=420):
    dark = st.session_state.get("dark_mode", True)
    font_color = "#4B9E78" if dark else "#166534"
    grid_color = "rgba(16,185,129,0.07)" if dark else "rgba(5,150,105,0.12)"
    colors = CHART_COLORS_DARK if dark else CHART_COLORS_LIGHT
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color, family="Inter", size=12),
        xaxis=dict(gridcolor=grid_color, showgrid=True, zeroline=False, tickcolor=font_color, linecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, showgrid=True, zeroline=False, tickcolor=font_color, linecolor=grid_color),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=grid_color, font=dict(color=font_color)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        colorway=colors,
    )
    return fig


def metric_html(icon, label, value, sub=None):
    sub_tag = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card">'
        f'  <span class="metric-icon">{icon}</span>'
        f'  <div class="metric-label">{label}</div>'
        f'  <div class="metric-value">{value}</div>'
        f'  {sub_tag}'
        f'</div>'
    )


def page_hero(name):
    icon, _ = PAGE_META.get(name, ("📊", "#10B981"))
    sub = PAGE_SUBTITLES.get(name, "")
    return (
        f'<div class="page-hero">'
        f'  <div class="page-hero-icon">{icon}</div>'
        f'  <div>'
        f'    <div class="page-hero-title">{name}</div>'
        f'    <div class="page-hero-subtitle">{sub}</div>'
        f'  </div>'
        f'</div>'
    )


def insight_card(ins):
    color = "#EF4444" if ins["severity"] == "high" else "#F59E0B" if ins["severity"] == "medium" else "#10B981"
    return (
        f'<div class="insight-card" style="border-left-color:{color}">'
        f'  <span class="insight-badge" style="color:{color}">{ins["type"]}</span>'
        f'  <div class="insight-msg">{ins["message"]}</div>'
        f'  <div class="insight-detail">{ins["detail"]}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    dark = st.session_state.dark_mode

    # ── SIDEBAR BRAND ──────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div class="kiko-brand">
        <span class="kiko-logo">⬡</span>
        <span class="kiko-name">KikoLens</span>
        <span class="kiko-tagline">Intelligence · Clarity · Precision</span>
    </div>""", unsafe_allow_html=True)

    # ── TOGGLE DARK/LIGHT ──────────────────────────────────────────────────
    st.sidebar.markdown('<span class="nav-label">Appearance</span>', unsafe_allow_html=True)
    new_dark = st.sidebar.toggle(
        "🌙 Dark mode" if dark else "☀️ Light mode",
        value=dark,
        key="dark_mode",
    )
    if new_dark != dark:
        st.rerun()

    # Inyectar CSS una vez conocemos el tema
    inject_css(new_dark)

    # ── DATA SOURCE ────────────────────────────────────────────────────────
    st.sidebar.markdown('<span class="nav-label">Data Source</span>', unsafe_allow_html=True)
    source_mode = st.sidebar.segmented_control(
        "", ["📁 File", "🗄️ Database"], key="s_mode",
        label_visibility="collapsed", default="📁 File",
    )
    source = None

    if source_mode == "📁 File":
        st.sidebar.markdown('<div class="uploader-wrap">', unsafe_allow_html=True)
        source = st.sidebar.file_uploader(
            "Drop your file here", type=["csv", "xlsx"], key="u",
            help="CSV or Excel — max 200 MB",
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    else:
        db_type = st.sidebar.selectbox("Engine", ["PostgreSQL", "MySQL", "SQLite"])
        if db_type == "SQLite":
            db_path = st.sidebar.text_input("Path to .db")
            # BUG A FIX: solo construir source si el path no está vacío
            if db_path.strip():
                source = f"sqlite:///{db_path.strip()}"
        else:
            host     = st.sidebar.text_input("Host", "localhost")
            port     = st.sidebar.text_input("Port", "5432" if db_type == "PostgreSQL" else "3306")
            user     = st.sidebar.text_input("User")
            password = st.sidebar.text_input("Password", type="password")
            database = st.sidebar.text_input("DB Name")
            driver   = "postgresql" if db_type == "PostgreSQL" else "mysql+pymysql"
            # BUG B FIX: solo construir source si todos los campos están rellenos
            if user.strip() and password.strip() and database.strip():
                source = f"{driver}://{user}:{password}@{host}:{port}/{database}"

    if source:
        current_name = source.name if hasattr(source, "name") else os.path.basename(str(source))
        if st.session_state.filename != current_name:
            try:
                with st.spinner("Indexing data…"):
                    st.session_state.df       = load_data(source)
                    st.session_state.filename = current_name
                    if "automl_res" in st.session_state:
                        del st.session_state.automl_res
                    st.session_state.imported_model_ready = False
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    # ── NAVIGATION ─────────────────────────────────────────────────────────
    if st.session_state.df is not None:
        df = st.session_state.df

        fname = st.session_state.filename or "dataset"
        st.sidebar.markdown(
            f'<div class="kiko-status">'
            f'  <span class="status-dot"></span>'
            f'  <span class="status-fname" title="{fname}">{fname}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.sidebar.markdown('<span class="nav-label" style="margin-top:1.25rem">Navigate</span>', unsafe_allow_html=True)
        page = st.sidebar.radio(
            "",
            ["🔭 Vision", "🧬 Discovery", "📐 Stats", "💬 Chat", "⚗️ Refinery", "🧠 AutoML", "✦ Clusters"],
            key="nav",
            label_visibility="collapsed",
        )
        page_key = page.split(" ", 1)[-1].strip()

        st.sidebar.markdown('<hr class="kiko-divider" style="margin:1.25rem 0.75rem">', unsafe_allow_html=True)
        st.sidebar.caption(f"**{len(df):,}** rows · **{len(df.columns)}** cols")

        # ── CONTENT ────────────────────────────────────────────────────────
        st.markdown(page_hero(page_key), unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════
        # VISION
        # ══════════════════════════════════════════════════════════════════
        if page_key == "Vision":
            health = (1 - df.isnull().sum().sum() / df.size) * 100
            nulls  = int(df.isnull().sum().sum())

            st.markdown(
                '<div class="metric-grid">'
                + metric_html("📦", "Total Rows", f"{len(df):,}", "records loaded")
                + metric_html("🧩", "Features",   f"{len(df.columns)}",  "columns detected")
                + metric_html(
                    "🩺", "Data Health", f"{health:.1f}%",
                    f"{nulls:,} missing values" if nulls else "No missing values",
                )
                + "</div>",
                unsafe_allow_html=True,
            )

            ctrl, plot_area = st.columns([1, 3])
            with ctrl:
                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                x         = st.selectbox("X Axis", df.columns, key="vx")
                y         = st.selectbox("Y Axis", df.columns, index=min(1, len(df.columns) - 1), key="vy")
                ctype     = st.radio("Chart Type", ["Scatter", "Bar", "Line", "Histogram", "Box"], key="vct")
                color_dim = st.selectbox("Color By", [None] + list(df.columns), key="vcd")
                st.markdown("</div>", unsafe_allow_html=True)

            with plot_area:
                kwargs = dict(color=color_dim)
                if ctype == "Scatter":     fig = px.scatter(df,   x=x, y=y, **kwargs)
                elif ctype == "Bar":       fig = px.bar(df,       x=x, y=y, **kwargs)
                elif ctype == "Line":      fig = px.line(df,      x=x, y=y, **kwargs)
                elif ctype == "Histogram": fig = px.histogram(df, x=x,      **kwargs)
                else:                      fig = px.box(df,       x=x, y=y, **kwargs)
                st.plotly_chart(style_chart(fig, height=460), use_container_width=True)

        # ══════════════════════════════════════════════════════════════════
        # DISCOVERY
        # ══════════════════════════════════════════════════════════════════
        elif page_key == "Discovery":
            with st.spinner("Running discovery engine…"):
                insights = run_discovery_engine(df)

            if not insights:
                st.info("No significant patterns or anomalies detected in this dataset.")
            else:
                st.caption(f"{len(insights)} insight{'s' if len(insights) != 1 else ''} found")
                col_a, col_b = st.columns(2)
                for i, ins in enumerate(insights):
                    with (col_a if i % 2 == 0 else col_b):
                        st.markdown(insight_card(ins), unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════
        # STATS
        # ══════════════════════════════════════════════════════════════════
        elif page_key == "Stats":
            tab_stats, tab_corr = st.tabs(["📊 Statistics", "🔗 Correlation"])

            with tab_stats:
                cmap = "YlGn" if not new_dark else "Greens"
                st.dataframe(
                    basic_statistics(df).style.background_gradient(cmap=cmap, axis=0).format(precision=2),
                    use_container_width=True,
                )
                if st.button("✨ Generate AI Insight"):
                    with st.spinner("Analyzing with Ollama…"):
                        st.info(get_ai_insights(df))

            with tab_corr:
                corr = correlation_matrix(df)
                if corr.empty:
                    st.info("No numeric columns available for correlation analysis.")
                else:
                    cscale = ["#EF4444", "#050E08" if new_dark else "#F0FDF4", "#10B981"]
                    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=cscale, zmin=-1, zmax=1)
                    fig.update_traces(textfont=dict(color="white" if new_dark else "#022C22", size=11))
                    st.plotly_chart(style_chart(fig, height=500), use_container_width=True)

        # ══════════════════════════════════════════════════════════════════
        # CHAT
        # ══════════════════════════════════════════════════════════════════
        elif page_key == "Chat":
            q = st.text_input(
                "Natural language query",
                placeholder="e.g. Total sales by category, top 10 customers by revenue…",
                key="chat_q",
            )
            if st.button("🚀 Run Query") and q:
                with st.spinner("Generating SQL…"):
                    sql = generate_sql_from_text(df, q)
                st.code(sql, language="sql")
                try:
                    result = run_sql_query(df, sql)
                    st.dataframe(result, use_container_width=True)
                except Exception as e:
                    st.error(f"Query error: {e}")

        # ══════════════════════════════════════════════════════════════════
        # REFINERY
        # ══════════════════════════════════════════════════════════════════
        elif page_key == "Refinery":
            c1, c2 = st.columns(2)

            with c1:
                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                st.markdown("#### 🗑️ Drop Columns")
                cols = st.multiselect("Select columns to remove", df.columns)
                if st.button("Remove Selected") and cols:
                    st.session_state.df = df.drop(columns=cols)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                st.markdown("#### 🩹 Fill Missing Values")
                null_cols = df.columns[df.isnull().any()].tolist()
                if null_cols:
                    t = st.selectbox("Column", null_cols)
                    # BUG C FIX: ofrecer solo estrategias válidas según el tipo de columna
                    is_num = pd.api.types.is_numeric_dtype(df[t])
                    strategies = ["Mean", "Median", "Mode", "Zero"] if is_num else ["Mode", "Constant"]
                    s = st.radio("Strategy", strategies, horizontal=True)
                    if st.button("Apply Fix"):
                        if   s == "Mean":   v = df[t].mean()
                        elif s == "Median": v = df[t].median()
                        elif s == "Mode":   v = df[t].mode()[0]
                        elif s == "Zero":   v = 0
                        else:               v = st.text_input("Fill value", "unknown")
                        st.session_state.df[t] = df[t].fillna(v)
                        st.rerun()
                else:
                    st.success("✅ No missing values in this dataset.")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<hr class="kiko-divider">', unsafe_allow_html=True)
            st.caption("Preview — first 10 rows")
            cmap_r = "Greens" if new_dark else "YlGn"
            st.dataframe(
                df.head(10).style.background_gradient(cmap=cmap_r, axis=None),
                use_container_width=True,
            )

        # ══════════════════════════════════════════════════════════════════
        # AUTOML
        # ══════════════════════════════════════════════════════════════════
        elif page_key == "AutoML":
            mode = st.radio("Mode", ["💎 Auto", "🛠️ Manual", "📤 Import"], horizontal=True, key="aml_mode")
            target_col = st.selectbox("Target Variable", df.columns, key="aml_target")

            manual_type = None
            imported_model = None

            if mode == "🛠️ Manual":
                manual_type = st.selectbox(
                    "Algorithm",
                    ["Random Forest", "Gradient Boosting", "Decision Tree", "Ridge/Logistic", "Lasso/KNN"],
                )
            elif mode == "📤 Import":
                up = st.file_uploader("Upload .pkl model", type=["pkl"])
                if up:
                    imported_model = joblib.load(up)
                    st.session_state.imported_model_ready = True
                    st.success("Model loaded successfully.")

            col_btn, col_opt = st.columns([1, 2])
            export_needed   = col_opt.checkbox("Export model after training")
            # BUG G FIX: separar el trigger de Import del botón de entrenamiento
            train_triggered  = col_btn.button("🚀 Train Pipeline")
            import_triggered = (mode == "📤 Import") and st.session_state.imported_model_ready and train_triggered

            if train_triggered:
                with st.spinner("Optimizing engine…"):
                    if import_triggered and imported_model:
                        res = run_automl_analysis(df, target_col, manual_type="Random Forest")
                        res["model"]     = imported_model
                        res["narrative"] = "### 📤 Custom Model Ready\nAnalysis based on imported pre-trained intelligence."
                    else:
                        res = run_automl_analysis(df, target_col, manual_type=manual_type)
                    st.session_state.automl_res = res

            if "automl_res" in st.session_state and st.session_state.automl_res["target"] == target_col:
                res = st.session_state.automl_res

                st.markdown(
                    '<div class="metric-grid">'
                    + metric_html("🎯", "Model Metrics", str(res["metrics"]))
                    + "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(res["narrative"])
                st.markdown('<hr class="kiko-divider">', unsafe_allow_html=True)
                st.markdown("#### Model Performance Dashboard")

                pc1, pc2 = st.columns(2)
                pc1.plotly_chart(style_chart(res["plots"]["performance"]), use_container_width=True)
                pc2.plotly_chart(style_chart(res["plots"]["importance"]),  use_container_width=True)

                if export_needed:
                    out_p = Path("./kikolens_output") / f"kikolens_{target_col}.pkl"
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(res["model"], out_p)
                    with open(out_p, "rb") as f:
                        st.download_button("💾 Download .pkl", f, f"{target_col}_model.pkl")

                st.markdown('<hr class="kiko-divider">', unsafe_allow_html=True)
                st.markdown("### 🔬 Explain & Simulate")
                t1, t2 = st.tabs(["🧩 Explain Row", "🎮 What-If Simulator"])

                with t1:
                    idx = st.number_input("Row Index", 0, len(df) - 1, 0)
                    if st.button("Explain Row Prediction"):
                        row  = df.iloc[idx]
                        data = []
                        for f, imp in res["feature_importance"].items():
                            bf = f.split("_")[0] if "_" in f else f
                            if bf in df.columns and pd.api.types.is_numeric_dtype(df[bf]):
                                data.append({"Factor": f, "Impact": (row[bf] - df[bf].mean()) * imp})
                        # BUG D FIX: comprobar que hay datos antes de renderizar el gráfico
                        if data:
                            ef = px.bar(pd.DataFrame(data).sort_values("Impact"), x="Impact", y="Factor", orientation="h")
                            st.plotly_chart(style_chart(ef), use_container_width=True)
                        else:
                            st.info("No numeric features available to explain this prediction.")

                with t2:
                    top_f = list(res["feature_importance"].keys())[:5]
                    # BUG E FIX: comprobar que hay features antes de crear columnas
                    if not top_f:
                        st.info("No features available for simulation.")
                    else:
                        sim_v    = {}
                        sim_cols = st.columns(len(top_f))
                        for i, f in enumerate(top_f):
                            bf = next((c for c in res["categorical_mappings"] if f.startswith(c)), None)
                            if bf:
                                sim_v[bf] = sim_cols[i].selectbox(f, res["categorical_mappings"][bf])
                            elif f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
                                cmin, cmax = float(df[f].min()), float(df[f].max())
                                # BUG F FIX: usar number_input si min == max (columna constante)
                                if cmin < cmax:
                                    sim_v[f] = sim_cols[i].slider(f, cmin, cmax, float(df[f].mean()))
                                else:
                                    sim_v[f] = sim_cols[i].number_input(f, value=cmin)
                        if st.button("🎯 Simulate"):
                            ir = pd.DataFrame([0] * len(res["columns"]), index=res["columns"]).T
                            for f, v in sim_v.items():
                                if f in ir.columns:
                                    ir[f] = v
                                else:
                                    d = f"{f}_{v}"
                                    if d in ir.columns:
                                        ir[d] = 1
                            pred = res["model"].predict(
                                res["scaler"].transform(res["imputer"].transform(ir))
                            )[0]
                            st.metric("Predicted Outcome", f"{pred:.4f}")

        # ══════════════════════════════════════════════════════════════════
        # CLUSTERS
        # ══════════════════════════════════════════════════════════════════
        elif page_key == "Clusters":
            col_s, col_btn = st.columns([3, 1])
            k = col_s.slider("Number of Segments", 2, 8, 3)
            with col_btn:
                st.write("")
                run_btn = st.button("▶ Run Clustering")
            if run_btn:
                with st.spinner("Segmenting data…"):
                    res = run_clustering_analysis(df, k)
                st.plotly_chart(style_chart(res["fig"], height=480), use_container_width=True)
                st.markdown('<hr class="kiko-divider">', unsafe_allow_html=True)
                st.caption("Cluster profiles")
                cmap_c = "Greens" if new_dark else "YlGn"
                st.dataframe(
                    res["profiles"].style.background_gradient(cmap=cmap_c),
                    use_container_width=True,
                )

    # ── WELCOME SCREEN ─────────────────────────────────────────────────────
    else:
        st.markdown("""
        <div class="welcome-screen">
            <span class="welcome-logo">⬡</span>
            <div class="welcome-wordmark">KIKO<span class="g">LENS</span></div>
            <div class="welcome-tag">Local AI · Data Intelligence</div>
            <div class="welcome-features">
                <div class="welcome-feat"><span class="welcome-feat-icon">🔭</span><div class="welcome-feat-name">Vision</div></div>
                <div class="welcome-feat"><span class="welcome-feat-icon">🧬</span><div class="welcome-feat-name">Discovery</div></div>
                <div class="welcome-feat"><span class="welcome-feat-icon">📐</span><div class="welcome-feat-name">Stats</div></div>
                <div class="welcome-feat"><span class="welcome-feat-icon">💬</span><div class="welcome-feat-name">Chat</div></div>
                <div class="welcome-feat"><span class="welcome-feat-icon">⚗️</span><div class="welcome-feat-name">Refinery</div></div>
                <div class="welcome-feat"><span class="welcome-feat-icon">🧠</span><div class="welcome-feat-name">AutoML</div></div>
                <div class="welcome-feat"><span class="welcome-feat-icon">✦</span><div class="welcome-feat-name">Clusters</div></div>
            </div>
            <div class="welcome-hint">Upload your data from the sidebar to begin</div>
        </div>
        """, unsafe_allow_html=True)


main()
