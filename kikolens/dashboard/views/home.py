import pandas as pd
import streamlit as st

from kikolens.analyze.autopilot import column_roles
from kikolens.analyze.discovery import run_discovery_engine
from kikolens.analyze.quality import infer_types, profile_quality
from kikolens.analyze.timeseries import detect_datetime_columns
from kikolens.dashboard import components as ui
from kikolens.dashboard import viz
from kikolens.dashboard.i18n import t

TITLE = "Home"
SUBTITLE = "Your data at a glance, and where to go next."

SECTION_GROUP = {"Autopilot": "Data", "Quality and cleaning": "Data", "Data story": "Data", "Explore": "Data", "Privacy": "Data",
                 "Model lab": "Models", "Regression": "Models", "Anomalies": "Analysis"}
ROLE_LABEL = {"number": "numbers", "category": "categories", "date": "dates", "identifier": "identifiers", "text": "free text",
              "constant": "constant"}
MAX_SPARKS = 8


def _go(section: str) -> None:
    st.session_state["nav"] = section
    st.session_state["nav_group"] = SECTION_GROUP[section]


def _trend(df: pd.DataFrame, typed: pd.DataFrame, roles: dict):
    dates = [c for c in detect_datetime_columns(typed)]
    numbers = [c for c, r in roles.items() if r == "number"]
    if not dates or not numbers:
        return None, None
    d, v = dates[0], numbers[-1]
    frame = pd.DataFrame({"d": pd.to_datetime(typed[d], errors="coerce"), "v": typed[v]}).dropna()
    if len(frame) < 10:
        return None, None
    return v, frame.set_index("d")["v"].resample("W").mean().dropna()


def render(df: pd.DataFrame) -> None:
    typed, _ = infer_types(df)
    roles = column_roles(typed)
    quality = profile_quality(typed)
    name = str(st.session_state.get("filename") or "dataset")

    kinds = pd.Series(roles).value_counts()
    chips = viz.chips([(f"{n} {t(ROLE_LABEL.get(k, k))}", "accent" if k == "number" else "") for k, n in kinds.items()])
    verdict = "healthy" if quality["score"] >= 90 else "needs a little cleaning" if quality["score"] >= 70 else "needs attention"
    gauge = viz.gauge(quality["score"], t("Quality"), caption=t(verdict))
    st.markdown(viz.hero(name, f"{len(df):,} rows, {df.shape[1]} columns, {quality['missing_pct']:.1f}% of cells empty.",
                         chips + viz.missing_strip(df), gauge), unsafe_allow_html=True)

    cols = st.columns(3)
    cols[0].button(t("Run the autopilot"), key="home_pilot", on_click=_go, args=("Autopilot",), type="primary", width="stretch")
    cols[1].button(t("Clean the data"), key="home_clean", on_click=_go, args=("Quality and cleaning",), width="stretch")
    cols[2].button(t("Tell me the story"), key="home_story", on_click=_go, args=("Data story",), width="stretch")

    st.markdown(f"#### {t('Numeric columns')}")
    numbers = [c for c, r in roles.items() if r == "number"][:MAX_SPARKS]
    items = []
    trend_name, trend = _trend(df, typed, roles)
    if trend is not None:
        items.append(viz.tile(f"{trend_name} {t('over time')}", f"{trend.iloc[-1]:,.4g}",
                              f"{(trend.iloc[-1] / trend.iloc[0] - 1):+.0%} {t('since the start')}" if trend.iloc[0] else "",
                              viz.sparkline(trend.to_numpy(), 170, 44)))
    for col in numbers:
        s = typed[col].dropna()
        items.append(viz.tile(col, f"{s.mean():,.4g}", f"{t('from')} {s.min():,.4g} {t('to')} {s.max():,.4g}", viz.histogram_spark(s, width=170, height=44)))
    if items:
        st.markdown(viz.tiles(items), unsafe_allow_html=True)
    else:
        st.caption(t("No numeric columns to summarise."))

    st.markdown(f"#### {t('Worth a look')}")
    try:
        findings = run_discovery_engine(typed)[:4]
    except Exception:  # pragma: no cover - defensivo
        findings = []
    if not quality["issues"] and not findings:
        ui.note(t("Nothing stands out. The data look clean."), "ok")
    for issue in quality["issues"][:3]:
        ui.note(issue["message"], issue["severity"] if issue["severity"] in ("high", "medium", "low") else "")
    for finding in findings:
        ui.note(finding["message"].replace("**", ""), "accent")
