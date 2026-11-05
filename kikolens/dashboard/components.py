"""Piezas de interfaz reutilizables."""
import html

import pandas as pd
import streamlit as st

from kikolens.dashboard.i18n import t
from kikolens.dashboard.theme import style_chart


def header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{html.escape(t(subtitle))}</p>" if subtitle else ""
    st.markdown(f'<div class="kl-head"><h1>{html.escape(t(title))}</h1>{sub}</div>', unsafe_allow_html=True)


def stats(items) -> None:
    """items: iterable de (etiqueta, valor) o (etiqueta, valor, detalle)."""
    cells = []
    for item in items:
        label, value, *rest = item
        detail = f"<i>{html.escape(str(rest[0]))}</i>" if rest and rest[0] else ""
        cells.append(f'<div class="kl-stat"><span>{html.escape(str(t(label)))}</span>'
                     f'<b>{html.escape(str(value))}</b>{detail}</div>')
    st.markdown(f'<div class="kl-stats">{"".join(cells)}</div>', unsafe_allow_html=True)


def note(message: str, kind: str = "", detail: str = "", bold: str = "") -> None:
    """Aviso con borde de color. `kind`: high, medium, low, ok o accent. Todo el texto se escapa."""
    lead = f"<b>{html.escape(t(bold))}</b> " if bold else ""
    extra = f"<small>{html.escape(t(detail))}</small>" if detail else ""
    st.markdown(f'<div class="kl-note {kind}">{lead}{html.escape(t(message))}{extra}</div>', unsafe_allow_html=True)


def label(text: str, sidebar: bool = False) -> None:
    target = st.sidebar if sidebar else st
    target.markdown(f'<span class="kl-label">{html.escape(t(text))}</span>', unsafe_allow_html=True)


def chart(fig, height: int = 420, key=None, select: bool = False):
    """Dibuja un gráfico. Con `select=True` se puede marcar con clic, recuadro o lazo y devuelve lo marcado."""
    if select:
        return st.plotly_chart(style_chart(fig, height), width="stretch", theme=None, key=key, on_select="rerun", selection_mode=("points", "box"))
    st.plotly_chart(style_chart(fig, height), width="stretch", theme=None, key=key)


def table(df, height=None, **kwargs) -> None:
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, width="stretch", **kwargs)


def heat_table(df, columns=None, cmap: str = "Oranges", **kwargs) -> None:
    """Tabla con las columnas numéricas coloreadas según su valor (más intenso, más alto). Si no se puede dar formato, se muestra tal cual."""
    cols = [c for c in (columns or df.columns) if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    try:
        styled = df.style.background_gradient(cmap=cmap, subset=cols, axis=0) if cols and len(df) > 1 else df
    except Exception:  # pragma: no cover - sin matplotlib o con tipos raros
        styled = df
    st.dataframe(styled, width="stretch", **kwargs)


def download_csv(df, filename: str, label_text: str = "Download CSV") -> None:
    st.download_button(label_text, df.to_csv(index=False).encode("utf-8"), filename, "text/csv")


def how_to_read(page: str) -> None:
    """Al final de una sección: qué mirar, de qué desconfiar y los términos que ayudan. Solo las secciones que tienen guía."""
    from kikolens.dashboard import i18n
    from kikolens.learn import glossary

    lang = i18n.current()
    found = glossary.guide(page, lang)
    if found is None:
        return
    with st.expander(t("How to read this page")):
        st.markdown("**" + t("What to look at") + "**")
        st.write(found["reads"])
        st.markdown("**" + t("Watch out") + "**")
        st.write(found["watch_out"])
        if found["terms"]:
            st.markdown("**" + t("Words that help") + "**")
            for entry in found["terms"]:
                st.markdown(f"* **{entry['term']}**: {entry['definition']}")
