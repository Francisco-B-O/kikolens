"""Piezas de interfaz reutilizables."""
import html

import streamlit as st

from kikolens.dashboard.theme import style_chart


def header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="kl-head"><h1>{html.escape(title)}</h1>{sub}</div>', unsafe_allow_html=True)


def stats(items) -> None:
    """items: iterable de (etiqueta, valor) o (etiqueta, valor, detalle)."""
    cells = []
    for item in items:
        label, value, *rest = item
        detail = f"<i>{html.escape(str(rest[0]))}</i>" if rest and rest[0] else ""
        cells.append(f'<div class="kl-stat"><span>{html.escape(str(label))}</span>'
                     f'<b>{html.escape(str(value))}</b>{detail}</div>')
    st.markdown(f'<div class="kl-stats">{"".join(cells)}</div>', unsafe_allow_html=True)


def note(message: str, kind: str = "", detail: str = "", bold: str = "") -> None:
    """Aviso con borde de color. `kind`: high, medium, low, ok o accent. Todo el texto se escapa."""
    lead = f"<b>{html.escape(bold)}</b> " if bold else ""
    extra = f"<small>{html.escape(detail)}</small>" if detail else ""
    st.markdown(f'<div class="kl-note {kind}">{lead}{html.escape(message)}{extra}</div>', unsafe_allow_html=True)


def label(text: str, sidebar: bool = False) -> None:
    target = st.sidebar if sidebar else st
    target.markdown(f'<span class="kl-label">{html.escape(text)}</span>', unsafe_allow_html=True)


def chart(fig, height: int = 420, key=None) -> None:
    st.plotly_chart(style_chart(fig, height), width="stretch", key=key)


def table(df, height=None, **kwargs) -> None:
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, width="stretch", **kwargs)


def download_csv(df, filename: str, label_text: str = "Download CSV") -> None:
    st.download_button(label_text, df.to_csv(index=False).encode("utf-8"), filename, "text/csv")
