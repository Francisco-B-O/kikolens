"""Piezas visuales en HTML y SVG sin dependencias: anillo de puntuación, minigráficos, tarjetas y etiquetas. Todo el texto que viene
de los datos se escapa; los colores usan las variables CSS del tema, así que siguen el modo claro y oscuro."""
import html
import math

import numpy as np
import pandas as pd


def esc(value) -> str:
    return html.escape(str(value))


def gauge(value: float, label: str = "", maximum: float = 100.0, size: int = 132, caption: str = "") -> str:
    """Anillo con el valor al centro. El color pasa de rojo a verde según la proporción."""
    ratio = 0.0 if not maximum else max(0.0, min(1.0, float(value) / float(maximum)))
    radius = size / 2 - 10
    circumference = 2 * math.pi * radius
    colour = "var(--high)" if ratio < 0.5 else "var(--medium)" if ratio < 0.8 else "var(--ok)"
    centre = size / 2
    return (f'<div class="kl-gauge" style="width:{size}px">'
            f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img" aria-label="{esc(label)} {value:.0f}">'
            f'<circle cx="{centre}" cy="{centre}" r="{radius:.1f}" fill="none" stroke="var(--line)" stroke-width="9"/>'
            f'<circle cx="{centre}" cy="{centre}" r="{radius:.1f}" fill="none" stroke="{colour}" stroke-width="9" stroke-linecap="round" '
            f'stroke-dasharray="{circumference * ratio:.1f} {circumference:.1f}" transform="rotate(-90 {centre} {centre})"/>'
            f'<text x="{centre}" y="{centre + 2}" text-anchor="middle" dominant-baseline="middle" '
            f'style="font:600 {size * 0.26:.0f}px Georgia,serif;fill:var(--ink)">{value:.0f}</text></svg>'
            f'<div class="kl-gauge-label">{esc(label)}</div>' + (f'<div class="kl-gauge-cap">{esc(caption)}</div>' if caption else "") + "</div>")


def sparkline(values, width: int = 150, height: int = 40, kind: str = "line") -> str:
    """Minigráfico de línea (con relleno suave) o de barras. Ignora los vacíos y admite series largas (se resumen)."""
    series = pd.Series(values).dropna().astype(float)
    if len(series) < 2:
        return f'<svg width="{width}" height="{height}" aria-hidden="true"></svg>'
    if len(series) > width:
        series = series.groupby(np.arange(len(series)) * width // len(series)).mean()
    lo, hi = float(series.min()), float(series.max())
    span = (hi - lo) or 1.0
    pad = 3
    xs = np.linspace(pad, width - pad, len(series))
    ys = height - pad - (series.to_numpy() - lo) / span * (height - 2 * pad)
    if kind == "bars":
        w = max((width - 2 * pad) / len(series) - 1, 1)
        bars = "".join(f'<rect x="{x - w / 2:.1f}" y="{y:.1f}" width="{w:.1f}" height="{height - pad - y:.1f}" fill="var(--accent)" opacity=".85"/>'
                       for x, y in zip(xs, ys))
        return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-hidden="true">{bars}</svg>'
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"{xs[0]:.1f},{height - pad} {points} {xs[-1]:.1f},{height - pad}"
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-hidden="true">'
            f'<polygon points="{area}" fill="var(--accent)" opacity=".12"/>'
            f'<polyline points="{points}" fill="none" stroke="var(--accent)" stroke-width="1.8" stroke-linejoin="round"/>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="2.6" fill="var(--accent)"/></svg>')


def histogram_spark(values, bins: int = 14, width: int = 150, height: int = 40) -> str:
    """Distribución de una columna numérica como barras diminutas."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.nunique() < 2:
        return sparkline([], width, height)
    counts, _ = np.histogram(series, bins=bins)
    return sparkline(counts, width, height, "bars")


def chip(text: str, kind: str = "") -> str:
    return f'<span class="kl-chip {esc(kind)}">{esc(text)}</span>'


def chips(items) -> str:
    """Etiquetas en fila. Cada elemento es un texto o una pareja (texto, clase)."""
    out = []
    for item in items:
        text, kind = (item, "") if isinstance(item, str) else item
        out.append(chip(text, kind))
    return '<div class="kl-chips">' + "".join(out) + "</div>"


def tile(label: str, value, detail: str = "", spark: str = "") -> str:
    """Tarjeta de cifra con un minigráfico opcional (el SVG lo genera esta misma librería)."""
    return (f'<div class="kl-tile"><span>{esc(label)}</span><b>{esc(value)}</b>'
            + (f"<i>{esc(detail)}</i>" if detail else "") + (f'<div class="kl-tile-spark">{spark}</div>' if spark else "") + "</div>")


def tiles(items) -> str:
    return '<div class="kl-tiles">' + "".join(items) + "</div>"


def hero(title: str, subtitle: str, left_html: str = "", right_html: str = "") -> str:
    return (f'<div class="kl-hero"><div class="kl-hero-main"><h2>{esc(title)}</h2><p>{esc(subtitle)}</p>{left_html}</div>'
            f'<div class="kl-hero-side">{right_html}</div></div>')


def missing_strip(df: pd.DataFrame, max_columns: int = 24) -> str:
    """Una barra por columna: el color más intenso es más porcentaje de vacíos."""
    cols = list(df.columns)[:max_columns]
    cells = []
    for col in cols:
        pct = float(df[col].isna().mean() * 100)
        opacity = 0.12 + min(pct, 100) / 100 * 0.88
        cells.append(f'<div class="kl-miss" title="{esc(col)}: {pct:.1f}% empty" style="background:var(--accent);opacity:{opacity:.2f}"></div>')
    return '<div class="kl-strip">' + "".join(cells) + "</div>"
