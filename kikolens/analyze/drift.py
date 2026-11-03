"""Comparación de dos datasets (referencia y actual): qué columnas han cambiado de distribución. Sirve para vigilar
si los datos nuevos se parecen a aquellos con los que se entrenó un modelo, o para comparar dos periodos."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from kikolens.analyze.quality import convert_type, infer_types

PSI_MODERATE = 0.1
PSI_HIGH = 0.25
EPS = 1e-4
MAX_CATEGORIES = 50
LOW_CARDINALITY = 12


def _numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Índice de estabilidad poblacional con cortes en los cuantiles de la referencia."""
    ref, cur = reference.dropna().astype(float), current.dropna().astype(float)
    if ref.empty or cur.empty:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3 or ref.nunique() <= LOW_CARDINALITY:  # pocos valores distintos: se compara como categorías
        return psi_categorical(reference, current)
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.dropna().astype(str).value_counts(normalize=True)
    cur = current.dropna().astype(str).value_counts(normalize=True)
    if ref.empty or cur.empty:
        return float("nan")
    cats = ref.index.union(cur.index)
    p, q = ref.reindex(cats, fill_value=0).to_numpy(), cur.reindex(cats, fill_value=0).to_numpy()
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def _verdict(psi: float, p_value: float) -> str:
    if np.isnan(psi):
        return "no data"
    if psi >= PSI_HIGH:
        return "drift"
    if psi >= PSI_MODERATE or (not np.isnan(p_value) and p_value < 0.001 and psi >= PSI_MODERATE / 2):
        return "moderate"
    return "stable"


def align_types(reference: pd.DataFrame, current: pd.DataFrame) -> tuple:
    """Fechas y números guardados como texto pasan a su tipo real, y la columna se trata igual en ambos datasets."""
    reference, converted = infer_types(reference)
    current, already = infer_types(current)
    for col, kind in converted:
        if col in current.columns and (col, kind) not in already:
            current, _ = convert_type(current, col, kind)
    return reference, current


def compare_datasets(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Compara las columnas comunes. Devuelve una tabla por columna y un resumen de cambios de estructura."""
    if reference.empty or current.empty:
        raise ValueError("Both datasets need at least one row.")
    reference, current = align_types(reference, current)
    common = [c for c in reference.columns if c in current.columns]
    if not common:
        raise ValueError("The two datasets have no columns in common.")
    rows = []
    for col in common:
        ref, cur = reference[col], current[col]
        row = {"column": col, "missing % (ref)": ref.isna().mean() * 100, "missing % (now)": cur.isna().mean() * 100}
        if _numeric(ref) and _numeric(cur):
            r, c = ref.dropna().astype(float), cur.dropna().astype(float)
            p = float(stats.ks_2samp(r, c).pvalue) if len(r) and len(c) else float("nan")
            psi = psi_numeric(ref, cur)
            std = r.std()
            row.update(type="numeric", psi=psi, p_value=p,
                       change=f"mean {r.mean():.4g} to {c.mean():.4g}" if len(r) and len(c) else "",
                       shift=float((c.mean() - r.mean()) / std) if std and len(c) else 0.0)
        elif pd.api.types.is_datetime64_any_dtype(ref) or pd.api.types.is_datetime64_any_dtype(cur):
            row.update(type="date", psi=float("nan"), p_value=float("nan"), shift=0.0,
                       change=f"{str(ref.min())[:10]} to {str(ref.max())[:10]}  >  {str(cur.min())[:10]} to {str(cur.max())[:10]}")
        else:
            if ref.nunique() > 20 and ref.nunique() > 0.5 * max(ref.notna().sum(), 1):
                row.update(type="identifier", psi=float("nan"), p_value=float("nan"), shift=0.0,
                           change="a different value in almost every row")
                row["verdict"] = "info"
                rows.append(row)
                continue
            new = sorted(set(cur.dropna().astype(str)) - set(ref.dropna().astype(str)))
            gone = sorted(set(ref.dropna().astype(str)) - set(cur.dropna().astype(str)))
            table = pd.concat([ref.dropna().astype(str).value_counts(), cur.dropna().astype(str).value_counts()],
                              axis=1, keys=["ref", "cur"]).fillna(0)
            p = float("nan")
            if 2 <= len(table) <= MAX_CATEGORIES * 4 and table.to_numpy().sum() > 0:
                try:
                    p = float(stats.chi2_contingency(table.T.to_numpy())[1])
                except ValueError:
                    p = float("nan")
            psi = psi_categorical(ref, cur)
            notes = []
            if new:
                notes.append(f"new: {', '.join(new[:4])}")
            if gone:
                notes.append(f"gone: {', '.join(gone[:4])}")
            row.update(type="category", psi=psi, p_value=p, change="; ".join(notes), shift=0.0)
        row["verdict"] = "info" if row["type"] == "date" else _verdict(row["psi"], row["p_value"])
        if abs(row["missing % (now)"] - row["missing % (ref)"]) >= 10 and row["verdict"] == "stable":
            row["verdict"] = "moderate"
            row["change"] = (row["change"] + "; " if row["change"] else "") + "share of empty values changed"
        rows.append(row)
    table = pd.DataFrame(rows)
    order = {"drift": 0, "moderate": 1, "stable": 2, "info": 3, "no data": 4}
    table = table.sort_values(["verdict", "psi"], key=lambda s: s.map(order) if s.name == "verdict" else -s.fillna(-1),
                              ignore_index=True)
    drifted = int((table["verdict"] == "drift").sum())
    moderate = int((table["verdict"] == "moderate").sum())
    tested = int(table["verdict"].isin(["drift", "moderate", "stable"]).sum())
    only_ref = [c for c in reference.columns if c not in current.columns]
    only_cur = [c for c in current.columns if c not in reference.columns]
    type_changes = [c for c in common if _numeric(reference[c]) != _numeric(current[c])]
    share = (drifted + 0.5 * moderate) / max(tested, 1)
    level = "high" if share >= 0.3 or drifted >= 3 else "medium" if share >= 0.1 else "low"
    return {"table": table, "drifted": drifted, "moderate": moderate, "stable": int((table["verdict"] == "stable").sum()),
            "only_in_reference": only_ref, "only_in_current": only_cur, "type_changes": type_changes,
            "rows": (len(reference), len(current)), "level": level,
            "summary": f"{drifted} of {tested} columns drifted and {moderate} changed moderately "
                       f"({len(reference):,} reference rows against {len(current):,} now)."}


def distribution_figure(reference: pd.DataFrame, current: pd.DataFrame, column: str) -> go.Figure:
    reference, current = align_types(reference, current)
    ref, cur = reference[column], current[column]
    fig = go.Figure()
    if _numeric(ref) and _numeric(cur):
        edges = np.unique(np.quantile(pd.concat([ref, cur]).dropna(), np.linspace(0, 1, 31)))
        for series, name in ((ref, "Reference"), (cur, "Now")):
            counts = np.histogram(series.dropna(), edges)[0]
            fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts / max(counts.sum(), 1), name=name, opacity=0.6))
        fig.update_layout(barmode="overlay", yaxis_title="Share of rows", yaxis_tickformat=".0%")
    else:
        shares = pd.concat([ref.dropna().astype(str).value_counts(normalize=True),
                            cur.dropna().astype(str).value_counts(normalize=True)], axis=1, keys=["Reference", "Now"]).fillna(0)
        shares = shares.sort_values("Reference", ascending=False).head(15)
        for name in shares.columns:
            fig.add_trace(go.Bar(x=shares.index, y=shares[name], name=name))
        fig.update_layout(barmode="group", yaxis_title="Share of rows", yaxis_tickformat=".0%")
    fig.update_layout(title=f"'{column}' then and now")
    return fig


def split_by_position(df: pd.DataFrame, fraction: float = 0.5) -> tuple:
    """Divide en un primer tramo (referencia) y el resto (actual), útil para comparar el pasado con lo reciente."""
    if not 0.1 <= fraction <= 0.9:
        raise ValueError("The split must be between 10% and 90%.")
    cut = int(len(df) * fraction)
    if cut < 5 or len(df) - cut < 5:
        raise ValueError("Each part needs at least 5 rows.")
    return df.iloc[:cut], df.iloc[cut:]
