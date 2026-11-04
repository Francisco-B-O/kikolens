"""Análisis de supervivencia: cuánto tarda en ocurrir algo (abandono, avería, compra) cuando hay casos que aún no
han ocurrido (censurados). Kaplan-Meier con intervalo de Greenwood y test de log-rank entre grupos."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2, norm


def _clean(durations, events):
    d = pd.to_numeric(pd.Series(durations), errors="coerce").reset_index(drop=True)
    e = pd.Series(events).reset_index(drop=True)
    if e.dtype == object or not pd.api.types.is_numeric_dtype(e):
        raw = e.astype(str).str.strip().str.lower()
        e = raw.isin({"1", "true", "yes", "y", "t", "event", "churned", "dead"}).astype(float).where(e.notna())
    e = pd.to_numeric(e, errors="coerce")
    keep = d.notna() & e.notna() & (d >= 0)
    if keep.sum() < 5:
        raise ValueError("At least 5 rows with a valid duration (0 or more) and event are needed.")
    if not set(e[keep].unique()) <= {0, 1}:
        raise ValueError("The event column must be 0/1 (or yes/no): 1 when the event happened, 0 when it is still pending.")
    return d[keep].to_numpy(float), e[keep].to_numpy(int)


def kaplan_meier(durations, events, confidence: float = 0.95) -> pd.DataFrame:
    """Curva de supervivencia: tiempo, en riesgo, eventos, S(t) y su intervalo (log-log de Greenwood)."""
    d, e = _clean(durations, events)
    times = np.unique(d[e == 1])
    surv, var_sum, rows = 1.0, 0.0, [{"time": 0.0, "at_risk": len(d), "events": 0, "survival": 1.0, "lower": 1.0, "upper": 1.0}]
    z = norm.ppf(0.5 + confidence / 2)
    for t in times:
        at_risk = int((d >= t).sum())
        happened = int(((d == t) & (e == 1)).sum())
        surv *= 1 - happened / at_risk
        if at_risk > happened:
            var_sum += happened / (at_risk * (at_risk - happened))
        if 0 < surv < 1 and var_sum > 0:
            se = np.sqrt(var_sum) / abs(np.log(surv))
            lower, upper = surv ** np.exp(z * se), surv ** np.exp(-z * se)
        else:
            lower = upper = surv
        rows.append({"time": float(t), "at_risk": at_risk, "events": happened, "survival": surv,
                     "lower": float(lower), "upper": float(upper)})
    return pd.DataFrame(rows)


def median_survival(curve: pd.DataFrame):
    """El primer tiempo en que la supervivencia baja de 0,5; None si la curva nunca llega."""
    below = curve[curve["survival"] <= 0.5]
    return float(below["time"].iloc[0]) if len(below) else None


def logrank(durations, events, groups) -> dict:
    """Test de log-rank: ¿difieren las curvas de los grupos? Devuelve estadístico, grados de libertad y p."""
    d, e = _clean(durations, events)
    g = pd.Series(groups).reset_index(drop=True).astype(str)
    g = g[pd.to_numeric(pd.Series(durations).reset_index(drop=True), errors="coerce").notna().to_numpy()
          & pd.Series(events).reset_index(drop=True).notna().to_numpy()]
    g = g.to_numpy()[: len(d)]
    labels = sorted(set(g))
    if len(labels) < 2:
        raise ValueError("At least two groups are needed to compare curves.")
    times = np.unique(d[e == 1])
    k = len(labels)
    observed, expected = np.zeros(k), np.zeros(k)
    cov = np.zeros((k, k))
    for t in times:
        at_risk = np.array([((d >= t) & (g == lab)).sum() for lab in labels], float)
        total = at_risk.sum()
        happened = ((d == t) & (e == 1))
        events_total = happened.sum()
        obs = np.array([(happened & (g == lab)).sum() for lab in labels], float)
        observed += obs
        expected += events_total * at_risk / total
        if total > 1:
            factor = events_total * (total - events_total) / (total - 1)
            cov += factor * (np.diag(at_risk) / total - np.outer(at_risk, at_risk) / total ** 2)
    diff = (observed - expected)[:-1]
    sub = cov[:-1, :-1]
    stat = float(diff @ np.linalg.pinv(sub) @ diff)
    p = float(chi2.sf(stat, k - 1))
    return {"statistic": stat, "df": k - 1, "p_value": p, "groups": labels,
            "observed": dict(zip(labels, observed.tolist())), "expected": dict(zip(labels, expected.tolist()))}


def survival_by_group(df: pd.DataFrame, duration: str, event: str, group: str | None = None, max_groups: int = 8) -> dict:
    for col in (duration, event) + ((group,) if group else ()):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    data = df[[duration, event] + ([group] if group else [])].dropna()
    curves, medians = {}, {}
    if group:
        counts = data[group].astype(str).value_counts()
        if len(counts) > max_groups:
            raise ValueError(f"'{group}' has {len(counts)} groups; choose a column with at most {max_groups}.")
        for label in counts.index:
            part = data[data[group].astype(str) == label]
            if len(part) >= 5:
                curves[label] = kaplan_meier(part[duration], part[event])
    else:
        curves["all"] = kaplan_meier(data[duration], data[event])
    medians = {k: median_survival(v) for k, v in curves.items()}
    test = None
    if group and len(curves) >= 2:
        keep = data[group].astype(str).isin(curves)
        test = logrank(data.loc[keep, duration], data.loc[keep, event], data.loc[keep, group])
    return {"curves": curves, "medians": medians, "test": test, "figure": survival_figure(curves),
            "summary": _summary(medians, test, group)}


def survival_figure(curves: dict) -> go.Figure:
    fig = go.Figure()
    for name, curve in curves.items():
        fig.add_trace(go.Scatter(x=curve["time"], y=curve["survival"], mode="lines", line_shape="hv", name=str(name)))
        fig.add_trace(go.Scatter(x=list(curve["time"]) + list(curve["time"])[::-1],
                                 y=list(curve["upper"]) + list(curve["lower"])[::-1], fill="toself", opacity=0.12,
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.update_layout(title="Share still waiting for the event", xaxis_title="Time", yaxis_title="Survival",
                      yaxis_range=[0, 1.02])
    return fig


def _summary(medians: dict, test: dict | None, group) -> str:
    bits = [f"{name}: {'half had the event by ' + format(m, 'g') if m is not None else 'fewer than half had it'}"
            for name, m in medians.items()]
    text = "; ".join(bits) + "."
    if test:
        verdict = "differ clearly" if test["p_value"] < 0.05 else "do not differ clearly"
        text += f" The groups of '{group}' {verdict} (log-rank p = {test['p_value']:.3g})."
    return text
