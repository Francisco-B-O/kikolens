"""La historia de los datos: recorre un dataset como lo haría un analista y lo cuenta en capítulos cortos, cada uno con su frase, su
cifra clave y un gráfico: qué hay, qué calidad tiene, qué destaca, qué se mueve junto, cómo evoluciona, qué grupos difieren,
qué filas son raras y qué conviene hacer después. Un capítulo que no se puede calcular se omite sin detener a los demás."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from kikolens.analyze import timeseries, tsanomaly
from kikolens.analyze.anomalies import detect_anomalies
from kikolens.analyze.autopilot import column_roles, suggest_target
from kikolens.analyze.discovery import run_discovery_engine
from kikolens.analyze.quality import infer_types, profile_quality

MAX_LEVELS = 12


def _chapter(kicker: str, title: str, text: str, figure=None, table=None, stats_=None) -> dict:
    return {"kicker": kicker, "title": title, "text": text, "figure": figure, "table": table, "stats": stats_ or []}


def _overview(df, typed, roles, quality):
    kinds = pd.Series(roles).value_counts().to_dict()
    names = {"number": "numeric", "category": "categorical", "date": "date", "identifier": "identifier", "text": "free-text",
             "constant": "constant"}
    parts = ", ".join(f"{n} {names.get(k, k)}" for k, n in kinds.items())
    fig = go.Figure(go.Bar(x=list(kinds.values()), y=[names.get(k, k) for k in kinds], orientation="h"))
    fig.update_layout(title="Columns by kind", showlegend=False)
    return _chapter("The data", f"{len(df):,} rows and {df.shape[1]} columns",
                    f"There are {parts} columns. Quality is {quality['score']:.0f} out of 100.", fig,
                    stats_=[("Rows", f"{len(df):,}"), ("Columns", df.shape[1]), ("Empty cells", f"{quality['missing_pct']:.1f}%")])


def _quality(df, typed, roles, quality):
    issues = quality["issues"]
    if not issues:
        return _chapter("Quality", "The data are clean", "No missing values, duplicates, constant or mistyped columns were found.")
    worst = quality["columns"].sort_values("missing_pct", ascending=False).head(8)
    worst = worst[worst["missing_pct"] > 0]
    fig = None
    if len(worst):
        fig = go.Figure(go.Bar(x=worst["missing_pct"][::-1], y=worst["column"][::-1], orientation="h"))
        fig.update_layout(title="Empty values by column (%)", showlegend=False)
    lines = " ".join(i["message"] for i in issues[:3])
    return _chapter("Quality", f"{len(issues)} thing(s) to fix", lines, fig)


def _stands_out(df, typed, roles, quality):
    found = run_discovery_engine(typed)
    if not found:
        return None
    text = " ".join(f["message"].replace("**", "") for f in found[:4])
    return _chapter("What stands out", f"{len(found)} automatic finding(s)", text)


def _relationships(df, typed, roles, quality):
    numbers = [c for c, r in roles.items() if r == "number" and typed[c].nunique() > 5]
    if len(numbers) < 2:
        return None
    corr = typed[numbers].corr(method="spearman")
    pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().rename("rho").reset_index()
    pairs.columns = ["a", "b", "rho"]
    pairs["abs"] = pairs["rho"].abs()
    top = pairs.sort_values("abs", ascending=False).head(5)
    best = top.iloc[0]
    sample = typed[[best["a"], best["b"]]].dropna()
    sample = sample.sample(min(len(sample), 1500), random_state=0)
    fig = go.Figure(go.Scatter(x=sample[best["a"]], y=sample[best["b"]], mode="markers", opacity=0.55))
    fig.update_layout(title=f"{best['b']} against {best['a']}", xaxis_title=best["a"], yaxis_title=best["b"])
    strength = "very strongly" if best["abs"] > 0.8 else "clearly" if best["abs"] > 0.5 else "weakly"
    text = (f"{best['a']} and {best['b']} move {strength} together ({best['rho']:+.2f}, rank correlation). "
            + ("They carry much the same information, so a model may need only one of them." if best["abs"] > 0.9 else ""))
    table = top[["a", "b", "rho"]].round(3).rename(columns={"a": "column", "b": "with", "rho": "rank correlation"})
    return _chapter("What moves together", f"{best['a']} and {best['b']}", text.strip(), fig, table)


def _over_time(df, typed, roles, quality):
    dates = [c for c, r in roles.items() if r == "date"]
    numbers = [c for c, r in roles.items() if r == "number"]
    if not dates or not numbers:
        return None
    value = suggest_target(typed, roles)
    value = value if value in numbers else numbers[-1]
    series = timeseries.prepare_series(typed, dates[0], value, agg="mean")
    if len(series) < 12:
        return None
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines", name=value))
    fig.update_layout(title=f"{value} over time", showlegend=False)
    spikes = None
    try:
        spikes = tsanomaly.detect(series)
    except ValueError:
        pass
    if spikes is not None and len(spikes["anomalies"]):
        a = spikes["anomalies"]
        fig.add_trace(go.Scatter(x=a["date"], y=a["value"], mode="markers", marker=dict(size=9, symbol="x"), name="Unusual"))
        fig.update_layout(showlegend=True)
    change = series.iloc[-1] / series.iloc[0] - 1 if series.iloc[0] else np.nan
    text = (f"{value} {'rose' if change > 0 else 'fell'} {abs(change):.0%} from the first to the last period. "
            if np.isfinite(change) else "") + (spikes["summary"] if spikes is not None else "")
    return _chapter("Over time", f"How {value} evolves", text.strip(), fig)


def _groups(df, typed, roles, quality):
    value = suggest_target(typed, roles)
    if value is None or roles.get(value) != "number":
        return None
    best = None
    y = typed[value]
    for col, role in roles.items():
        if role != "category" or not 2 <= typed[col].nunique() <= MAX_LEVELS:
            continue
        frame = pd.DataFrame({"g": typed[col].astype(str), "y": y}).dropna()
        groups = [g["y"].to_numpy() for _, g in frame.groupby("g") if len(g) >= 5]
        if len(groups) < 2 or frame["y"].var() == 0:
            continue
        grand = frame["y"].mean()
        eta2 = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / ((frame["y"] - grand) ** 2).sum()
        p = float(stats.kruskal(*groups).pvalue)
        if best is None or eta2 > best[0]:
            best = (float(eta2), col, p, frame)
    if best is None:
        return None
    eta2, col, p, frame = best
    means = frame.groupby("g")["y"].agg(["mean", "size"]).sort_values("mean")
    fig = go.Figure(go.Bar(x=means["mean"], y=means.index, orientation="h"))
    fig.update_layout(title=f"Average {value} by {col}", showlegend=False)
    high, low = means.index[-1], means.index[0]
    text = (f"{col} explains {eta2:.0%} of the variation in {value}: {high} averages {means['mean'].iloc[-1]:,.4g} and {low} "
            f"{means['mean'].iloc[0]:,.4g}. " + ("The difference is unlikely to be chance." if p < 0.05 else "The difference could be chance."))
    return _chapter("Groups that differ", f"{value} by {col}", text, fig, means.round(3).reset_index().rename(columns={"g": col}))


def _unusual(df, typed, roles, quality):
    numbers = [c for c, r in roles.items() if r == "number"]
    if len(numbers) < 2 or len(df) < 30:
        return None
    res = detect_anomalies(typed)
    if not res["n_flagged"]:
        return None
    table = res["table"].head(5)
    return _chapter("Unusual rows", f"{res['n_flagged']:,} rows look different", res["summary"] + " The five most unusual are listed with the reason.",
                    None, table.round(3))


def _next(df, typed, roles, quality, chapters):
    steps = []
    if quality["issues"]:
        steps.append("Open Quality and cleaning and apply the safe fixes first.")
    if suggest_target(typed, roles):
        steps.append("Run the Autopilot or the Model lab to see how well the data predict the main numeric column.")
    if any(r == "date" for r in roles.values()):
        steps.append("Use Time series to forecast it with an honest range.")
    if any(c["kicker"] == "Groups that differ" for c in chapters):
        steps.append("Use Experiments to test whether the group difference is reliable.")
    steps.append("Pin the figures you want to follow in the Panel.")
    return _chapter("Next", "Where to go from here", " ".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))


def build_story(df: pd.DataFrame) -> dict:
    """Los capítulos de la historia y los que se omitieron (con el motivo)."""
    if df is None or df.empty or df.shape[1] < 2:
        raise ValueError("The story needs a dataset with at least 2 columns and some rows.")
    typed, _ = infer_types(df)
    roles = column_roles(typed)
    quality = profile_quality(typed)
    chapters, skipped = [], []
    for builder in (_overview, _quality, _stands_out, _relationships, _over_time, _groups, _unusual):
        try:
            chapter = builder(df, typed, roles, quality)
        except Exception as e:  # un capítulo roto no debe tumbar la historia
            skipped.append(f"{builder.__name__.strip('_')}: {e}")
            continue
        if chapter:
            chapters.append(chapter)
    chapters.append(_next(df, typed, roles, quality, chapters))
    return {"chapters": chapters, "skipped": skipped, "roles": roles, "quality": quality,
            "summary": f"{len(chapters)} chapters about {len(df):,} rows and {df.shape[1]} columns."}


def story_markdown(story: dict, title: str = "Data story") -> str:
    """La historia como Markdown, para guardarla o pegarla en un documento."""
    out = [f"# {title}", ""]
    for c in story["chapters"]:
        out += [f"## {c['title']}", f"*{c['kicker']}*", "", c["text"], ""]
        if isinstance(c["table"], pd.DataFrame) and len(c["table"]):
            out += [c["table"].head(8).to_markdown(index=False), ""]
    return "\n".join(out)
