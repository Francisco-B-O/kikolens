"""Atribución multicanal: repartir el mérito de cada conversión entre los canales por los que pasó la persona.

A partir de los contactos (persona, momento, canal) y de si convirtió, compara seis formas de repartir el mérito:
  * primer contacto, último contacto, lineal, con caída en el tiempo y por posición (40-20-40): reglas fijas;
  * cadena de Markov: cuánto bajaría la probabilidad de convertir si el canal no existiera (efecto de eliminación);
  * Shapley: el reparto que daría a cada canal su aportación media a todas las combinaciones posibles de canales.
Atribuir no es medir el efecto causal: reparte el mérito de lo que pasó entre los canales presentes, y los canales que se activan porque la persona ya iba a comprar
(búsqueda de marca, remarketing) se llevan más de lo que aportan. Para saber cuánto aporta un canal hay que probarlo apagándolo (un experimento) o usar el modelo de mezcla."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

MODELS = ("first touch", "last touch", "linear", "time decay", "position based", "markov", "shapley")
MAX_SHAPLEY_CHANNELS = 10
MIN_JOURNEYS = 30


def build_journeys(df: pd.DataFrame, user: str, time: str, channel: str, converted: str, value: str | None = None) -> pd.DataFrame:
    """Un recorrido por persona: canales en orden, si convirtió (y cuándo) y el valor. Los contactos posteriores a la conversión no cuentan."""
    for c in (user, time, channel, converted) + ((value,) if value else ()):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    if value and not pd.api.types.is_numeric_dtype(df[value]):
        raise ValueError(f"'{value}' must be numeric.")
    data = df.dropna(subset=[user, time]).copy()
    data[time] = pd.to_datetime(data[time], errors="coerce")
    data = data.dropna(subset=[time])
    flag = data[converted]
    if not (pd.api.types.is_bool_dtype(flag) or pd.api.types.is_numeric_dtype(flag)):
        flag = flag.astype(str).str.lower().isin(["1", "true", "yes", "y", "si", "sí", "converted"])
    data["_conv"] = flag.fillna(False).astype(bool)
    data = data.sort_values([user, time], kind="stable")
    rows = []
    for key, part in data.groupby(user, sort=False):
        conv_rows = part[part["_conv"]]
        if len(conv_rows):
            end = conv_rows[time].iloc[0]
            part = part[part[time] <= end]
        touches = part.dropna(subset=[channel])
        if touches.empty:
            continue
        done = bool(len(conv_rows))
        rows.append({"user": key, "channels": tuple(touches[channel].astype(str)), "times": tuple(touches[time]), "converted": done,
                     "value": float(part.loc[part["_conv"], value].sum()) if (value and done) else (1.0 if done else 0.0), "conversion_time": part[time].iloc[-1] if done else pd.NaT})
    journeys = pd.DataFrame(rows)
    if len(journeys) < MIN_JOURNEYS:
        raise ValueError(f"At least {MIN_JOURNEYS} people with contacts are needed (found {len(journeys)}).")
    if not journeys["converted"].any():
        raise ValueError("Nobody converted in this data, so there is nothing to attribute.")
    return journeys


# ── Reglas fijas ─────────────────────────────────────────────────────────────
def _heuristic(journeys: pd.DataFrame, half_life_days: float) -> pd.DataFrame:
    out = {m: {} for m in ("first touch", "last touch", "linear", "time decay", "position based")}

    def add(model, ch, credit):
        out[model][ch] = out[model].get(ch, 0.0) + credit

    for _, j in journeys[journeys["converted"]].iterrows():
        chans, times, value = j["channels"], j["times"], j["value"]
        n = len(chans)
        add("first touch", chans[0], value)
        add("last touch", chans[-1], value)
        for ch in chans:
            add("linear", ch, value / n)
        ages = np.array([max((j["conversion_time"] - t).total_seconds() / 86400, 0.0) for t in times])
        weights = 0.5 ** (ages / half_life_days)
        for ch, w in zip(chans, weights / weights.sum()):
            add("time decay", ch, value * w)
        if n == 1:
            add("position based", chans[0], value)
        elif n == 2:
            add("position based", chans[0], value * 0.5)
            add("position based", chans[1], value * 0.5)
        else:
            add("position based", chans[0], value * 0.4)
            add("position based", chans[-1], value * 0.4)
            for ch in chans[1:-1]:
                add("position based", ch, value * 0.2 / (n - 2))
    return pd.DataFrame(out).fillna(0.0)


# ── Cadena de Markov ─────────────────────────────────────────────────────────
def _transitions(journeys: pd.DataFrame, channels: list) -> np.ndarray:
    """Matriz de transición: estados = inicio, canales, conversión, nada. Las dos últimas son absorbentes."""
    k = len(channels)
    idx = {c: i + 1 for i, c in enumerate(channels)}
    start, conv, null = 0, k + 1, k + 2
    counts = np.zeros((k + 3, k + 3))
    for _, j in journeys.iterrows():
        path = [start] + [idx[c] for c in j["channels"]] + [conv if j["converted"] else null]
        for a, b in zip(path[:-1], path[1:]):
            counts[a, b] += 1
    counts[conv, conv] = counts[null, null] = 1
    return counts / counts.sum(axis=1, keepdims=True).clip(min=1)


def _conversion_probability(P: np.ndarray) -> float:
    n = P.shape[0]
    conv = n - 2
    transient = list(range(n - 2))
    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, [conv])]
    absorb = np.linalg.solve(np.eye(len(transient)) - Q, R)
    return float(absorb[0, 0])


def markov_attribution(journeys: pd.DataFrame, channels: list) -> tuple:
    """Efecto de eliminación: lo que baja la probabilidad de convertir si todos los caminos que pasan por un canal acaban en nada."""
    P = _transitions(journeys, channels)
    base = _conversion_probability(P)
    k = len(channels)
    removal = {}
    for i, ch in enumerate(channels):
        Pr = P.copy()
        Pr[:, i + 1] = 0.0
        Pr[:, k + 2] += P[:, i + 1]                                                 # lo que iba a ese canal va a «nada»
        Pr[i + 1, :] = 0.0
        Pr[i + 1, k + 2] = 1.0
        removal[ch] = 1 - _conversion_probability(Pr) / base if base > 0 else 0.0
    total = sum(removal.values())
    share = {ch: (removal[ch] / total if total > 0 else 0.0) for ch in channels}
    return share, removal, base


# ── Shapley ──────────────────────────────────────────────────────────────────
def shapley_attribution(journeys: pd.DataFrame, channels: list) -> dict:
    """Valor de una coalición S = conversiones de los recorridos que solo usan canales de S. El reparto de Shapley da a cada canal su aportación media a todas las coaliciones."""
    n = len(channels)
    if n > MAX_SHAPLEY_CHANNELS:
        raise ValueError(f"Shapley is exact up to {MAX_SHAPLEY_CHANNELS} channels (found {n}); group the smaller ones as 'other'.")
    pos = {c: i for i, c in enumerate(channels)}
    weight = np.zeros(1 << n)
    for _, j in journeys[journeys["converted"]].iterrows():
        mask = 0
        for ch in set(j["channels"]):
            mask |= 1 << pos[ch]
        weight[mask] += j["value"]
    value = np.zeros(1 << n)
    for mask in range(1 << n):                                                      # v(S) = suma de lo convertido por recorridos cuyo conjunto de canales está dentro de S
        sub = mask
        total = 0.0
        while True:
            total += weight[sub]
            if sub == 0:
                break
            sub = (sub - 1) & mask
        value[mask] = total
    fact = [math.factorial(i) for i in range(n + 1)]
    result = {}
    for ch in channels:
        i = pos[ch]
        phi = 0.0
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            size = bin(mask).count("1")
            phi += fact[size] * fact[n - size - 1] / fact[n] * (value[mask | (1 << i)] - value[mask])
        result[ch] = phi
    return result


# ── Análisis completo ────────────────────────────────────────────────────────
def attribute(df: pd.DataFrame, user: str, time: str, channel: str, converted: str, value: str | None = None, half_life_days: float = 7.0, other_below: float = 0.0) -> dict:
    """Reparte las conversiones con todos los modelos y compara. `other_below`: los canales con menos de esa fracción de contactos se agrupan como «other» (hace falta con >10 canales)."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive.")
    journeys = build_journeys(df, user, time, channel, converted, value)
    counts = pd.Series([c for path in journeys["channels"] for c in path]).value_counts()
    channels = list(counts.index)
    if other_below > 0 or len(channels) > MAX_SHAPLEY_CHANNELS:
        cut = max(other_below, 0.0)
        keep = [c for c in channels if counts[c] / counts.sum() >= cut][:MAX_SHAPLEY_CHANNELS - 1] if cut > 0 else channels[:MAX_SHAPLEY_CHANNELS - 1]
        mapper = {c: (c if c in keep else "other") for c in channels}
        journeys = journeys.assign(channels=journeys["channels"].map(lambda p: tuple(mapper[c] for c in p)))
        channels = [c for c in channels if c in keep] + ["other"]
    if len(channels) < 2:
        raise ValueError("At least two channels are needed to share credit between them.")
    total = float(journeys.loc[journeys["converted"], "value"].sum())
    table = _heuristic(journeys, half_life_days).reindex(channels).fillna(0.0)
    share, removal, base = markov_attribution(journeys, channels)
    table["markov"] = pd.Series(share).reindex(channels) * total
    shap = shapley_attribution(journeys, channels)
    table["shapley"] = pd.Series(shap).reindex(channels)
    table = table[list(MODELS)]
    shares = table / total
    appear = pd.Series({c: sum(c in p for p in journeys["channels"]) for c in channels})
    last = table["last touch"] / total
    rows = pd.DataFrame({"channel": channels, "journeys with the channel": appear.reindex(channels).to_numpy(), "removal effect": [removal[c] for c in channels]})
    summary_table = rows.merge(shares.add_suffix(" (share)").rename_axis("channel").reset_index(), on="channel")
    converting = journeys[journeys["converted"]]
    days = [(j["conversion_time"] - j["times"][0]).total_seconds() / 86400 for _, j in converting.iterrows()]
    paths = converting["channels"].map(lambda p: " > ".join(p)).value_counts().head(10)
    notes = []
    gap = (shares["markov"] - shares["last touch"]).abs()
    biggest = gap.idxmax()
    if gap.max() > 0.1:
        direction = "more" if shares.loc[biggest, "markov"] > shares.loc[biggest, "last touch"] else "less"
        notes.append(f"'{biggest}' gets {direction} credit when the whole path counts ({shares.loc[biggest, 'markov']:.0%} by Markov) than under last touch ({shares.loc[biggest, 'last touch']:.0%}): "
                     "the models disagree on who matters, and that is the information.")
    single = float((converting["channels"].map(len) == 1).mean())
    if single > 0.6:
        notes.append(f"{single:.0%} of the conversions came after a single contact, so every model gives similar answers; multi-touch attribution has little to add.")
    if int(converting.shape[0]) < 200:
        notes.append(f"Only {converting.shape[0]} conversions: the shares move a lot with a few more or fewer.")
    notes.append("Attribution splits the credit for what happened; it does not measure what each channel causes. Channels that reach people already about to buy (brand search, "
                 "remarketing) look better than they are. Test the big ones by switching them off in a region or a group.")
    fig = go.Figure()
    for model in ("last touch", "linear", "markov", "shapley"):
        fig.add_trace(go.Bar(x=channels, y=shares[model], name=model))
    fig.update_layout(barmode="group", title="Share of the credit by channel and model", yaxis_tickformat=".0%")
    return {"table": summary_table, "credit": table, "shares": shares, "journeys": journeys, "conversions": int(converting.shape[0]), "journeys_total": int(len(journeys)),
            "conversion_rate": float(journeys["converted"].mean()), "total_value": total, "median_days_to_convert": float(np.median(days)), "mean_touches": float(converting["channels"].map(len).mean()),
            "top_paths": paths, "markov_conversion_probability": base, "channels": channels, "notes": notes, "figure": fig,
            "summary": (f"{converting.shape[0]:,} conversions out of {len(journeys):,} people; converting people had {converting['channels'].map(len).mean():.1f} contacts on average over "
                        f"{np.median(days):.1f} days. Last touch gives most credit to '{last.idxmax()}' ({last.max():.0%}); by Markov it is '{shares['markov'].idxmax()}' ({shares['markov'].max():.0%}) "
                        f"and by Shapley '{shares['shapley'].idxmax()}' ({shares['shapley'].max():.0%}).")}
