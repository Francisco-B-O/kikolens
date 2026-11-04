"""Equidad de un modelo de clasificación: compara cómo trata a cada grupo (por ejemplo sexo o región). No dice qué es
justo, que es una decisión humana, pero muestra dónde hay diferencias que conviene poder justificar."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

MIN_GROUP = 10
FOUR_FIFTHS = 0.8


def group_metrics(y_true, y_pred, groups, positive) -> pd.DataFrame:
    """Por grupo: tamaño, tasa de selección (cuántos reciben el resultado positivo), tasa de verdaderos positivos,
    de falsos positivos y exactitud."""
    y = pd.Series(np.asarray(y_true)).astype(str).reset_index(drop=True)
    p = pd.Series(np.asarray(y_pred)).astype(str).reset_index(drop=True)
    g = pd.Series(np.asarray(groups)).astype(str).reset_index(drop=True)
    if not len(y) == len(p) == len(g):
        raise ValueError("The true values, predictions and groups must have the same length.")
    positive = str(positive)
    if positive not in set(y) | set(p):
        raise ValueError(f"The positive class '{positive}' does not appear in the data.")
    rows = []
    for label in sorted(g.unique()):
        m = g == label
        yt, yp = y[m] == positive, p[m] == positive
        pos, neg = yt.sum(), (~yt).sum()
        rows.append({"group": label, "rows": int(m.sum()), "selection rate": float(yp.mean()),
                     "true positive rate": float((yp & yt).sum() / pos) if pos else np.nan,
                     "false positive rate": float((yp & ~yt).sum() / neg) if neg else np.nan,
                     "accuracy": float((y[m] == p[m]).mean()), "enough rows": int(m.sum()) >= MIN_GROUP})
    return pd.DataFrame(rows)


def audit(pipe, X: pd.DataFrame, y: pd.Series, groups: pd.Series, positive) -> dict:
    """Mide las diferencias entre grupos y las compara con la regla de los cuatro quintos y con diferencias de 10 puntos."""
    table = group_metrics(y, pipe.predict(X), groups.loc[X.index] if hasattr(groups, "loc") else groups, positive)
    usable = table[table["enough rows"]]
    if len(usable) < 2:
        raise ValueError(f"At least two groups with {MIN_GROUP} or more rows are needed.")
    rate = usable["selection rate"]
    ratio = float(rate.min() / rate.max()) if rate.max() > 0 else np.nan
    gaps = {"demographic parity gap": float(rate.max() - rate.min()),
            "equal opportunity gap": float(usable["true positive rate"].max() - usable["true positive rate"].min()),
            "false positive gap": float(usable["false positive rate"].max() - usable["false positive rate"].min())}
    flags = []
    if ratio < FOUR_FIFTHS:
        low, high = usable.loc[rate.idxmin(), "group"], usable.loc[rate.idxmax(), "group"]
        flags.append(f"'{low}' receives the positive result at {ratio:.0%} of the rate of '{high}', "
                     f"below the usual 80% guideline.")
    if gaps["equal opportunity gap"] > 0.1:
        flags.append(f"Real positives are found much more often in some groups than others "
                     f"(gap of {gaps['equal opportunity gap']:.0%} in true positive rate).")
    if gaps["false positive gap"] > 0.1:
        flags.append(f"False alarms are not spread evenly (gap of {gaps['false positive gap']:.0%}).")
    skipped = table[~table["enough rows"]]["group"].tolist()
    return {"table": table, "selection_ratio": ratio, "gaps": gaps, "flags": flags, "skipped": skipped,
            "figure": _figure(usable),
            "summary": ("No large differences between groups were found." if not flags else f"{len(flags)} difference(s) to review. ")
            + (f" Groups with fewer than {MIN_GROUP} rows were left out: {', '.join(skipped)}." if skipped else "")}


def _figure(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in ("selection rate", "true positive rate", "false positive rate"):
        fig.add_trace(go.Bar(name=col, x=table["group"], y=table[col]))
    fig.update_layout(barmode="group", title="How the model treats each group", yaxis_tickformat=".0%")
    return fig
