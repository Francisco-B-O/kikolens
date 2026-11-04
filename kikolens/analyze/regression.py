"""Regresión explicada: mínimos cuadrados (para un número) y regresión logística (para una categoría de dos valores) con
coeficientes, errores estándar, intervalos, valores p y diagnósticos, y cada efecto dicho en una frase. Es la herramienta
para entender qué empuja un resultado, no para predecir lo mejor posible (para eso está el laboratorio)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from kikolens.analyze.quality import infer_types
from kikolens.utils.dtypes import categorical_columns

MAX_CATEGORIES = 12
MIN_ROWS_PER_TERM = 5


def design(df: pd.DataFrame, target: str, features=None, standardize_note: bool = True):
    """Matriz de diseño con constante. Las categorías se codifican en indicadores, dejando fuera la más frecuente (la
    referencia). Devuelve (X, y, info)."""
    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found.")
    df, _ = infer_types(df)
    if features is None:
        features = [c for c in df.columns if c != target]
    features = [c for c in features if c in df.columns and c != target]
    if not features:
        raise ValueError("Choose at least one column to explain the target with.")
    parts, dropped, references = [], [], {}
    for col in features:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            parts.append(((s - s.min()).dt.days.astype(float)).rename(f"{col} (days)"))
        elif pd.api.types.is_bool_dtype(s):
            parts.append(s.astype(float).rename(col))
        elif pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) <= 1:
                dropped.append((col, "constant"))
            else:
                parts.append(s.astype(float).rename(col))
        elif col in categorical_columns(df):
            counts = s.value_counts()
            if len(counts) < 2:
                dropped.append((col, "constant"))
            elif len(counts) > MAX_CATEGORIES:
                dropped.append((col, f"more than {MAX_CATEGORIES} categories"))
            else:
                reference = counts.index[0]
                references[col] = str(reference)
                dummies = pd.get_dummies(s.astype(str).where(s.notna()), prefix=col, prefix_sep=" = ", dtype=float)
                dummies = dummies.drop(columns=f"{col} = {reference}", errors="ignore")
                dummies[s.isna().to_numpy()] = np.nan
                parts.append(dummies)
        else:
            dropped.append((col, "unsupported type"))
    if not parts:
        raise ValueError("None of the chosen columns can be used (they are constant or not numbers or categories).")
    X = pd.concat(parts, axis=1)
    y = df[target]
    return X, y, {"dropped": dropped, "references": references}


def _prepare(df, target, features, kind):
    X, y, info = design(df, target, features)
    if kind == "ols":
        if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
            raise ValueError(f"'{target}' must be a number for a linear regression.")
        y = y.astype(float)
    data = pd.concat([X, y.rename("__y__")], axis=1).dropna()
    n_missing = len(X) - len(data)
    if len(data) < MIN_ROWS_PER_TERM * (X.shape[1] + 1):
        raise ValueError(f"{len(data)} complete rows is too few for {X.shape[1]} terms; at least "
                         f"{MIN_ROWS_PER_TERM * (X.shape[1] + 1)} are needed. Choose fewer columns.")
    return data.drop(columns="__y__"), data["__y__"], info, n_missing


def _vif(X: pd.DataFrame) -> pd.Series:
    if X.shape[1] < 2:
        return pd.Series(1.0, index=X.columns)
    corr = np.corrcoef(X.to_numpy(float), rowvar=False)
    inverse = np.linalg.pinv(corr)
    return pd.Series(np.diag(inverse), index=X.columns)


def _table(names, beta, cov, dof, X: pd.DataFrame, y_scale: float, coverage: float, logistic: bool) -> pd.DataFrame:
    se = np.sqrt(np.diag(cov))
    stat = beta / se
    p = 2 * (stats.norm.sf(np.abs(stat)) if logistic else stats.t.sf(np.abs(stat), dof))
    crit = stats.norm.ppf(0.5 + coverage / 2) if logistic else stats.t.ppf(0.5 + coverage / 2, dof)
    table = pd.DataFrame({"term": names, "coefficient": beta, "std error": se, "z" if logistic else "t": stat, "p-value": p,
                          "low": beta - crit * se, "high": beta + crit * se})
    sd = np.concatenate([[np.nan], X.std(ddof=1).to_numpy()])
    table["standardized"] = beta * sd / y_scale if y_scale else np.nan
    return table


def ols(df: pd.DataFrame, target: str, features=None, coverage: float = 0.95) -> dict:
    """Mínimos cuadrados ordinarios con errores estándar clásicos."""
    X, y, info, n_missing = _prepare(df, target, features, "ols")
    n, k = X.shape[0], X.shape[1] + 1
    A = np.column_stack([np.ones(n), X.to_numpy(float)])
    beta, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    fitted = A @ beta
    resid = y.to_numpy() - fitted
    dof = n - k
    rss, tss = float(resid @ resid), float(((y - y.mean()) ** 2).sum())
    sigma2 = rss / dof
    cov = np.linalg.pinv(A.T @ A) * sigma2
    r2 = 1 - rss / tss if tss else np.nan
    adj = 1 - (1 - r2) * (n - 1) / dof
    f_stat = ((tss - rss) / (k - 1)) / sigma2 if k > 1 and sigma2 else np.nan
    f_p = float(stats.f.sf(f_stat, k - 1, dof))
    table = _table(["(intercept)"] + list(X.columns), beta, cov, dof, X, float(y.std(ddof=1)), coverage, False)
    table["VIF"] = [np.nan] + _vif(X).tolist()
    dw = float(np.sum(np.diff(resid) ** 2) / rss) if rss else np.nan      # sólo tiene sentido si las filas tienen un orden
    aux = np.linalg.lstsq(A, resid ** 2, rcond=None)[0]
    aux_r2 = 1 - np.sum((resid ** 2 - A @ aux) ** 2) / np.sum((resid ** 2 - np.mean(resid ** 2)) ** 2)
    bp_p = float(stats.chi2.sf(n * aux_r2, k - 1)) if k > 1 else np.nan
    normal_p = float(stats.shapiro(resid[:5000]).pvalue) if n >= 8 else np.nan
    model = {"n": n, "terms": k - 1, "r2": float(r2), "adjusted r2": float(adj), "F": float(f_stat), "F p-value": f_p,
             "rmse": float(np.sqrt(rss / n)), "residual std": float(np.sqrt(sigma2)), "durbin-watson": dw,
             "heteroscedasticity p": bp_p, "residual normality p": normal_p}
    result = {"kind": "ols", "target": target, "table": table, "model": model, "fitted": fitted, "residuals": resid,
              "dropped": info["dropped"], "references": info["references"], "rows dropped for missing values": n_missing}
    result["flags"] = _flags(result)
    result["sentences"] = _sentences(result)
    result["summary"] = (f"{', '.join(table['term'][1:3])}{' and others' if len(table) > 3 else ''} explain "
                         f"{r2:.0%} of the variation in {target} (adjusted {adj:.0%}); the model as a whole is "
                         f"{'clearly better than the average alone' if f_p < 0.05 else 'not clearly better than the average alone'} "
                         f"(F p = {f_p:.3g}).")
    return result


def logit(df: pd.DataFrame, target: str, features=None, positive=None, coverage: float = 0.95, max_iter: int = 60) -> dict:
    """Regresión logística por Newton-Raphson. Los coeficientes se muestran también como razones de probabilidades (odds)."""
    X, y, info, n_missing = _prepare(df, target, features, "logit")
    labels = sorted(y.astype(str).unique())
    if len(labels) != 2:
        raise ValueError(f"'{target}' must have exactly two values for a logistic regression; it has {len(labels)}.")
    positive = str(positive) if positive is not None else labels[1]
    if positive not in labels:
        raise ValueError(f"'{positive}' is not a value of '{target}'.")
    t = (y.astype(str) == positive).to_numpy(float)
    n, k = X.shape[0], X.shape[1] + 1
    if t.sum() < MIN_ROWS_PER_TERM or (n - t.sum()) < MIN_ROWS_PER_TERM:
        raise ValueError("Each outcome needs at least 5 rows.")
    A = np.column_stack([np.ones(n), X.to_numpy(float)])
    scale = np.concatenate([[1.0], X.std(ddof=1).replace(0, 1).to_numpy()])   # mejor condicionamiento
    Z = A / scale
    beta, converged = np.zeros(k), False
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-np.clip(Z @ beta, -35, 35)))
        w = np.clip(p * (1 - p), 1e-10, None)
        hessian = (Z * w[:, None]).T @ Z + 1e-8 * np.eye(k)
        step = np.linalg.solve(hessian, Z.T @ (t - p))
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            converged = True
            break
    p = 1 / (1 + np.exp(-np.clip(Z @ beta, -35, 35)))
    cov_z = np.linalg.inv((Z * np.clip(p * (1 - p), 1e-10, None)[:, None]).T @ Z + 1e-8 * np.eye(k))
    beta_raw = beta / scale
    cov = cov_z / np.outer(scale, scale)
    table = _table(["(intercept)"] + list(X.columns), beta_raw, cov, n - k, X, 1.0, coverage, True)
    for col in ("coefficient", "low", "high"):
        table["odds ratio" if col == "coefficient" else f"odds {col}"] = np.exp(table[col])
    table["VIF"] = [np.nan] + _vif(X).tolist()
    p0 = t.mean()
    ll = float(np.sum(t * np.log(np.clip(p, 1e-15, 1)) + (1 - t) * np.log(np.clip(1 - p, 1e-15, 1))))
    ll0 = float(n * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0)))
    pred = p >= 0.5
    ranks = stats.rankdata(p)
    auc = float((ranks[t == 1].sum() - t.sum() * (t.sum() + 1) / 2) / (t.sum() * (n - t.sum())))
    model = {"n": n, "terms": k - 1, "positive": positive, "log-likelihood": ll, "mcfadden r2": 1 - ll / ll0,
             "accuracy": float((pred == t).mean()), "baseline accuracy": float(max(p0, 1 - p0)), "auc": auc,
             "converged": converged, "lr p-value": float(stats.chi2.sf(2 * (ll - ll0), k - 1))}
    result = {"kind": "logit", "target": target, "table": table, "model": model, "fitted": p, "dropped": info["dropped"],
              "references": info["references"], "rows dropped for missing values": n_missing}
    result["flags"] = _flags(result)
    result["sentences"] = _sentences(result)
    result["summary"] = (f"The model separates '{positive}' from the rest with AUC {auc:.2f} and accuracy "
                         f"{model['accuracy']:.0%} (always guessing the commonest gives {model['baseline accuracy']:.0%}).")
    return result


def _flags(result: dict) -> list:
    table, model = result["table"], result["model"]
    out = []
    if result["kind"] == "ols":
        if model["heteroscedasticity p"] < 0.05:
            out.append("The spread of the errors changes with the fitted values (Breusch-Pagan), so the standard errors and p-values "
                       "are too optimistic. Treat borderline p-values with caution.")
        if model["residual normality p"] < 0.01 and model["n"] < 200:
            out.append("The errors are not close to normal, which matters with few rows. The intervals may be unreliable.")
        if not 1.5 <= model["durbin-watson"] <= 2.5:
            out.append(f"Neighbouring rows have related errors (Durbin-Watson {model['durbin-watson']:.2f}). If the rows are in time order, "
                       "the model misses a trend or seasonal pattern.")
    else:
        if not model["converged"]:
            out.append("The fit did not fully converge, often because one column perfectly separates the outcomes. Remove it or simplify.")
        if (table["coefficient"].abs() > 15).any():
            out.append("A coefficient is extreme, a sign of (near) perfect separation: that column almost gives the answer away.")
    heavy = table[table["VIF"] > 10]
    if len(heavy):
        out.append(f"{', '.join(heavy['term'])} {'is' if len(heavy) == 1 else 'are'} largely explained by other columns (VIF above 10), "
                   "so their individual coefficients are unstable. Drop or combine them.")
    return out


def _sentences(result: dict, top: int = 6) -> list:
    table = result["table"].iloc[1:]
    significant = table[table["p-value"] < 0.05].reindex(table["standardized"].abs().sort_values(ascending=False).index).dropna(subset=["p-value"])
    target, out = result["target"], []
    for _, row in significant.head(top).iterrows():
        term = row["term"]
        if result["kind"] == "ols":
            if " = " in term:
                col, value = term.split(" = ", 1)
                out.append(f"Compared with '{result['references'].get(col, 'the reference')}', '{value}' goes with a {target} "
                           f"{abs(row['coefficient']):,.4g} {'higher' if row['coefficient'] > 0 else 'lower'}, other things equal.")
            else:
                out.append(f"Each extra unit of {term} goes with {target} {'rising' if row['coefficient'] > 0 else 'falling'} by "
                           f"{abs(row['coefficient']):,.4g}, other things equal.")
        else:
            ratio = row["odds ratio"]
            change = f"multiplies the odds by {ratio:.2f}" if ratio >= 1 else f"cuts the odds to {ratio:.2f} of what they were"
            if " = " in term:
                col, value = term.split(" = ", 1)
                out.append(f"Compared with '{result['references'].get(col, 'the reference')}', '{value}' {change} of '{result['model']['positive']}'.")
            else:
                out.append(f"Each extra unit of {term} {change} of '{result['model']['positive']}'.")
    if not out:
        out.append("No term is clearly different from zero at the 5% level.")
    return out


def coefficient_figure(result: dict) -> go.Figure:
    t = result["table"].iloc[1:].copy()
    col = "odds ratio" if result["kind"] == "logit" else "coefficient"
    lo, hi = ("odds low", "odds high") if result["kind"] == "logit" else ("low", "high")
    t = t.reindex(t["standardized"].abs().sort_values().index)
    fig = go.Figure(go.Scatter(x=t[col], y=t["term"], mode="markers",
                               error_x=dict(type="data", symmetric=False, array=t[hi] - t[col], arrayminus=t[col] - t[lo]),
                               marker=dict(size=8)))
    fig.add_vline(x=1.0 if result["kind"] == "logit" else 0.0, line_dash="dot")
    fig.update_layout(title="Effect of each term with its 95% range" + (" (odds ratio; 1 means no effect)" if result["kind"] == "logit" else ""))
    if result["kind"] == "logit":
        fig.update_xaxes(type="log")
    return fig


def residual_figure(result: dict) -> go.Figure:
    if result["kind"] != "ols":
        raise ValueError("Residual plots are for linear regression.")
    fig = go.Figure(go.Scatter(x=result["fitted"], y=result["residuals"], mode="markers", opacity=0.6))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title="Errors against fitted values (a fan or a curve means something is missing)",
                      xaxis_title="Fitted value", yaxis_title="Error")
    return fig
