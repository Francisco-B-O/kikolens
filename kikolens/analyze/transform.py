"""Transformaciones de datos y feature engineering. Cada operación recibe un DataFrame y parámetros simples
(serializables a JSON) y devuelve (nuevo_df, mensaje) sin modificar el original. Las expresiones se evalúan con un
intérprete aritmético propio, nunca con eval, para que una receta de un tercero no pueda ejecutar código."""
import ast
import operator
import re
import warnings

import numpy as np
import pandas as pd

FILTER_OPERATORS = ["==", "!=", ">", ">=", "<", "<=", "in", "not in", "contains", "starts with", "is empty", "is not empty"]
DATE_PARTS = ["year", "quarter", "month", "week", "day", "weekday", "hour", "is_weekend", "days_since"]
AGGREGATIONS = ["sum", "mean", "median", "min", "max", "count", "nunique", "std"]
SCALE_METHODS = ["standard", "minmax", "robust"]
MAX_ONE_HOT = 50


def _need(df: pd.DataFrame, *columns) -> None:
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")


def _need_numeric(df: pd.DataFrame, column: str) -> None:
    _need(df, column)
    if not pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric.")


def _unique_name(df: pd.DataFrame, name: str) -> str:
    candidate, i = name, 2
    while candidate in df.columns:
        candidate, i = f"{name}_{i}", i + 1
    return candidate


# ── Expresiones seguras ──────────────────────────────────────────────────────
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_COMPARE = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt, ast.LtE: operator.le,
            ast.Eq: operator.eq, ast.NotEq: operator.ne}
_FUNCTIONS = {"log": np.log, "log1p": np.log1p, "log10": np.log10, "sqrt": np.sqrt, "abs": np.abs, "exp": np.exp,
              "round": np.round, "floor": np.floor, "ceil": np.ceil, "sin": np.sin, "cos": np.cos,
              "max": np.maximum, "min": np.minimum}
MAX_EXPRESSION = 300
MAX_DEPTH = 40


def _evaluate(node, env, depth=0):
    if depth > MAX_DEPTH:
        raise ValueError("The expression is too deeply nested.")
    ev = lambda child: _evaluate(child, env, depth + 1)
    if isinstance(node, ast.Expression):
        return ev(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"Unknown column or name '{node.id}'.")
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](ev(node.left), ev(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = ev(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
        return _COMPARE[type(node.ops[0])](ev(node.left), ev(node.comparators[0])).astype(float)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
        return _FUNCTIONS[node.func.id](*[ev(a) for a in node.args])
    raise ValueError("Only numbers, columns, + - * / // % **, comparisons and the functions "
                     f"{', '.join(sorted(_FUNCTIONS))} are allowed in an expression.")


def evaluate_expression(df: pd.DataFrame, expression: str) -> pd.Series:
    """Evalúa una expresión aritmética sobre las columnas. Los nombres con espacios van entre `acentos graves`."""
    expression = str(expression).strip()
    if not expression:
        raise ValueError("The expression is empty.")
    if len(expression) > MAX_EXPRESSION:
        raise ValueError(f"The expression is longer than {MAX_EXPRESSION} characters.")
    aliases = {}

    def alias(match):
        name = match.group(1)
        _need(df, name)
        key = f"_col{len(aliases)}"
        aliases[key] = df[name]
        return key

    source = re.sub(r"`([^`]+)`", alias, expression)
    env = {c: df[c] for c in df.columns if isinstance(c, str) and c.isidentifier()}
    env.update(aliases)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"The expression is not valid: {e.msg}.") from None
    with np.errstate(all="ignore"):
        result = _evaluate(tree, env)
    if np.isscalar(result):
        result = pd.Series(result, index=df.index)
    result = pd.Series(result, index=df.index)
    return result.replace([np.inf, -np.inf], np.nan)


# ── Filas ────────────────────────────────────────────────────────────────────
def filter_rows(df, column, op, value=None):
    """Conserva las filas que cumplen una condición sobre una columna."""
    _need(df, column)
    if op not in FILTER_OPERATORS:
        raise ValueError(f"Unknown operator '{op}'. Use one of {FILTER_OPERATORS}.")
    s = df[column]
    numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    datetime = pd.api.types.is_datetime64_any_dtype(s)

    def scalar(raw):
        if numeric:
            try:
                return float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"'{raw}' is not a number, but '{column}' is numeric.") from None
        if datetime:
            parsed = pd.to_datetime(raw, errors="coerce")
            if pd.isna(parsed):
                raise ValueError(f"'{raw}' is not a date.")
            return parsed
        return str(raw)

    if op == "is empty":
        mask = s.isna()
    elif op == "is not empty":
        mask = s.notna()
    else:
        if value is None or value == "":
            raise ValueError("A value is required for this condition.")
        if op in ("in", "not in"):
            items = value if isinstance(value, (list, tuple)) else str(value).split(",")
            items = [scalar(str(v).strip()) for v in items]
            mask = s.isin(items) if numeric or datetime else s.astype(str).isin(items)
            mask = mask if op == "in" else ~mask
        elif op in ("contains", "starts with"):
            text = s.astype(str)
            mask = text.str.contains(str(value), case=False, regex=False) if op == "contains" \
                else text.str.lower().str.startswith(str(value).lower())
            mask = mask & s.notna()
        else:
            target = scalar(value)
            compare = s if numeric or datetime else s.astype(str)
            ops = {"==": operator.eq, "!=": operator.ne, ">": operator.gt, ">=": operator.ge,
                   "<": operator.lt, "<=": operator.le}
            if not (numeric or datetime) and op not in ("==", "!="):
                raise ValueError(f"'{column}' is text; use == or != (or 'contains').")
            mask = ops[op](compare, target) & s.notna()
            if op == "!=":
                mask = ops[op](compare, target)
    out = df[mask.fillna(False)].reset_index(drop=True)
    shown = "" if op.startswith("is ") else f" {value}"
    return out, f"Kept {len(out):,} of {len(df):,} rows where '{column}' {op}{shown}."


def sort_rows(df, by, ascending=True):
    by = [by] if isinstance(by, str) else list(by)
    _need(df, *by)
    return df.sort_values(by, ascending=ascending, kind="stable").reset_index(drop=True), \
        f"Sorted by {', '.join(by)} ({'ascending' if ascending else 'descending'})."


def sample_rows(df, n=None, fraction=None, seed=42):
    if (n is None) == (fraction is None):
        raise ValueError("Give either a number of rows or a fraction.")
    if fraction is not None:
        if not 0 < fraction <= 1:
            raise ValueError("The fraction must be in (0, 1].")
        out = df.sample(frac=fraction, random_state=seed)
    else:
        n = int(n)
        if n < 1:
            raise ValueError("The number of rows must be at least 1.")
        out = df.sample(n=min(n, len(df)), random_state=seed)
    return out.reset_index(drop=True), f"Sampled {len(out):,} of {len(df):,} rows."


# ── Columnas nuevas ──────────────────────────────────────────────────────────
def add_column(df, name, expression):
    """Crea una columna a partir de una expresión aritmética, por ejemplo `price * units` o `log(`unit price`)`."""
    name = str(name).strip()
    if not name:
        raise ValueError("The new column needs a name.")
    out = df.copy()
    out[name] = evaluate_expression(df, expression)
    verb = "Replaced" if name in df.columns else "Created"
    return out, f"{verb} column '{name}' = {expression}."


def concat_columns(df, columns, name, separator=" "):
    columns = list(columns)
    if len(columns) < 2:
        raise ValueError("Choose at least two columns to combine.")
    _need(df, *columns)
    out = df.copy()
    parts = [df[c].astype("string") for c in columns]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined + separator + part
    out[name] = joined.astype(object).where(joined.notna())
    return out, f"Combined {', '.join(columns)} into '{name}'."


def split_column(df, column, separator, names):
    _need(df, column)
    names = list(names)
    if not separator:
        raise ValueError("A separator is required.")
    if not names:
        raise ValueError("Give at least one name for the new columns.")
    parts = df[column].astype("string").str.split(separator, n=len(names) - 1, expand=True)
    out = df.copy()
    for i, name in enumerate(names):
        out[name] = parts[i].astype(object).where(parts[i].notna()) if i in parts.columns else np.nan
    return out, f"Split '{column}' on '{separator}' into {', '.join(names)}."


def extract_date_parts(df, column, parts):
    """Añade columnas derivadas de una fecha: año, mes, día de la semana, fin de semana, días desde el inicio."""
    _need(df, column)
    parts = list(parts)
    bad = [p for p in parts if p not in DATE_PARTS]
    if bad or not parts:
        raise ValueError(f"Choose parts among {DATE_PARTS}.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dates = pd.to_datetime(df[column], errors="coerce")
    if dates.notna().sum() == 0:
        raise ValueError(f"'{column}' does not contain dates.")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    out = df.copy()
    made = []
    for part in parts:
        if part == "week":
            values = dates.dt.isocalendar().week.astype(float)
        elif part == "is_weekend":
            values = (dates.dt.dayofweek >= 5).astype(float).where(dates.notna())
        elif part == "days_since":
            values = (dates - dates.min()).dt.days.astype(float)
        else:
            values = getattr(dates.dt, "dayofweek" if part == "weekday" else part).astype(float)
        new = f"{column}_{part}"
        out[new] = values
        made.append(new)
    return out, f"Extracted {', '.join(made)} from '{column}'."


def bin_column(df, column, bins=5, method="quantile", name=None):
    """Convierte una variable numérica en tramos (por cuantiles o de igual anchura)."""
    _need_numeric(df, column)
    bins = int(bins)
    if not 2 <= bins <= 50:
        raise ValueError("The number of bins must be between 2 and 50.")
    if method not in ("quantile", "uniform"):
        raise ValueError("Method must be 'quantile' or 'uniform'.")
    new = name or f"{column}_bin"
    s = df[column]
    cut = pd.qcut(s, bins, duplicates="drop") if method == "quantile" else pd.cut(s, bins)
    out = df.copy()
    out[new] = cut.astype(str).where(s.notna())
    return out, f"Binned '{column}' into {cut.nunique()} {method} bins as '{new}'."


def one_hot_encode(df, column, max_categories=MAX_ONE_HOT, drop_original=True):
    _need(df, column)
    n = df[column].nunique()
    if n > max_categories:
        raise ValueError(f"'{column}' has {n:,} categories; the limit is {max_categories}.")
    dummies = pd.get_dummies(df[column], prefix=column, dtype=float)
    out = pd.concat([df.drop(columns=[column]) if drop_original else df, dummies], axis=1)
    return out, f"One-hot encoded '{column}' into {dummies.shape[1]} columns."


def scale_column(df, column, method="standard"):
    _need_numeric(df, column)
    if method not in SCALE_METHODS:
        raise ValueError(f"Method must be one of {SCALE_METHODS}.")
    s = df[column].astype(float)
    if method == "standard":
        center, spread = s.mean(), s.std()
    elif method == "minmax":
        center, spread = s.min(), s.max() - s.min()
    else:
        center, spread = s.median(), s.quantile(0.75) - s.quantile(0.25)
    if not spread or pd.isna(spread):
        raise ValueError(f"'{column}' has no spread, so it cannot be scaled.")
    out = df.copy()
    out[column] = (s - center) / spread
    return out, f"Scaled '{column}' with the {method} method."


def log_transform(df, column):
    _need_numeric(df, column)
    s = df[column].astype(float)
    if (s.dropna() < 0).any():
        raise ValueError(f"'{column}' has negative values; a log transform needs values of 0 or more.")
    out = df.copy()
    out[column] = np.log1p(s)
    return out, f"Applied log(1 + x) to '{column}'."


def rank_column(df, column, pct=True):
    _need_numeric(df, column)
    out = df.copy()
    out[f"{column}_rank"] = df[column].rank(pct=pct, method="average")
    return out, f"Added {'percentile' if pct else 'position'} rank of '{column}'."


def replace_values(df, column, mapping):
    _need(df, column)
    if not mapping:
        raise ValueError("Give at least one value to replace.")
    out = df.copy()
    s = out[column]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        try:
            mapping = {float(k): v for k, v in mapping.items()}
        except (TypeError, ValueError):
            raise ValueError(f"'{column}' is numeric; the values to replace must be numbers.") from None
    out[column] = s.replace(mapping)
    return out, f"Replaced {len(mapping)} value(s) in '{column}'."


def lag_features(df, column, lags, order_by=None, group_by=None):
    """Valores anteriores de una columna (útil para series temporales o historiales por cliente)."""
    _need_numeric(df, column)
    lags = sorted({int(l) for l in lags})
    if not lags or lags[0] < 1 or lags[-1] > 366:
        raise ValueError("Lags must be between 1 and 366.")
    out = df.sort_values(order_by, kind="stable") if order_by else df.copy()
    if order_by:
        _need(df, order_by)
    if group_by:
        _need(df, group_by)
    source = out.groupby(group_by)[column] if group_by else out[column]
    for lag in lags:
        out[f"{column}_lag{lag}"] = source.shift(lag)
    return out.sort_index().reset_index(drop=True) if order_by else out, \
        f"Added lagged copies of '{column}' ({', '.join(map(str, lags))})."


def rolling_feature(df, column, window, stat="mean", order_by=None, group_by=None):
    _need_numeric(df, column)
    window = int(window)
    if not 2 <= window <= 1000:
        raise ValueError("The window must be between 2 and 1000 rows.")
    if stat not in ("mean", "sum", "min", "max", "std", "median"):
        raise ValueError("Statistic must be mean, sum, min, max, std or median.")
    out = df.sort_values(order_by, kind="stable") if order_by else df.copy()
    if order_by:
        _need(df, order_by)
    if group_by:
        _need(df, group_by)
    source = out.groupby(group_by)[column] if group_by else out[column]
    rolled = source.rolling(window, min_periods=max(1, window // 2))
    values = getattr(rolled, stat)()
    out[f"{column}_roll{window}_{stat}"] = values.reset_index(level=0, drop=True) if group_by else values
    return out.sort_index().reset_index(drop=True) if order_by else out, \
        f"Added the rolling {stat} of '{column}' over {window} rows."


# ── Cambios de forma ─────────────────────────────────────────────────────────
def group_aggregate(df, by, aggregations):
    """Resume el dataset: una fila por grupo. `aggregations` es {columna: [funciones]}."""
    by = [by] if isinstance(by, str) else list(by)
    _need(df, *by)
    if not aggregations:
        raise ValueError("Choose at least one column to summarize.")
    spec = {}
    for column, funcs in aggregations.items():
        _need(df, column)
        funcs = [funcs] if isinstance(funcs, str) else list(funcs)
        bad = [f for f in funcs if f not in AGGREGATIONS]
        if bad:
            raise ValueError(f"Unknown aggregation {bad}. Use one of {AGGREGATIONS}.")
        spec[column] = funcs
    out = df.groupby(by, dropna=False).agg(spec)
    out.columns = [f"{c}_{f}" for c, f in out.columns]
    out = out.reset_index()
    return out, f"Grouped by {', '.join(by)}: {len(out):,} groups."


def pivot_table(df, index, columns, values, aggfunc="sum"):
    _need(df, index, columns, values)
    if aggfunc not in AGGREGATIONS:
        raise ValueError(f"Aggregation must be one of {AGGREGATIONS}.")
    n = df[columns].nunique()
    if n > 100:
        raise ValueError(f"'{columns}' has {n:,} distinct values, too many for columns.")
    out = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)
    out.columns = [f"{values}_{c}" for c in out.columns]
    return out.reset_index(), f"Pivoted '{values}' by {index} (rows) and {columns} (columns)."


def melt_columns(df, id_columns, value_columns, var_name="variable", value_name="value"):
    id_columns, value_columns = list(id_columns), list(value_columns)
    _need(df, *id_columns, *value_columns)
    if not value_columns:
        raise ValueError("Choose the columns to stack.")
    out = df.melt(id_vars=id_columns, value_vars=value_columns, var_name=var_name, value_name=value_name)
    return out, f"Stacked {len(value_columns)} columns into '{var_name}' / '{value_name}'."


def rename_columns(df, mapping):
    if not mapping:
        raise ValueError("Give at least one column to rename.")
    _need(df, *mapping)
    new = [mapping.get(c, c) for c in df.columns]
    if len(set(new)) != len(new):
        raise ValueError("Renaming would create duplicate column names.")
    return df.rename(columns=mapping), f"Renamed {len(mapping)} column(s)."


def merge_datasets(df, other_df, on, how="left", other="other"):
    """Une otro dataset por una o varias columnas clave (left, inner, outer, right)."""
    on = [on] if isinstance(on, str) else list(on)
    if how not in ("left", "inner", "outer", "right"):
        raise ValueError("How must be left, inner, outer or right.")
    missing = [c for c in on if c not in df.columns or c not in other_df.columns]
    if missing:
        raise ValueError(f"Key column(s) {missing} must exist in both datasets.")
    out = df.merge(other_df, on=on, how=how, suffixes=("", f"_{other}"))
    if len(out) > 20 * max(len(df), 1) and len(out) > 1_000_000:
        raise ValueError("The join would create more than a million extra rows; check that the keys are unique.")
    return out, f"Joined '{other}' on {', '.join(on)} ({how}): {len(out):,} rows."


def append_dataset(df, other_df, other="other"):
    out = pd.concat([df, other_df], ignore_index=True, sort=False)
    return out, f"Appended {len(other_df):,} rows from '{other}'."


OPERATIONS = {
    "filter_rows": filter_rows, "sort_rows": sort_rows, "sample_rows": sample_rows, "add_column": add_column,
    "concat_columns": concat_columns, "split_column": split_column, "extract_date_parts": extract_date_parts,
    "bin_column": bin_column, "one_hot_encode": one_hot_encode, "scale_column": scale_column,
    "log_transform": log_transform, "rank_column": rank_column, "replace_values": replace_values,
    "lag_features": lag_features, "rolling_feature": rolling_feature, "group_aggregate": group_aggregate,
    "pivot_table": pivot_table, "melt_columns": melt_columns, "rename_columns": rename_columns,
    "merge_datasets": merge_datasets, "append_dataset": append_dataset,
}
# Operaciones que necesitan un segundo dataset (se pasa como `other_df`)
DATASET_OPERATIONS = {"merge_datasets", "append_dataset"}
