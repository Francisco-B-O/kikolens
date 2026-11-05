"""Reglas de validación definidas por quien conoce los datos ("el precio es mayor que 0", "el estado es uno de..."). Se
guardan como JSON, se comprueban sobre cualquier dataset y se pueden sugerir a partir de los datos actuales."""
import json
import re
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FORMAT_VERSION = 1
CHECKS = ["not_null", "unique", "range", "in_set", "regex", "compare", "not_future"]
COMPARATORS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.equal, "!=": np.not_equal}
MAX_PATTERN = 200
_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)[+*{]")          # (a+)+, (a*)*: tiempo exponencial con ciertas entradas
MAX_RULES = 100


def _need(df: pd.DataFrame, *columns) -> None:
    for c in columns:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")


def violations(df: pd.DataFrame, rule: dict) -> pd.Series:
    """Serie booleana: True en las filas que incumplen la regla (los vacíos solo incumplen `not_null`)."""
    check, col = rule.get("check"), rule.get("column")
    if check not in CHECKS:
        raise ValueError(f"Unknown check '{check}'. Use one of {CHECKS}.")
    _need(df, col)
    s = df[col]
    if check == "not_null":
        return s.isna()
    if check == "unique":
        return s.duplicated(keep=False) & s.notna()
    if check == "range":
        lo, hi = rule.get("min"), rule.get("max")
        if lo is None and hi is None:
            raise ValueError("A range rule needs a minimum, a maximum or both.")
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError(f"'{col}' is not numeric.")
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= s < float(lo)
        if hi is not None:
            bad |= s > float(hi)
        return bad & s.notna()
    if check == "in_set":
        values = rule.get("values")
        if not values:
            raise ValueError("An allowed-values rule needs at least one value.")
        allowed = {str(v) for v in values}
        return ~s.astype(str).isin(allowed) & s.notna()
    if check == "regex":
        pattern = str(rule.get("pattern", ""))
        if not pattern or len(pattern) > MAX_PATTERN:
            raise ValueError(f"The pattern must be 1 to {MAX_PATTERN} characters.")
        if _NESTED_QUANTIFIER.search(pattern):
            raise ValueError("The pattern repeats a repeated group, which can take very long to evaluate. Simplify it.")
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"The pattern is not valid: {e}") from None
        return ~s.astype(str).map(lambda v: bool(compiled.fullmatch(v))) & s.notna()
    if check == "compare":
        other, op = rule.get("other"), rule.get("op")
        _need(df, other)
        if op not in COMPARATORS:
            raise ValueError(f"Operator must be one of {list(COMPARATORS)}.")
        a, b = s, df[other]
        both = a.notna() & b.notna()
        try:
            ok = pd.Series(COMPARATORS[op](a.to_numpy(), b.to_numpy()), index=df.index)
        except TypeError:
            raise ValueError(f"'{col}' and '{other}' cannot be compared.") from None
        return ~ok & both
    if not pd.api.types.is_datetime64_any_dtype(s):
        raise ValueError(f"'{col}' is not a date.")
    return (s > pd.Timestamp.now()) & s.notna()


def describe(rule: dict) -> str:
    check, col = rule["check"], rule["column"]
    if check == "not_null":
        return f"{col} must not be empty"
    if check == "unique":
        return f"{col} must be unique"
    if check == "range":
        lo, hi = rule.get("min"), rule.get("max")
        return f"{col} must be " + (f"between {lo} and {hi}" if lo is not None and hi is not None else
                                    f"at least {lo}" if lo is not None else f"at most {hi}")
    if check == "in_set":
        shown = ", ".join(map(str, rule["values"][:6])) + (" ..." if len(rule["values"]) > 6 else "")
        return f"{col} must be one of: {shown}"
    if check == "regex":
        return f"{col} must match {rule['pattern']}"
    if check == "compare":
        return f"{col} must be {rule['op']} {rule['other']}"
    return f"{col} must not be in the future"


@dataclass
class RuleSet:
    rules: list = field(default_factory=list)

    def add(self, rule: dict) -> str:
        if rule.get("check") not in CHECKS:
            raise ValueError(f"Unknown check '{rule.get('check')}'. Use one of {CHECKS}.")
        if len(self.rules) >= MAX_RULES:
            raise ValueError(f"At most {MAX_RULES} rules are supported.")
        rule = {**rule, "id": rule.get("id") or uuid.uuid4().hex[:8]}
        self.rules.append(rule)
        return rule["id"]

    def remove(self, rule_id: str) -> None:
        self.rules = [r for r in self.rules if r["id"] != rule_id]

    def to_json(self) -> str:
        return json.dumps({"kikolens_rules": FORMAT_VERSION, "rules": self.rules}, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, text: str) -> "RuleSet":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"The rules are not valid JSON: {e.msg}.") from None
        if not isinstance(raw, dict) or "rules" not in raw:
            raise ValueError("This file is not a KikoLens rule set.")
        if raw.get("kikolens_rules", 0) > FORMAT_VERSION:
            raise ValueError("These rules were made by a newer version of KikoLens.")
        rs = cls()
        for rule in raw["rules"]:
            if not isinstance(rule, dict):
                raise ValueError("Malformed rule.")
            rs.add(rule)
        return rs

    def validate(self, df: pd.DataFrame) -> dict:
        """Comprueba todas las reglas. Una regla que no se puede aplicar se informa, no detiene a las demás."""
        rows, failing = [], {}
        for rule in self.rules:
            entry = {"rule": describe(rule) if rule["check"] in CHECKS else str(rule), "id": rule["id"]}
            try:
                bad = violations(df, rule)
            except (ValueError, TypeError) as e:
                rows.append({**entry, "violations": None, "pass %": None, "error": str(e)})
                continue
            n = int(bad.sum())
            failing[rule["id"]] = df.index[bad].tolist()
            rows.append({**entry, "violations": n, "pass %": 100 * (1 - n / max(len(df), 1)), "error": ""})
        table = pd.DataFrame(rows, columns=["rule", "violations", "pass %", "error", "id"])
        checked = table[table["error"] == ""]
        broken = int((checked["violations"] > 0).sum())
        return {"table": table, "violations": failing, "rows_with_problems": len({i for v in failing.values() for i in v}),
                "passed": broken == 0 and (table["error"] == "").all(), "broken": broken,
                "summary": (f"{broken} of {len(checked)} rule(s) are broken." if len(checked) else "No rule could be checked.")}

    def violating_rows(self, df: pd.DataFrame, rule_id: str, limit: int = 200) -> pd.DataFrame:
        rule = next((r for r in self.rules if r["id"] == rule_id), None)
        if rule is None:
            raise ValueError("Unknown rule.")
        return df[violations(df, rule)].head(limit)


def suggest_rules(df: pd.DataFrame) -> RuleSet:
    """Reglas razonables a partir de los datos: columnas completas, identificadores únicos, rangos y valores permitidos."""
    rs = RuleSet()
    n = len(df)
    for col in df.columns:
        s = df[col]
        if n >= 10 and s.notna().all():
            rs.add({"check": "not_null", "column": col})
        if n >= 10 and s.notna().all() and s.nunique() == n and not pd.api.types.is_float_dtype(s):
            rs.add({"check": "unique", "column": col})
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.notna().any():
            if (s.dropna() >= 0).all():
                rs.add({"check": "range", "column": col, "min": 0})
        elif not pd.api.types.is_datetime64_any_dtype(s) and 2 <= s.nunique() <= 12 and n >= 20:
            rs.add({"check": "in_set", "column": col, "values": sorted(map(str, s.dropna().unique()))})
        if pd.api.types.is_datetime64_any_dtype(s) and (s.dropna() <= pd.Timestamp.now()).all():
            rs.add({"check": "not_future", "column": col})
    return rs
