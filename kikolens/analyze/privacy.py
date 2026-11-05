"""Privacidad: detecta datos personales (correos, teléfonos, tarjetas, IBAN, DNI/NIE, IP...), los oculta o sustituye por una
huella irreversible, y mide con k-anonimato cuánto se puede identificar a una persona por la combinación de sus datos.
Todo ocurre en local. Las operaciones son pasos de receta, así que se repiten idénticas con datos nuevos."""
import hashlib
import re

import numpy as np
import pandas as pd

from kikolens.analyze import quality
from kikolens.utils.dtypes import is_text

SAMPLE = 2000
MIN_SHARE = 0.6
NAME_HINTS = re.compile(r"(^|_|\b)(name|surname|firstname|first_name|lastname|last_name|fullname|nombre|apellidos?|e-?mail|mail|"
                        r"phone|tel|telefono|mobile|movil|address|direccion|dni|nie|passport|pasaporte|ssn|iban|card|tarjeta|"
                        r"birth|birthday|nacimiento|dob)(s)?($|_|\b)", re.IGNORECASE)
QUASI_HINTS = re.compile(r"(age|edad|birth|nacimiento|zip|postal|cp|gender|sex|sexo|city|ciudad|region|provincia|country|pais|"
                         r"job|occupation|profesion|salary|salario|nationality)", re.IGNORECASE)


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return total % 10 == 0


def _iban(value: str) -> bool:
    v = value.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", v):
        return False
    rearranged = v[4:] + v[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


def _dni(value: str) -> bool:
    v = value.replace("-", "").replace(" ", "").upper()
    m = re.fullmatch(r"([XYZ]|\d)(\d{7})([A-Z])", v)
    if not m:
        return False
    first = {"X": "0", "Y": "1", "Z": "2"}.get(m.group(1), m.group(1))
    return "TRWAGMYFPDXBNJZSQVHLCKE"[int(first + m.group(2)) % 23] == m.group(3)


def _card(value: str) -> bool:
    v = re.sub(r"[ -]", "", value)
    return v.isdigit() and 13 <= len(v) <= 19 and _luhn(v)


def _phone(value: str) -> bool:
    return bool(re.fullmatch(r"\+?[\d\s().-]{9,18}", value)) and 9 <= len(re.sub(r"\D", "", value)) <= 15


DETECTORS = {
    "email": lambda v: bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", v)),
    "credit card": _card,
    "IBAN": _iban,
    "Spanish ID (DNI/NIE)": _dni,
    "US social security number": lambda v: bool(re.fullmatch(r"\d{3}-\d{2}-\d{4}", v)),
    "IP address": lambda v: bool(re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", v)) and all(int(p) <= 255 for p in v.split(".")),
    "phone number": _phone,
}


def _mask_example(value: str) -> str:
    return value[:1] + "*" * max(len(value) - 2, 1) + value[-1:] if len(value) > 2 else "*" * len(value)


def scan(df: pd.DataFrame, sample: int = SAMPLE, min_share: float = MIN_SHARE) -> pd.DataFrame:
    """Columnas que parecen contener datos personales, por su contenido (patrones validados) o por su nombre."""
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        found = None
        if len(s) and (is_text(s) or pd.api.types.is_integer_dtype(s)):
            values = s.astype(str).str.strip()
            values = values.sample(min(len(values), sample), random_state=0) if len(values) > sample else values
            for kind, check in DETECTORS.items():
                hits = values.map(check)
                if hits.mean() >= min_share:
                    found = {"column": col, "kind": kind, "share": float(hits.mean()), "reason": "content",
                             "example": _mask_example(values[hits].iloc[0])}
                    break
        if found is None and NAME_HINTS.search(str(col)):
            found = {"column": col, "kind": "possible personal data", "share": np.nan, "reason": "name",
                     "example": _mask_example(str(s.iloc[0])) if len(s) else ""}
        if found:
            found["risk"] = "high" if found["kind"] in ("credit card", "IBAN", "US social security number", "Spanish ID (DNI/NIE)") else "medium"
            rows.append(found)
    return pd.DataFrame(rows, columns=["column", "kind", "share", "reason", "risk", "example"])


# ── Operaciones (pasos de receta) ────────────────────────────────────────────
def mask_column(df: pd.DataFrame, column: str, keep: int = 2, char: str = "*"):
    """Sustituye todo menos los últimos `keep` caracteres. Los vacíos se quedan vacíos."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    if keep < 0 or keep > 8:
        raise ValueError("keep must be between 0 and 8.")
    if len(char) != 1:
        raise ValueError("The mask character must be a single character.")

    def mask(value):
        if pd.isna(value):
            return value
        text = str(value)
        if len(text) <= keep:             # un valor corto no puede mostrarse entero
            return char * len(text)
        return char * (len(text) - keep) + (text[-keep:] if keep else "")

    out = df.copy()
    out[column] = df[column].map(mask)
    return out, f"Masked '{column}' (kept the last {keep} characters)."


def hash_column(df: pd.DataFrame, column: str, salt: str = "", length: int = 12):
    """Sustituye cada valor por una huella SHA-256 con sal: iguales entre sí (sirve para unir y contar) pero irreversibles. La sal
    queda en la receta: quien la tenga puede comprobar si un valor conocido está en los datos, así que no la compartas si importa."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    if not 6 <= length <= 64:
        raise ValueError("length must be between 6 and 64.")

    def digest(value):
        if pd.isna(value):
            return value
        return hashlib.sha256(f"{salt}{value}".encode("utf-8")).hexdigest()[:length]

    out = df.copy()
    out[column] = df[column].map(digest)
    return out, f"Replaced '{column}' with salted SHA-256 fingerprints."


OPERATIONS = {"mask_column": mask_column, "hash_column": hash_column}
ACTIONS = ("mask", "hash", "drop")


def anonymize(df: pd.DataFrame, plan: dict, salt: str = ""):
    """`plan`: columna -> 'mask' | 'hash' | 'drop'. Devuelve (nuevo_df, pasos) con pasos (operación, parámetros, mensaje)."""
    steps = []
    for column, action in plan.items():
        if action not in ACTIONS:
            raise ValueError(f"Action for '{column}' must be one of {ACTIONS}.")
        if action == "mask":
            df, message = mask_column(df, column)
            steps.append(("mask_column", {"column": column}, message))
        elif action == "hash":
            df, message = hash_column(df, column, salt=salt)
            steps.append(("hash_column", {"column": column, "salt": salt}, message))
        else:
            df, message = quality.drop_columns(df, [column])
            steps.append(("drop_columns", {"columns": [column]}, message))
    return df, steps


# ── k-anonimato ──────────────────────────────────────────────────────────────
def suggest_quasi_identifiers(df: pd.DataFrame) -> list:
    """Columnas que, combinadas, suelen identificar a alguien: edad, código postal, sexo, ciudad, profesión..."""
    return [c for c in df.columns if QUASI_HINTS.search(str(c))]


def k_anonymity(df: pd.DataFrame, quasi_identifiers: list, k: int = 5, top: int = 15) -> dict:
    """Tamaño del grupo de cada fila según los datos que la describen. Con k = 5, nadie es distinguible de menos de 4 personas más."""
    cols = [c for c in quasi_identifiers if c in df.columns]
    if not cols:
        raise ValueError("Choose at least one column that describes a person (age, postal code, sex, city...).")
    if k < 2:
        raise ValueError("k must be at least 2.")
    keyed = df[cols].astype(object).where(df[cols].notna(), "(empty)").astype(str)
    sizes = keyed.groupby(cols, dropna=False)[cols[0]].transform("size")
    below = sizes < k
    smallest = (keyed.assign(_size=sizes).drop_duplicates().sort_values("_size").head(top)
                .rename(columns={"_size": "people in group"}).reset_index(drop=True))
    unique = int((sizes == 1).sum())
    return {"min_k": int(sizes.min()), "k": k, "share_below_k": float(below.mean()), "unique_rows": unique,
            "groups": int(keyed.drop_duplicates().shape[0]), "smallest_groups": smallest, "row_group_sizes": sizes,
            "summary": (f"The smallest group has {int(sizes.min())} row(s). {below.mean():.1%} of the rows are in a group smaller than {k}"
                        f"{f', and {unique:,} are unique' if unique else ''}. "
                        + ("The combination is safe at this k." if not below.any() else
                           "Generalise (age bands, the first digits of a postal code) or remove a column to make groups bigger."))}
