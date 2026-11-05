"""Magias de IPython / Jupyter:  `%load_ext kikolens`

    %kl_profile df                  calidad de un DataFrame
    %kl_autopilot df [columna]      primer análisis completo
    %kl_privacy df                  columnas con datos personales
    %%kl_sql df                     (en la primera línea el DataFrame; el cuerpo de la celda es la consulta SQL de solo lectura)

Las magias devuelven el mismo `Result` que la API, así que el cuaderno lo muestra como una tabla legible."""
import pandas as pd


def _frame(ipython, name: str) -> pd.DataFrame:
    name = (name or "").strip()
    if not name:
        raise ValueError("Give the name of a DataFrame, for example: %kl_profile df")
    if name not in ipython.user_ns:
        raise ValueError(f"There is no variable called '{name}' in this notebook.")
    value = ipython.user_ns[name]
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"'{name}' is not a DataFrame.")
    return value


def register(ipython) -> None:
    from kikolens import api

    def kl_profile(line=""):
        """Calidad de un DataFrame: %kl_profile df"""
        return api.profile(_frame(ipython, line))

    def kl_autopilot(line=""):
        """Primer análisis completo: %kl_autopilot df [columna a predecir]"""
        name, _, target = line.strip().partition(" ")
        return api.autopilot(_frame(ipython, name), target.strip() or None)

    def kl_privacy(line=""):
        """Columnas con datos personales: %kl_privacy df"""
        return api.scan_privacy(_frame(ipython, line))

    def kl_sql(line="", cell=""):
        """SQL de solo lectura sobre un DataFrame. La tabla se llama df: %%kl_sql nombre_del_dataframe"""
        if not cell.strip():
            raise ValueError("Write the SQL query in the body of the cell.")
        return api.query(_frame(ipython, line), cell)

    ipython.register_magic_function(kl_profile, "line", "kl_profile")
    ipython.register_magic_function(kl_autopilot, "line", "kl_autopilot")
    ipython.register_magic_function(kl_privacy, "line", "kl_privacy")
    ipython.register_magic_function(kl_sql, "cell", "kl_sql")
