import io

import pandas as pd
import requests

from kikolens.config import ollama_generate_url, ollama_model, ollama_timeout


def _extract_sql(text: str) -> str:
    """Quita los bloques markdown que el modelo suele añadir."""
    text = text.strip()
    for fence in ("```sql", "```"):
        if fence in text:
            return text.split(fence, 1)[1].split("```")[0].strip()
    return text


def generate_sql_from_text(df: pd.DataFrame, question: str, model_name: str = None) -> str:
    """Traduce una pregunta en lenguaje natural a una consulta DuckDB sobre la tabla `df`."""
    buffer = io.StringIO()
    df.info(buf=buffer)

    try:
        sample = df.head(3).to_markdown()
    except ImportError:  # tabulate no instalado
        sample = df.head(3).to_string()

    prompt = f"""
    You are a professional SQL expert. Generate a DuckDB SQL query to answer the user's question based on the following DataFrame schema.

    ### SCHEMA INFO:
    {buffer.getvalue()}

    ### SAMPLE DATA:
    {sample}

    ### TABLE NAME:
    The table is always called 'df'.

    ### USER QUESTION:
    {question}

    ### INSTRUCTIONS:
    - Return ONLY the SQL code block. No explanations.
    - Use DuckDB syntax.
    - If you need to compare strings, use ILIKE for case-insensitivity.
    - Be precise with column names.

    ### SQL QUERY:
    """

    try:
        response = requests.post(
            ollama_generate_url(),
            json={"model": model_name or ollama_model(), "prompt": prompt, "stream": False},
            timeout=ollama_timeout(),
        )
        response.raise_for_status()
        return _extract_sql(response.json().get("response", ""))
    except Exception as e:
        return f"-- Error generating SQL: {e}"
