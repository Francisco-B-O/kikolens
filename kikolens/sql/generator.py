import os
import pandas as pd
import requests
import io

def generate_sql_from_text(df: pd.DataFrame, question: str, model_name: str = "llama3.1") -> str:
    """
    Translates a natural language question into a DuckDB SQL query.
    Provides the AI with the actual schema and sample data for high precision.
    """
    # 1. Capture Schema Context
    buffer = io.StringIO()
    df.info(buf=buffer)
    schema_info = buffer.getvalue()
    
    # 2. Capture Value Context (Sample of top values for categoricals)
    sample_data = df.head(3).to_markdown()
    
    prompt = f"""
    You are a professional SQL expert. Generate a DuckDB SQL query to answer the user's question based on the following DataFrame schema.
    
    ### SCHEMA INFO:
    {schema_info}
    
    ### SAMPLE DATA:
    {sample_data}
    
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

    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        ollama_url = f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api/generate"
        response = requests.post(ollama_url, json=data, timeout=10)
        response.raise_for_status()
        sql = response.json().get("response", "").strip()
        # Clean markdown wrappers if present
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql:
            sql = sql.split("```")[1].split("```")[0].strip()
        return sql
    except Exception as e:
        return f"-- Error generating SQL: {e}"
