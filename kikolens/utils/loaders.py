import pandas as pd
import json
import os

def load_data(source) -> pd.DataFrame:
    # 0. Handle Streamlit UploadedFile objects
    if hasattr(source, "name"):
        filename = source.name
        if filename.endswith('.csv'):
            return pd.read_csv(source)
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(source)
        elif filename.endswith('.json'):
            return pd.read_json(source)
        raise ValueError(f"Unsupported uploaded file format: {filename}")

    source_str = str(source).replace("\\", "/")
    
    # 1. Handle SQL/Remote URIs
    sql_prefixes = ('postgresql://', 'mysql://', 'sqlite:', 'oracle://', 'mssql://', 'mariadb://')
    
    if source_str.lower().startswith(sql_prefixes) or "://" in source_str:
        # Aggressive SQLite protocol fix for Windows
        if source_str.lower().startswith("sqlite:"):
            path_part = source_str.split(":", 1)[1].lstrip("/")
            source_str = f"sqlite:///{path_part}"
        
        try:
            from sqlalchemy import create_engine, inspect
            engine = create_engine(source_str)
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            if not tables:
                raise ValueError("No tables found in the database.")
            return pd.read_sql(f"SELECT * FROM {tables[0]}", engine)
        except ImportError as ie:
            if "mysql" in source_str:
                raise ImportError("MySQL Driver missing. Please run: pip install pymysql cryptography")
            elif "postgresql" in source_str:
                raise ImportError("PostgreSQL Driver missing. Please run: pip install psycopg2-binary")
            else:
                raise ie
        except Exception as e:
            raise ValueError(f"Database connection failed: {e}")

    # 2. Local Files
    if source_str.endswith('.csv'):
        return pd.read_csv(source_str)
    elif source_str.endswith(('.xlsx', '.xls')):
        return pd.read_excel(source_str)
    elif source_str.endswith('.json'):
        # Intentar primero pd.read_json (JSON arrays estándar)
        # Si falla, asumir NDJSON (una línea por objeto)
        try:
            return pd.read_json(source_str)
        except Exception:
            with open(source_str, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
            return pd.DataFrame(data)
    
    raise ValueError(f"Unsupported source format: {source_str}")
