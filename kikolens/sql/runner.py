import pandas as pd
import duckdb

def run_sql_query(filepath_or_df: any, query: str) -> pd.DataFrame:
    """
    Executes a SQL query on a given DataFrame or file path using DuckDB.
    Uses an explicit connection and table registration for stability.
    """
    con = None
    try:
        # 1. Validation: Ensure query is a SELECT/Read-only
        if not query.strip().upper().startswith(("SELECT", "WITH", "VALUES", "PRAGMA", "DESCRIBE")):
            raise ValueError("KikoLens Security: Only read-only queries (SELECT) are allowed.")
        
        # 2. Initialize In-Memory Connection
        con = duckdb.connect(database=':memory:')
        
        # 3. Register Data Context
        if isinstance(filepath_or_df, pd.DataFrame):
            # Register DataFrame as a virtual table named 'df'
            con.register('df', filepath_or_df)
            result = con.execute(query).df()
        else:
            # For files, we use DuckDB's direct read capabilities
            source = str(filepath_or_df).replace("\\", "/")
            if source.endswith('.csv'):
                # We can't register a file as 'df' easily in one go without creating a view
                con.execute(f"CREATE OR REPLACE VIEW df AS SELECT * FROM read_csv_auto('{source}')")
                result = con.execute(query).df()
            elif source.endswith('.parquet'):
                con.execute(f"CREATE OR REPLACE VIEW df AS SELECT * FROM read_parquet('{source}')")
                result = con.execute(query).df()
            else:
                 # Fallback: Load to pandas first, then register
                 df_temp = pd.read_excel(source)
                 con.register('df', df_temp)
                 result = con.execute(query).df()
                 
        return result
    except Exception as e:
        raise ValueError(f"SQL Execution Failed: {e}")
    finally:
        # Ensure connection is closed to free memory
        try: con.close()
        except: pass
