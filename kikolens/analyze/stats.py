import pandas as pd
import numpy as np

def basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates enhanced descriptive statistics for the DataFrame.
    Returns numeric health metrics for better styling.
    """
    stats = df.describe(include='all').T
    
    # Add Missing Values info as NUMERIC for styling
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    
    stats['Missing Values'] = null_counts.astype(float)
    stats['Missing %'] = null_pct.round(2) # Keep as float for gradient styling
    
    # Reorder columns to put health metrics first
    cols = ['Missing Values', 'Missing %'] + [c for c in stats.columns if c not in ['Missing Values', 'Missing %']]
    return stats[cols]

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a robust correlation matrix for numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()
    non_constant_cols = [col for col in numeric_df.columns if numeric_df[col].nunique() > 1]
    if not non_constant_cols:
        return pd.DataFrame()
    return numeric_df[non_constant_cols].corr().fillna(0)
