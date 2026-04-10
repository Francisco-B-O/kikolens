import pandas as pd
import numpy as np

def run_discovery_engine(df: pd.DataFrame, max_rows: int = 5000, max_cols: int = 50):
    """
    Scans the dataset for proactive insights, anomalies, and patterns.
    Uses intelligent sampling to keep performance high on large datasets.
    """
    insights = []
    
    # --- Intelligent Sampling ---
    df_sample = df
    if len(df) > max_rows:
        df_sample = df.sample(n=max_rows, random_state=42)
    
    # 1. 🔗 High Correlation Hunter
    numeric_df = df_sample.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        # Limit columns if too wide
        if len(numeric_df.columns) > max_cols:
            # Pick top 50 with highest variance
            cols = numeric_df.var().sort_values(ascending=False).head(max_cols).index
            numeric_df = numeric_df[cols]
            
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find pairs with > 0.85 correlation
        high_corr = [column for column in upper.columns if any(upper[column] > 0.85)]
        for col in high_corr:
            partner = upper[col][upper[col] > 0.85].index[0]
            score = upper[col][partner]
            insights.append({
                "type": "🔗 Correlation Alert",
                "severity": "high",
                "message": f"**{col}** and **{partner}** are strongly linked ({score:.2f}).",
                "detail": "When one changes, the other almost certainly follows. Consider dropping one to reduce redundancy."
            })

    # 2. ⚠️ Anomaly/Outlier Hunter (IQR Method)
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric_df[col] < (Q1 - 1.5 * IQR)) | (numeric_df[col] > (Q3 + 1.5 * IQR))).sum()
        
        # Scale outlier count to full dataset if sampled
        if len(df) > max_rows:
            estimated_outliers = int(outliers * (len(df) / max_rows))
        else:
            estimated_outliers = outliers
            
        if estimated_outliers > 0 and estimated_outliers < (len(df) * 0.05): # Rare outliers (less than 5%)
            insights.append({
                "type": "⚠️ Anomaly Detected",
                "severity": "medium",
                "message": f"Column **{col}** has ~{estimated_outliers} extreme values.",
                "detail": "These 'black sheep' values might distort your average. Check if they are errors or VIP cases."
            })

    # 3. 📊 Dominance Detector (Categorical)
    cat_df = df_sample.select_dtypes(include=['object', 'category'])
    if not cat_df.empty:
        if len(cat_df.columns) > max_cols:
             cols = [c for c in cat_df.columns if cat_df[c].nunique() < 50][:max_cols] # Prioritize low cardinality
             cat_df = cat_df[cols]

        for col in cat_df.columns:
            if cat_df[col].nunique() < 50: # Only low cardinality
                top_val_pct = cat_df[col].value_counts(normalize=True).iloc[0]
                if top_val_pct > 0.85:
                    top_val = cat_df[col].value_counts().index[0]
                    insights.append({
                        "type": "📊 Low Variance",
                        "severity": "low",
                        "message": f"Column **{col}** is dominated by '{top_val}' ({top_val_pct:.1%}).",
                        "detail": "This column hardly varies. It might not be useful for prediction models."
                    })

    # 4. 🆔 ID Column Detector
    for col in df_sample.columns:
        if df_sample[col].nunique() == len(df_sample):
            insights.append({
                "type": "🆔 Unique Identifier",
                "severity": "info",
                "message": f"**{col}** appears to be a unique ID.",
                "detail": "This is great for tracking, but remember to exclude it from ML training."
            })

    return insights
