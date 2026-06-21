import pandas as pd
import numpy as np


def _detect_correlations(numeric_df, max_cols):
    insights = []
    if numeric_df.empty:
        return insights

    if len(numeric_df.columns) > max_cols:
        cols = numeric_df.var().sort_values(ascending=False).head(max_cols).index
        numeric_df = numeric_df[cols]

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    for col in [c for c in upper.columns if any(upper[c] > 0.85)]:
        matches = upper[col][upper[col] > 0.85].index
        if len(matches) == 0:
            continue
        partner = matches[0]
        score = upper[col][partner]
        insights.append({
            "type": "🔗 Correlation Alert",
            "severity": "high",
            "message": f"**{col}** and **{partner}** are strongly linked ({score:.2f}).",
            "detail": "When one changes, the other almost certainly follows. Consider dropping one to reduce redundancy."
        })

    return insights


def _detect_outliers(numeric_df, df_len, max_rows):
    insights = []

    for col in numeric_df.columns:
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((numeric_df[col] < (q1 - 1.5 * iqr)) | (numeric_df[col] > (q3 + 1.5 * iqr))).sum()

        estimated = int(outliers * (df_len / max_rows)) if df_len > max_rows else outliers

        if 0 < estimated < df_len * 0.05:
            insights.append({
                "type": "⚠️ Anomaly Detected",
                "severity": "medium",
                "message": f"Column **{col}** has ~{estimated} extreme values.",
                "detail": "These 'black sheep' values might distort your average. Check if they are errors or VIP cases."
            })

    return insights


def _detect_dominance(cat_df, max_cols):
    insights = []
    if cat_df.empty:
        return insights

    if len(cat_df.columns) > max_cols:
        cat_df = cat_df[[c for c in cat_df.columns if cat_df[c].nunique() < 50][:max_cols]]

    for col in cat_df.columns:
        if cat_df[col].nunique() >= 50:
            continue
        top_val_pct = cat_df[col].value_counts(normalize=True).iloc[0]
        if top_val_pct > 0.85:
            top_val = cat_df[col].value_counts().index[0]
            insights.append({
                "type": "📊 Low Variance",
                "severity": "low",
                "message": f"Column **{col}** is dominated by '{top_val}' ({top_val_pct:.1%}).",
                "detail": "This column hardly varies. It might not be useful for prediction models."
            })

    return insights


def _detect_ids(df_sample):
    insights = []
    for col in df_sample.columns:
        if df_sample[col].nunique() == len(df_sample):
            insights.append({
                "type": "🆔 Unique Identifier",
                "severity": "info",
                "message": f"**{col}** appears to be a unique ID.",
                "detail": "This is great for tracking, but remember to exclude it from ML training."
            })
    return insights


def run_discovery_engine(df: pd.DataFrame, max_rows: int = 5000, max_cols: int = 50):
    df_sample = df.sample(n=max_rows, random_state=42) if len(df) > max_rows else df
    numeric_df = df_sample.select_dtypes(include=[np.number])
    cat_df = df_sample.select_dtypes(include=["object", "category"])

    insights = []
    insights += _detect_correlations(numeric_df, max_cols)
    insights += _detect_outliers(numeric_df, len(df), max_rows)
    insights += _detect_dominance(cat_df, max_cols)
    insights += _detect_ids(df_sample)

    return insights
