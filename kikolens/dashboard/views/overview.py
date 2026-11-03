import pandas as pd
import streamlit as st

from kikolens.analyze.quality import profile_quality
from kikolens.dashboard import components as ui

TITLE = "Overview"
SUBTITLE = "What is in this dataset, at a glance."


def render(df: pd.DataFrame) -> None:
    q = profile_quality(df)
    ui.stats([
        ("Rows", f"{q['rows']:,}"),
        ("Columns", q["columns_count"]),
        ("Quality score", f"{q['score']:.0f}/100", f"{len(q['issues'])} issue(s)"),
        ("Missing cells", f"{q['missing_pct']:.1f}%", f"{q['missing_cells']:,} cells"),
        ("Memory", f"{q['memory_mb']:.1f} MB"),
    ])

    kinds = pd.Series({
        "Numeric": df.select_dtypes("number").shape[1],
        "Text / categorical": df.select_dtypes(include=["object", "category", "string"]).shape[1],
        "Date / time": df.select_dtypes(include=["datetime", "datetimetz"]).shape[1],
        "Boolean": df.select_dtypes("bool").shape[1],
    })
    st.caption("Column types: " + " · ".join(f"{k} {v}" for k, v in kinds.items() if v))

    st.markdown("#### Columns")
    columns = q["columns"].rename(columns={"missing_pct": "missing %", "suspected_type": "looks like"})
    ui.table(
        columns[["column", "dtype", "missing %", "unique", "outliers", "looks like"]],
        hide_index=True,
        column_config={"missing %": st.column_config.ProgressColumn("missing %", min_value=0, max_value=100, format="%.1f")},
    )

    st.markdown("#### Preview")
    rows = len(df)
    if len(df) > 5:
        rows = st.slider("Rows to show", 5, min(200, len(df)), min(20, len(df)), key="ov_rows")
    ui.table(df.head(rows), hide_index=False)
