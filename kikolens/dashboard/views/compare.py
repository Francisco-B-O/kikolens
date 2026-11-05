import pandas as pd
import streamlit as st

from kikolens.analyze import drift as dr
from kikolens.dashboard import components as ui
from kikolens.utils.loaders import load_data

TITLE = "Compare data"
SUBTITLE = "See what changed between two datasets, two periods or the raw and the cleaned data."

LEVEL_NOTE = {"low": ("ok", "The data look the same."), "medium": ("medium", "Some columns have changed."),
              "high": ("high", "The data have changed a lot.")}


def _choose(df: pd.DataFrame):
    options = ["First part against the rest", "Original file against the working data", "Another file against the open one"]
    mode = st.radio("Compare", options, key="cp_mode")
    if mode == options[0]:
        fraction = st.slider("Share of rows used as the reference", 0.1, 0.9, 0.5, 0.05, key="cp_frac")
        ordered = df
        dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dates:
            by = st.selectbox("Order rows by", ["(row order)"] + dates, key="cp_order")
            if by != "(row order)":
                ordered = df.sort_values(by)
        return dr.split_by_position(ordered, fraction) + ("reference", "later part")
    if mode == options[1]:
        original = st.session_state.original_df
        if original is None:
            return None
        return original, df, "original file", "working data"
    file = st.file_uploader("File to compare", key="cp_file")
    if file is None:
        return None
    return df, load_data(file), "open dataset", file.name


def render(df: pd.DataFrame) -> None:
    try:
        chosen = _choose(df)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    except Exception as e:
        ui.note(f"Could not read the file: {e}", "high")
        return
    if not chosen:
        return
    reference, current, ref_name, cur_name = chosen
    try:
        result = dr.compare_datasets(reference, current)
    except ValueError as e:
        ui.note(str(e), "high")
        return

    kind, message = LEVEL_NOTE[result["level"]]
    ui.note(f"{message} {result['summary']}", kind)
    ui.stats([("Drifted", result["drifted"]), ("Moderate", result["moderate"]), ("Stable", result["stable"]),
              (ref_name, f"{result['rows'][0]:,} rows"), (cur_name, f"{result['rows'][1]:,} rows")])
    if result["only_in_reference"]:
        ui.note(f"Only in {ref_name}: {', '.join(map(str, result['only_in_reference']))}", "medium")
    if result["only_in_current"]:
        ui.note(f"Only in {cur_name}: {', '.join(map(str, result['only_in_current']))}", "medium")
    if result["type_changes"]:
        ui.note(f"Changed between numbers and text: {', '.join(map(str, result['type_changes']))}", "high")

    table = result["table"]
    shown = table.rename(columns={"psi": "PSI", "p_value": "p-value"})
    ui.table(shown[["column", "type", "verdict", "PSI", "p-value", "change", "missing % (ref)", "missing % (now)"]].round(4),
             hide_index=True)
    st.caption("PSI below 0.1 is stable, 0.1 to 0.25 moderate, above 0.25 a real shift in the distribution.")
    plottable = table.loc[table["type"].isin(["numeric", "category"]), "column"].tolist()
    if plottable:
        column = st.selectbox("Look at a column", plottable, key="cp_col")
        ui.chart(dr.distribution_figure(reference, current, column), 360)
    st.caption("Dates and columns that are different in every row (identifiers) are listed as info: they are expected to change.")
    ui.download_csv(table, "drift_report.csv", "Download report")
