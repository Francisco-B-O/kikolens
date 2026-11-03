import pandas as pd
import streamlit as st

from kikolens.analyze import quality as q
from kikolens.dashboard import components as ui
from kikolens.dashboard import state

TITLE = "Quality and cleaning"
SUBTITLE = "Find what is wrong with the data, fix it, and undo anything."


def _run(df, op, **params):
    try:
        new_df, message = q.apply_operation(df, op, **params)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    state.apply_change(new_df, message)
    st.rerun()


def _workbench(df: pd.DataFrame, profile: dict) -> None:
    op = st.selectbox("Operation", [
        "Fill missing values", "Convert column type", "Clip outliers", "Drop duplicate rows",
        "Drop rows with missing values", "Drop columns", "Drop constant columns",
        "Trim whitespace in text", "Standardize column names",
    ], key="q_op")

    if op == "Fill missing values":
        with_nulls = df.columns[df.isna().any()].tolist()
        if not with_nulls:
            ui.note("No missing values.", "ok")
            return
        col = st.selectbox("Column", with_nulls, key="q_fill_col")
        numeric = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
        strategy = st.radio("Strategy", q.FILL_NUMERIC if numeric else q.FILL_OTHER, horizontal=True, key="q_fill_s")
        value = st.text_input("Value", key="q_fill_v") if strategy == "constant" else None
        if numeric and strategy == "constant":
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = None
        if st.button("Apply", key="q_apply_fill", type="primary"):
            _run(df, "fill_missing", column=col, strategy=strategy, value=value)
    elif op == "Convert column type":
        col = st.selectbox("Column", df.columns, key="q_conv_col")
        hint = profile["columns"].set_index("column").loc[col, "suspected_type"]
        options = ["numeric", "datetime", "boolean", "text"]
        target = st.selectbox("Convert to", options, index=options.index(hint) if hint in options else 0, key="q_conv_t")
        if st.button("Apply", key="q_apply_conv", type="primary"):
            _run(df, "convert_type", column=col, target=target)
    elif op == "Clip outliers":
        numeric = df.select_dtypes("number").columns.tolist()
        if not numeric:
            ui.note("No numeric columns.", "low")
            return
        col = st.selectbox("Column", numeric, key="q_clip_col")
        factor = st.slider("IQR factor", 1.0, 3.0, 1.5, 0.1, key="q_clip_f")
        if st.button("Apply", key="q_apply_clip", type="primary"):
            _run(df, "clip_outliers", column=col, factor=factor)
    elif op == "Drop columns":
        cols = st.multiselect("Columns", df.columns, key="q_drop_cols")
        if st.button("Apply", key="q_apply_drop", type="primary") and cols:
            _run(df, "drop_columns", columns=cols)
    else:
        names = {
            "Drop duplicate rows": "drop_duplicates", "Drop rows with missing values": "drop_missing_rows",
            "Drop constant columns": "drop_constant_columns", "Trim whitespace in text": "strip_whitespace",
            "Standardize column names": "standardize_column_names",
        }
        if st.button("Apply", key="q_apply_simple", type="primary"):
            _run(df, names[op])


def render(df: pd.DataFrame) -> None:
    profile = q.profile_quality(df)
    ui.stats([
        ("Quality score", f"{profile['score']:.0f}/100"),
        ("Missing cells", f"{profile['missing_pct']:.1f}%", f"{profile['missing_cells']:,}"),
        ("Duplicate rows", f"{profile['duplicate_rows']:,}"),
        ("Constant columns", len(profile["constant_columns"])),
        ("Mistyped columns", len(profile["mistyped_columns"])),
    ])

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Issues found")
        if not profile["issues"]:
            ui.note("No issues detected.", "ok")
        for issue in profile["issues"]:
            ui.note(issue["message"], issue["severity"])
        st.markdown("#### Column profile")
        shown = profile["columns"].drop(columns="constant").rename(
            columns={"missing_pct": "missing %", "suspected_type": "looks like"})
        ui.table(shown, hide_index=True)

    with right:
        st.markdown("#### Fix")
        a, b, c = st.columns(3)
        if a.button("Auto-clean", key="q_auto", type="primary", help="Trim text, drop duplicates, fix types, drop constant columns."):
            new_df, log = q.auto_clean(df)
            state.apply_change(new_df, "Auto-clean: " + " ".join(log))
            st.rerun()
        if b.button("Undo", key="q_undo", disabled=not state.can_undo()):
            state.undo()
            st.rerun()
        if c.button("Reset", key="q_reset", disabled=not state.can_undo()):
            state.reset()
            st.rerun()
        _workbench(df, profile)

        st.markdown("#### History")
        log = st.session_state.clean_log
        if log:
            for line in reversed(log):
                st.caption(line)
        else:
            st.caption("No changes yet. The original file is never modified.")
        ui.download_csv(df, "cleaned.csv", "Download current data (CSV)")
