import pandas as pd
import streamlit as st

from kikolens.analyze import cleaning as cl
from kikolens.analyze import quality as q
from kikolens.analyze import rules as rl
from kikolens.dashboard import components as ui
from kikolens.dashboard import state
from kikolens.utils.dtypes import is_text

TITLE = "Quality and cleaning"
SUBTITLE = "Find what is wrong with the data, fix it, and undo anything."


def _run(df, op, **params):
    try:
        state.run_operation(op, **params)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    st.rerun()


def _workbench(df: pd.DataFrame, profile: dict) -> None:
    op = st.selectbox("Operation", [
        "Fill missing values", "Convert column type", "Clip outliers", "Drop duplicate rows",
        "Drop rows with missing values", "Drop columns", "Drop constant columns",
        "Trim whitespace in text", "Standardize column names", "Smart fill (KNN, iterative, by group)",
        "Merge similar spellings",
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
    elif op == "Smart fill (KNN, iterative, by group)":
        with_nulls = df.columns[df.isna().any()].tolist()
        if not with_nulls:
            ui.note("No missing values.", "ok")
            return
        cols = st.multiselect("Columns to fill", with_nulls, default=with_nulls, key="q_sf_cols")
        method = st.radio("Method", cl.IMPUTE_METHODS, horizontal=True, key="q_sf_m",
                          help="knn: from the most similar rows. iterative: predicts each gap from the other columns. "
                               "group: the median or most common value of each group.")
        group = st.selectbox("Groups", [c for c in df.columns if df[c].nunique() <= 30], key="q_sf_g") if method == "group" else None
        indicator = st.checkbox("Add a column marking the filled cells", key="q_sf_i")
        if st.button("Apply", key="q_apply_sf", type="primary") and cols:
            _run(df, "impute_advanced", method=method, columns=cols, group_by=group, add_indicator=indicator)
    elif op == "Merge similar spellings":
        texts = [c for c in df.columns if is_text(df[c])]
        if not texts:
            ui.note("No text columns.", "low")
            return
        col = st.selectbox("Column", texts, key="q_ms_col")
        threshold = st.slider("Similarity", 0.6, 1.0, 0.85, 0.01, key="q_ms_t")
        try:
            groups = cl.similar_values(df[col], threshold)
        except ValueError as e:
            ui.note(str(e), "high")
            return
        if groups.empty:
            ui.note("No similar spellings found.", "ok")
        else:
            ui.table(groups, hide_index=True, height=200)
            if st.button("Merge them", key="q_apply_ms", type="primary"):
                _run(df, "standardize_categories", column=col, threshold=threshold)
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


def _more_checks(df: pd.DataFrame) -> None:
    st.divider()
    a, b, c = st.tabs(["Missing data patterns", "Near-duplicate rows", "Validation rules"])
    with a:
        info = cl.missing_patterns(df)
        ui.note(info["summary"], "ok" if info["by_column"].empty else "medium" if len(info["related"]) else "low")
        if not info["by_column"].empty:
            ui.table(info["by_column"].round(2), hide_index=True)
            if not info["related"].empty:
                st.markdown("**Gaps that depend on another column**")
                ui.table(info["related"].round(4), hide_index=True)
            if not info["co_missing"].empty:
                st.markdown("**Columns that are empty together** (1 means always together)")
                ui.table(info["co_missing"].round(2))
            ui.table(info["patterns"].round(3), hide_index=True)
    with b:
        cols = st.multiselect("Compare using", df.columns, default=list(df.columns[:8]), key="q_nd_cols")
        threshold = st.slider("Similarity", 0.6, 0.99, 0.9, 0.01, key="q_nd_t")
        if st.button("Look for near-duplicates", key="q_nd_go"):
            try:
                st.session_state.q_nd = cl.near_duplicate_rows(df, cols or None, threshold)
            except ValueError as e:
                ui.note(str(e), "high")
        pairs = st.session_state.get("q_nd")
        if pairs is not None:
            ui.note(f"{len(pairs)} pair(s) found." if len(pairs) else "No near-duplicates found.", "medium" if len(pairs) else "ok")
            if len(pairs):
                ui.table(pairs.round(3), hide_index=True)
                drop = st.multiselect("Remove these rows", sorted(set(pairs["row B"])), key="q_nd_drop",
                                      help="Row B of each pair is offered. Check the pairs first.")
                if st.button("Remove selected rows", key="q_nd_apply") and drop:
                    new_df, message = cl.drop_rows(df, drop)
                    state.apply_change(new_df, message)
                    st.session_state.pop("q_nd", None)
                    st.rerun()
    with c:
        _rules(df)


def _rules(df: pd.DataFrame) -> None:
    if st.session_state.get("rules") is None:
        st.session_state.rules = rl.RuleSet()
    rules = st.session_state.rules
    c1, c2 = st.columns(2)
    if c1.button("Suggest rules from this data", key="q_rl_suggest"):
        st.session_state.rules = rl.suggest_rules(df)
        st.rerun()
    upload = c2.file_uploader("Open rules (JSON)", type=["json"], key="q_rl_open", label_visibility="collapsed")
    if upload is not None and st.session_state.get("q_rl_loaded") != upload.file_id:
        try:
            st.session_state.rules = rl.RuleSet.from_json(upload.getvalue().decode("utf-8"))
            st.session_state.q_rl_loaded = upload.file_id
            st.rerun()
        except (ValueError, UnicodeDecodeError) as e:
            ui.note(str(e), "high")
    with st.expander("Add a rule"):
        check = st.selectbox("Rule", rl.CHECKS, key="q_rl_check")
        col = st.selectbox("Column", df.columns, key="q_rl_col")
        rule = {"check": check, "column": col}
        if check == "range":
            x, y = st.columns(2)
            lo, hi = x.text_input("Minimum", key="q_rl_min"), y.text_input("Maximum", key="q_rl_max")
            try:
                rule.update(min=float(lo) if lo else None, max=float(hi) if hi else None)
            except ValueError:
                ui.note("Minimum and maximum must be numbers.", "high")
        elif check == "in_set":
            rule["values"] = [v.strip() for v in st.text_input("Allowed values (comma separated)", key="q_rl_vals").split(",") if v.strip()]
        elif check == "regex":
            rule["pattern"] = st.text_input("Pattern (the whole value must match)", key="q_rl_pat")
        elif check == "compare":
            x, y = st.columns(2)
            rule["op"] = x.selectbox("Must be", list(rl.COMPARATORS), key="q_rl_op")
            rule["other"] = y.selectbox("This column", [c for c in df.columns if c != col], key="q_rl_other")
        if st.button("Add rule", key="q_rl_add"):
            try:
                rl.violations(df, rule)
                rules.add(rule)
            except ValueError as e:
                ui.note(str(e), "high")
            else:
                st.rerun()
    if not rules.rules:
        st.caption("No rules yet. Add one or let KikoLens suggest some.")
        return
    result = rules.validate(df)
    ui.note(result["summary"], "ok" if result["passed"] else "high")
    ui.table(result["table"].drop(columns="id").round(2), hide_index=True)
    pick = st.selectbox("See the rows that break a rule", [r["id"] for r in rules.rules],
                        format_func=lambda i: rl.describe(next(r for r in rules.rules if r["id"] == i)), key="q_rl_see")
    try:
        bad = rules.violating_rows(df, pick)
        ui.table(bad, height=min(300, 60 + 35 * max(len(bad), 1))) if len(bad) else st.caption("No rows break this rule.")
    except ValueError as e:
        ui.note(str(e), "high")
    st.download_button("Download rules (JSON)", rules.to_json().encode("utf-8"), "rules.json", "application/json", key="q_rl_dl")


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
            new_df, steps = q.auto_clean_steps(df)
            state.apply_steps(new_df, steps, "Auto-clean: " + " ".join(m for _, _, m in steps))
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
    _more_checks(df)
