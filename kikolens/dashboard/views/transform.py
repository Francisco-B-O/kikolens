import pandas as pd
import streamlit as st

from kikolens.analyze import transform as t
from kikolens.dashboard import components as ui
from kikolens.dashboard import state
from kikolens.utils.loaders import load_data

TITLE = "Transform"
SUBTITLE = "Filter, create columns, reshape and combine datasets. Every step is saved in the recipe."


def _apply(op: str, **params) -> None:
    try:
        state.run_operation(op, **params)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    st.rerun()


def _go(label: str, key: str) -> bool:
    return st.button(label, key=key, type="primary")


def _numeric(df):
    return df.select_dtypes("number").columns.tolist()


def _rows(df: pd.DataFrame) -> None:
    kind = st.radio("Action", ["Filter", "Sort", "Sample"], horizontal=True, key="tf_rows_kind")
    if kind == "Filter":
        c1, c2, c3 = st.columns([2, 2, 3])
        col = c1.selectbox("Column", df.columns, key="tf_f_col")
        op = c2.selectbox("Condition", t.FILTER_OPERATORS, key="tf_f_op")
        value = None if op.startswith("is ") else c3.text_input("Value (comma separated for 'in')", key="tf_f_val")
        if _go("Apply filter", "tf_f_go"):
            _apply("filter_rows", column=col, op=op, value=value)
    elif kind == "Sort":
        cols = st.multiselect("Sort by", df.columns, key="tf_s_cols")
        ascending = st.radio("Order", ["Ascending", "Descending"], horizontal=True, key="tf_s_ord") == "Ascending"
        if _go("Sort", "tf_s_go") and cols:
            _apply("sort_rows", by=cols, ascending=ascending)
    else:
        mode = st.radio("Take", ["A fraction", "A number of rows"], horizontal=True, key="tf_sm_mode")
        if mode == "A fraction":
            frac = st.slider("Fraction of rows", 0.01, 1.0, 0.5, 0.01, key="tf_sm_frac")
            if _go("Sample", "tf_sm_go"):
                _apply("sample_rows", fraction=frac)
        else:
            n = st.number_input("Rows", 1, max(len(df), 1), min(1000, max(len(df), 1)), key="tf_sm_n")
            if _go("Sample", "tf_sm_go2"):
                _apply("sample_rows", n=int(n))


def _columns(df: pd.DataFrame) -> None:
    kind = st.radio("Action", ["Formula", "Combine text", "Split text", "Rename", "Replace values"], horizontal=True,
                    key="tf_c_kind")
    if kind == "Formula":
        name = st.text_input("New column name", key="tf_a_name")
        expr = st.text_input("Formula", placeholder="price * units    or    log1p(`unit price`) / 2", key="tf_a_expr")
        st.caption("Use column names, numbers, + - * / // % **, comparisons and log, log1p, sqrt, abs, exp, round, "
                   "floor, ceil, min, max. Put names with spaces between `backticks`.")
        if _go("Create column", "tf_a_go"):
            _apply("add_column", name=name, expression=expr)
    elif kind == "Combine text":
        cols = st.multiselect("Columns to combine", df.columns, key="tf_cc_cols")
        c1, c2 = st.columns(2)
        name = c1.text_input("New column name", key="tf_cc_name")
        sep = c2.text_input("Separator", " ", key="tf_cc_sep")
        if _go("Combine", "tf_cc_go"):
            _apply("concat_columns", columns=cols, name=name, separator=sep)
    elif kind == "Split text":
        col = st.selectbox("Column", df.columns, key="tf_sp_col")
        c1, c2 = st.columns(2)
        sep = c1.text_input("Separator", "-", key="tf_sp_sep")
        names = c2.text_input("New column names (comma separated)", key="tf_sp_names")
        if _go("Split", "tf_sp_go"):
            _apply("split_column", column=col, separator=sep, names=[n.strip() for n in names.split(",") if n.strip()])
    elif kind == "Rename":
        col = st.selectbox("Column", df.columns, key="tf_rn_col")
        new = st.text_input("New name", key="tf_rn_new").strip()
        if _go("Rename", "tf_rn_go") and new:
            _apply("rename_columns", mapping={col: new})
    else:
        col = st.selectbox("Column", df.columns, key="tf_rv_col")
        c1, c2 = st.columns(2)
        old = c1.text_input("Replace this value", key="tf_rv_old")
        new = c2.text_input("With", key="tf_rv_new")
        if _go("Replace", "tf_rv_go") and old != "":
            _apply("replace_values", column=col, mapping={old: new})


def _features(df: pd.DataFrame) -> None:
    kind = st.radio("Action", ["Date parts", "Bins", "Scale", "Log", "Rank", "One-hot"], horizontal=True, key="tf_ft_kind")
    nums = _numeric(df)
    if kind == "Date parts":
        col = st.selectbox("Date column", df.columns, key="tf_dp_col")
        parts = st.multiselect("Parts", t.DATE_PARTS, default=["year", "month", "weekday"], key="tf_dp_parts")
        if _go("Extract", "tf_dp_go"):
            _apply("extract_date_parts", column=col, parts=parts)
        return
    if kind == "One-hot":
        col = st.selectbox("Column", df.columns, key="tf_oh_col")
        if _go("Encode", "tf_oh_go"):
            _apply("one_hot_encode", column=col)
        return
    if not nums:
        ui.note("This needs a numeric column.", "low")
        return
    col = st.selectbox("Column", nums, key="tf_n_col")
    if kind == "Bins":
        c1, c2 = st.columns(2)
        n = c1.slider("Bins", 2, 20, 5, key="tf_b_n")
        method = c2.radio("Method", ["quantile", "uniform"], horizontal=True, key="tf_b_m")
        if _go("Create bins", "tf_b_go"):
            _apply("bin_column", column=col, bins=n, method=method)
    elif kind == "Scale":
        method = st.radio("Method", t.SCALE_METHODS, horizontal=True, key="tf_sc_m")
        if _go("Scale", "tf_sc_go"):
            _apply("scale_column", column=col, method=method)
    elif kind == "Log":
        if _go("Apply log(1 + x)", "tf_lg_go"):
            _apply("log_transform", column=col)
    else:
        if _go("Add rank", "tf_rk_go"):
            _apply("rank_column", column=col, pct=True)


def _sequence(df: pd.DataFrame) -> None:
    nums = _numeric(df)
    if not nums:
        ui.note("This needs a numeric column.", "low")
        return
    kind = st.radio("Action", ["Lagged values", "Rolling statistic"], horizontal=True, key="tf_sq_kind")
    c1, c2, c3 = st.columns(3)
    col = c1.selectbox("Column", nums, key="tf_sq_col")
    order = c2.selectbox("Order by", ["(row order)"] + list(df.columns), key="tf_sq_ord")
    group = c3.selectbox("Within group", ["(none)"] + list(df.columns), key="tf_sq_grp")
    extra = {"order_by": None if order.startswith("(") else order, "group_by": None if group.startswith("(") else group}
    if kind == "Lagged values":
        lags = st.text_input("Lags (comma separated)", "1, 7", key="tf_lag_l")
        if _go("Add lags", "tf_lag_go"):
            try:
                values = [int(v) for v in lags.split(",") if v.strip()]
            except ValueError:
                ui.note("Lags must be whole numbers.", "high")
                return
            _apply("lag_features", column=col, lags=values, **extra)
    else:
        c1, c2 = st.columns(2)
        window = c1.number_input("Window (rows)", 2, 1000, 7, key="tf_rl_w")
        stat = c2.selectbox("Statistic", ["mean", "sum", "min", "max", "std", "median"], key="tf_rl_s")
        if _go("Add rolling statistic", "tf_rl_go"):
            _apply("rolling_feature", column=col, window=int(window), stat=stat, **extra)


def _reshape(df: pd.DataFrame) -> None:
    kind = st.radio("Action", ["Group and summarize", "Pivot", "Stack columns"], horizontal=True, key="tf_rs_kind")
    nums = _numeric(df)
    if kind == "Group and summarize":
        by = st.multiselect("Group by", df.columns, key="tf_g_by")
        cols = st.multiselect("Summarize", [c for c in nums if c not in by], key="tf_g_cols")
        funcs = st.multiselect("With", t.AGGREGATIONS, default=["sum", "mean"], key="tf_g_funcs")
        if _go("Group", "tf_g_go") and by and cols and funcs:
            _apply("group_aggregate", by=by, aggregations={c: funcs for c in cols})
        ui.note("This replaces the data with one row per group. Undo brings it back.", "low")
    elif kind == "Pivot":
        if not nums:
            ui.note("This needs a numeric column to aggregate.", "low")
            return
        c1, c2, c3, c4 = st.columns(4)
        idx = c1.selectbox("Rows", df.columns, key="tf_pv_i")
        col = c2.selectbox("Columns", df.columns, index=min(1, len(df.columns) - 1), key="tf_pv_c")
        val = c3.selectbox("Values", nums, key="tf_pv_v")
        agg = c4.selectbox("Aggregate", t.AGGREGATIONS, key="tf_pv_a")
        if _go("Pivot", "tf_pv_go"):
            _apply("pivot_table", index=idx, columns=col, values=val, aggfunc=agg)
    else:
        ids = st.multiselect("Keep as identifiers", df.columns, key="tf_m_ids")
        vals = st.multiselect("Columns to stack", [c for c in df.columns if c not in ids], key="tf_m_vals")
        if _go("Stack", "tf_m_go") and vals:
            _apply("melt_columns", id_columns=ids, value_columns=vals)


def _combine(df: pd.DataFrame) -> None:
    datasets = st.session_state.datasets
    upload = st.file_uploader("Add another dataset", key="tf_up")
    if upload is not None and upload.name not in datasets:
        try:
            datasets[upload.name] = load_data(upload)
        except Exception as e:
            ui.note(str(e), "high")
    if not datasets:
        ui.note("Upload a second file to join it to this one or append its rows.", "low")
        return
    name = st.selectbox("Dataset", list(datasets), key="tf_cb_name")
    other = datasets[name]
    st.caption(f"{len(other):,} rows · columns: {', '.join(map(str, other.columns))}")
    kind = st.radio("Action", ["Join on key", "Append rows"], horizontal=True, key="tf_cb_kind")
    if kind == "Join on key":
        shared = [c for c in df.columns if c in other.columns]
        keys = st.multiselect("Key column(s)", shared, key="tf_j_keys")
        how = st.radio("Keep", ["left", "inner", "outer", "right"], horizontal=True, key="tf_j_how")
        if _go("Join", "tf_j_go") and keys:
            _apply("merge_datasets", on=keys, how=how, other=name)
    elif _go("Append", "tf_ap_go"):
        _apply("append_dataset", other=name)


def render(df: pd.DataFrame) -> None:
    ui.stats([("Rows", f"{len(df):,}"), ("Columns", len(df.columns)), ("Steps in recipe", len(st.session_state.recipe))])
    a, b, c = st.columns([1, 1, 6])
    if a.button("Undo", key="tf_undo", disabled=not state.can_undo()):
        state.undo()
        st.rerun()
    if b.button("Reset", key="tf_reset", disabled=not state.can_undo()):
        state.reset()
        st.rerun()

    tabs = st.tabs(["Rows", "Columns", "Features", "Sequences", "Reshape", "Combine"])
    for tab, body in zip(tabs, (_rows, _columns, _features, _sequence, _reshape, _combine)):
        with tab:
            body(df)

    st.markdown("#### Preview")
    ui.table(df.head(200), height=320)
    if st.session_state.clean_log:
        st.markdown("#### What was done")
        for line in reversed(st.session_state.clean_log):
            st.caption(line)
