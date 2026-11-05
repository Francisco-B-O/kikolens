import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kikolens.analyze import text as tx
from kikolens.analyze import textmining as tm
from kikolens.dashboard import components as ui

TITLE = "Text"
SUBTITLE = "Make sense of comments, reviews and free-text answers, all on this computer."


def _terms(series: pd.Series, language: str) -> None:
    info = tx.text_stats(series)
    ui.stats([("Texts", f"{info['rows'] - info['empty']:,}", f"{info['empty']:,} empty"), ("Words per text", f"{info['mean_words']:.1f}"),
              ("Duplicates", f"{info['duplicates']:,}"), ("Language", info["language"])])
    c1, c2 = st.columns(2)
    ngram = c1.radio("Look at", [1, 2, 3], format_func=lambda n: {1: "Words", 2: "Pairs of words", 3: "Triples"}[n],
                     horizontal=True, key="tx_ngram")
    n = c2.slider("Terms to show", 5, 40, 20, key="tx_n")
    try:
        terms = tx.top_terms(series, n, ngram, language)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    fig = go.Figure(go.Bar(x=terms["share"][::-1], y=terms["term"][::-1], orientation="h"))
    fig.update_layout(title="Share of texts that contain each term", xaxis_tickformat=".0%")
    ui.chart(fig, 120 + 22 * len(terms))
    query = st.text_input("Find a word or phrase", key="tx_query")
    if query:
        try:
            found = tx.search_texts(series, query)
        except ValueError as e:
            ui.note(str(e), "high")
        else:
            st.caption(f"{len(found)} match(es) shown.")
            ui.table(found, hide_index=True)


def _by_group(df: pd.DataFrame, col: str, language: str) -> None:
    options = [c for c in df.columns if c != col and 2 <= df[c].nunique() <= 12]
    nums = [c for c in df.select_dtypes("number").columns]
    if options:
        group = st.selectbox("Compare the groups of", options, key="tx_group")
        try:
            table = tx.terms_by_group(df, col, group, language=language)
        except ValueError as e:
            ui.note(str(e), "high")
        else:
            st.caption("Terms that appear much more in one group than in the rest (log odds).")
            ui.table(table.round(3), hide_index=True)
    if nums:
        st.markdown("#### Terms that go with a higher or lower number")
        value = st.selectbox("Number", nums, key="tx_num")
        try:
            ui.table(tx.association_with_numeric(df[col], df[value]).round(4), hide_index=True)
        except ValueError as e:
            ui.note(str(e), "low")


def _clusters(series: pd.Series, language: str) -> None:
    k = st.slider("Number of groups", 2, 12, 5, key="tx_k")
    if st.button("Group the texts", key="tx_cluster", type="primary"):
        try:
            st.session_state.tx_clusters = tx.cluster_texts(series, k, language)
        except ValueError as e:
            ui.note(str(e), "high")
    res = st.session_state.get("tx_clusters")
    if res:
        ui.table(res["summary"].round(3), hide_index=True)
        ui.download_csv(res["labels"].reset_index(names="row"), "text_groups.csv", "Download the group of each text")


def _classify(df: pd.DataFrame, col: str, language: str) -> None:
    targets = [c for c in df.columns if c != col and 2 <= df[c].nunique() <= 20]
    if not targets:
        ui.note("This needs a column with a few categories to learn from (for example a rating or a topic).", "low")
        return
    target = st.selectbox("Learn to predict", targets, key="tx_target")
    if st.button("Train the classifier", key="tx_train", type="primary"):
        try:
            with st.spinner("Training..."):
                st.session_state.tx_clf = (target, tx.train_text_classifier(df, col, target, language))
        except ValueError as e:
            ui.note(str(e), "high")
    cached = st.session_state.get("tx_clf")
    if not cached or cached[0] != target:
        return
    res = cached[1]
    ui.stats([("Accuracy", f"{res['accuracy']:.1%}", "cross-validated"), ("Always guessing the commonest", f"{res['baseline']:.1%}"),
              ("F1 (weighted)", f"{res['f1']:.3f}"), ("Texts used", f"{res['rows']:,}")])
    if res["accuracy"] <= res["baseline"] + 0.02:
        ui.note("The classifier is no better than always guessing the most common class.", "medium")
    st.markdown("**Terms that point to each class**")
    ui.table(res["drivers"].round(3), hide_index=True)
    sample = st.text_area("Try a text", key="tx_try")
    if sample.strip():
        out = tx.classify_texts(res, pd.Series([sample]))
        st.metric("Prediction", str(out["prediction"].iloc[0]), f"{out['confidence'].iloc[0]:.0%} confident")


def _tone(series: pd.Series, language: str) -> None:
    st.caption("A simple word list, so it misses irony and negations. Use it to get a rough picture, not for decisions.")
    try:
        tone = tx.lexicon_sentiment(series, language if language in tx.POSITIVE else None)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    counts = tone["tone"].value_counts().rename_axis("tone").reset_index(name="texts")
    fig = go.Figure(go.Bar(x=counts["tone"], y=counts["texts"]))
    fig.update_layout(title="Tone of the texts")
    ui.chart(fig, 300)


def _summary(series: pd.Series) -> None:
    st.caption("Sends a sample of the texts to the language model running on this computer (Ollama).")
    question = st.text_input("Question", "What are the main themes and complaints?", key="tx_q")
    if st.button("Summarize", key="tx_sum", type="primary"):
        try:
            with st.spinner("Waiting for the local model..."):
                st.session_state.tx_summary = tx.summarize_with_llm(series, question)
        except ValueError as e:
            ui.note(str(e), "high")
    if st.session_state.get("tx_summary"):
        st.markdown(st.session_state.tx_summary)


def _run(key: str, signature, fn):
    """Calcula al pulsar el botón y recuerda el resultado mientras no cambien las opciones."""
    if st.button("Run", key=f"{key}_go", type="primary"):
        try:
            st.session_state[key] = (signature, fn())
        except ValueError as e:
            st.session_state.pop(key, None)
            ui.note(str(e), "high")
    cached = st.session_state.get(key)
    return cached[1] if cached and cached[0] == signature else None


def _notes(res: dict) -> None:
    for note in res.get("notes", []):
        ui.note(note, "medium")


def _topics(df: pd.DataFrame, col: str, language: str) -> None:
    c1, c2, c3 = st.columns(3)
    auto = c1.checkbox("Choose the number of topics for me", True, key="tx_tm_auto")
    k = int(c2.number_input("Topics", 2, tm.MAX_K, 4, key="tx_tm_k", disabled=auto))
    method = c3.radio("Method", ["nmf", "lda"], horizontal=True, key="tx_tmethod", help="NMF is faster and steadier; LDA is the classic probabilistic model.")
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    date = st.selectbox("Follow them over time with", ["(no)"] + dates, key="tx_tdate")
    signature = (col, language, auto, k, method, date)
    chosen = None if auto else k

    def fit():
        if date.startswith("("):
            return tm.topics(df[col], chosen, method=method, language=language)
        return tm.topics_over_time(df, col, date, chosen, method=method, language=language)

    res = _run("tx_topics", signature, fit)
    if res is None:
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    ui.chart(res["figure"], 380 if "over_time" in res else 320)
    ui.table(res["table"].drop(columns="examples").round(3))
    for _, row in res["table"].iterrows():
        with st.expander(f"Topic {row['topic']}: {row['label']}"):
            for example in row["examples"]:
                st.markdown(f"> {example}")
    if len(res["coherence"]) > 1:
        with st.expander("Coherence for each number of topics"):
            ui.table(res["coherence"].round(3))


def _duplicates(series: pd.Series, language: str) -> None:
    c1, c2 = st.columns(2)
    threshold = c1.slider("How alike counts as the same text", 0.5, 1.0, 0.9, 0.01, key="tx_dthr")
    res = _run("tx_dups", (series.name, threshold), lambda: tm.find_duplicates(series, threshold))
    if res is not None:
        ui.note(res["summary"], "ok" if not res["groups"] else "accent")
        _notes(res)
        if res["groups"]:
            ui.table(res["table"].drop(columns="rows").round(3))
    st.subheader("Texts like this one")
    query = st.text_input("A phrase", key="tx_sim_q")
    if query:
        try:
            ui.table(tm.similar_texts(series, query, 8, language).round(3))
        except ValueError as e:
            ui.note(str(e), "high")


def _patterns(series: pd.Series) -> None:
    kinds = st.multiselect("Look for", list(tm.PATTERNS), default=list(tm.PATTERNS), key="tx_pat")
    res = _run("tx_patterns", (series.name, tuple(kinds)), lambda: tm.extract_patterns(series, kinds))
    if res is None:
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    ui.table(res["table"].round(3))
    if len(res["found"]):
        ui.table(res["found"].head(500))
        ui.download_csv(res["found"], "patterns_found.csv", "Download every match (CSV)")


def _phrases(series: pd.Series, language: str) -> None:
    c1, c2 = st.columns(2)
    n = c1.slider("Phrases to show", 5, 40, 15, key="tx_kp_n")
    min_count = c2.slider("A pair of words counts after (times)", 2, 50, 5, key="tx_col_n")
    try:
        ui.table(tm.keyphrases(series, n, language).round(2))
    except ValueError as e:
        ui.note(str(e), "medium")
    st.subheader("Words that go together")
    try:
        ui.table(tm.collocations(series, n, min_count, language).round(3))
    except ValueError as e:
        ui.note(str(e), "medium")


def _readability(series: pd.Series, language: str) -> None:
    try:
        res = tm.readability(series, language)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    ui.note(res["summary"], "ok")
    _notes(res)
    fig = go.Figure(go.Bar(x=list(res["distribution"].index.astype(str)), y=res["distribution"].values))
    fig.update_layout(title="Texts by reading ease", yaxis_title="Texts")
    ui.chart(fig, 300)


def render(df: pd.DataFrame) -> None:
    candidates = tx.free_text_columns(df)
    others = [c for c in df.columns if c not in candidates and pd.api.types.is_object_dtype(df[c])]
    if not candidates and not others:
        ui.note("No text columns found. This section is for columns with sentences, such as comments or reviews.", "low")
        return
    col = st.selectbox("Text column", candidates + others, key="tx_col")
    if col not in candidates:
        ui.note("This column does not look like free text; results may not mean much.", "low")
    series = df[col]
    language = st.radio("Language", list(tx.STOPWORDS), index=list(tx.STOPWORDS).index(tx.detect_language(series)),
                        horizontal=True, key="tx_lang")
    tabs = st.tabs(["Terms", "By group", "Themes", "Classify", "Tone", "Summary", "Topics", "Duplicates and similar", "Patterns", "Phrases", "Readability"])
    with tabs[0]:
        _terms(series, language)
    with tabs[1]:
        _by_group(df, col, language)
    with tabs[2]:
        _clusters(series, language)
    with tabs[3]:
        _classify(df, col, language)
    with tabs[4]:
        _tone(series, language)
    with tabs[5]:
        _summary(series)
    with tabs[6]:
        _topics(df, col, language)
    with tabs[7]:
        _duplicates(series, language)
    with tabs[8]:
        _patterns(series)
    with tabs[9]:
        _phrases(series, language)
    with tabs[10]:
        _readability(series, language)
