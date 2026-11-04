import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kikolens.analyze import text as tx
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
    tabs = st.tabs(["Terms", "By group", "Themes", "Classify", "Tone", "Summary"])
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
