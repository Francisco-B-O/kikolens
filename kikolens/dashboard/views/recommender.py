import pandas as pd
import streamlit as st

from kikolens.analyze import recommend as rc
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Recommendations"
SUBTITLE = "Who might like what, and which items go together, from a table of who used or rated what."


def render(df: pd.DataFrame) -> None:
    a, b, c = st.columns(3)
    user = a.selectbox("Person", df.columns, key="rec_u")
    item = b.selectbox("Item", [x for x in df.columns if x != user], key="rec_i")
    rating = c.selectbox("Rating (optional)", ["(every interaction counts the same)"] + list(df.select_dtypes("number").columns), key="rec_r")
    rate = None if rating.startswith("(") else rating
    try:
        model = rc.fit(df, user, item, rate)
    except ValueError as e:
        ui.note(str(e), "low")
        return
    ui.stats([("People", f"{len(model.users):,}"), ("Items", f"{len(model.items):,}"), ("Interactions", f"{model.matrix.nnz:,}")])
    tabs = st.tabs(["For a person", "Like this item", "How good is it?"])
    with tabs[0]:
        person = st.selectbox("Person", model.users, key="rec_p")
        n = st.slider("How many", 3, 30, 10, key="rec_n")
        seen = df[df[user].astype(str) == person][item].astype(str).unique().tolist()
        st.caption(f"{t('Already used')}: {', '.join(seen[:12])}{'...' if len(seen) > 12 else ''}")
        ui.table(model.recommend(person, n).round(4), hide_index=True)
    with tabs[1]:
        chosen = st.selectbox("Item", model.items, key="rec_it")
        ui.table(model.similar_items(chosen, 10).round(4), hide_index=True)
    with tabs[2]:
        st.caption(t("One interaction of each person is hidden, and the recommender has to bring it back among its first suggestions."))
        if st.button("Evaluate", key="rec_ev", type="primary"):
            try:
                st.session_state.rec_eval = rc.evaluate(df, user, item, rate)
            except ValueError as e:
                st.session_state.pop("rec_eval", None)
                ui.note(str(e), "high")
        res = st.session_state.get("rec_eval")
        if res:
            ui.note(res["summary"], "accent")
            ui.stats([("Recommender", f"{res['hit_rate']:.1%}"), ("Most popular items", f"{res['popular_hit_rate']:.1%}"), ("People tested", f"{res['people']:,}")])
