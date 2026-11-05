import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import textmining as tm

THEMES = {
    "ship": "delivery shipping package arrived late courier tracking box parcel delayed warehouse".split(),
    "quality": "quality broke broken material durable sturdy defect scratched fabric stitching zipper".split(),
    "price": "price expensive value money discount worth cost overpriced refund bargain coupon".split(),
}
COMMON = "the was is very and but this it my i really product item order".split()


def corpus(n=300, seed=0):
    rng = np.random.default_rng(seed)
    docs, labels = [], []
    for i in range(n):
        theme = list(THEMES)[i % 3]
        words = list(rng.choice(THEMES[theme], 6)) + list(rng.choice(COMMON, 5))
        rng.shuffle(words)
        docs.append(" ".join(words) + ".")
        labels.append(theme)
    return pd.Series(docs), np.array(labels)


def purity(labels, assigned):
    return pd.crosstab(labels, assigned).max(axis=1).sum() / len(labels)


# ── Temas ────────────────────────────────────────────────────────────────────
def test_topics_recover_the_three_themes_and_pick_three_by_coherence():
    texts, labels = corpus()
    res = tm.topics(texts)
    assert res["k"] == 3 and res["chosen_automatically"] and purity(labels, res["assignments"]["topic"]) > 0.95
    assert res["table"]["share"].round(2).tolist() == [0.33, 0.33, 0.33] and len(res["coherence"]) == 7 and not res["notes"]
    leading = " ".join(res["table"]["terms"])
    for theme in THEMES.values():
        assert any(word in leading for word in theme)
    assert all(len(ex) == 3 for ex in res["table"]["examples"]) and res["doc_topic"].shape == (300, 3) and len(res["figure"].data) == 1


def test_lda_also_works_and_a_fixed_number_is_respected():
    texts, labels = corpus()
    res = tm.topics(texts, 3, method="lda")
    assert res["k"] == 3 and not res["chosen_automatically"] and purity(labels, res["assignments"]["topic"]) > 0.9
    five = tm.topics(texts, 5)
    assert five["k"] == 5 and len(five["table"]) == 5 and abs(five["table"]["share"].sum() - 1) < 1e-9


def test_topics_validate_their_inputs():
    texts, _ = corpus()
    for kwargs, fragment in (({"k": 1}, "between 2"), ({"k": 99}, "between 2"), ({"max_k": 1}, "max_k"), ({"method": "x"}, "method")):
        with pytest.raises(ValueError, match=fragment):
            tm.topics(texts, **kwargs)
    with pytest.raises(ValueError, match="At least"):
        tm.topics(texts.head(3))
    with pytest.raises(ValueError, match="too short|usable terms|No term"):
        tm.topics(pd.Series(["a b", "c d"] * 20))


def test_topics_over_time_follow_a_theme_that_grows():
    rng = np.random.default_rng(1)
    rows = []
    for month in range(12):
        share_ship = 0.1 + 0.07 * month
        for _ in range(40):
            theme = "ship" if rng.random() < share_ship else rng.choice(["quality", "price"])
            words = list(rng.choice(THEMES[theme], 6)) + list(rng.choice(COMMON, 5))
            rows.append({"text": " ".join(words), "date": pd.Timestamp("2024-01-15") + pd.DateOffset(months=month)})
    res = tm.topics_over_time(pd.DataFrame(rows), "text", "date", k=3)
    shipping = next(c for c in res["over_time"].columns if any(w in c for w in THEMES["ship"]))
    assert res["over_time"][shipping].iloc[-1] > res["over_time"][shipping].iloc[0] + 0.3 and len(res["over_time"]) == 12
    assert "changed most" in res["summary"] and len(res["figure"].data) == 3
    with pytest.raises(ValueError, match="not found"):
        tm.topics_over_time(pd.DataFrame(rows), "text", "nope")


# ── Duplicados y parecidos ───────────────────────────────────────────────────
def test_near_duplicates_are_grouped_and_exact_ones_are_labelled():
    texts = pd.Series(["Please refund my order, the item arrived broken and late", "please refund my order. The item arrived broken and late!",
                       "Please refund my order, the item arrived brokenn and late", "Great service, will buy again soon", "Great service, will buy again soon",
                       "Completely different sentence about a blue umbrella in the rain", "Another unrelated remark on shipping boxes and tape",
                       "Yet another distinct comment regarding warranty claims", "Short unique note number one", "Short unique note number two is longer",
                       "The zipper stuck after two weeks of normal daily use"])
    res = tm.find_duplicates(texts, 0.85)
    sizes = sorted(res["table"]["size"])
    assert sizes == [2, 3] and res["redundant"] == 3
    exact = res["table"][res["table"]["exact"]].iloc[0]
    assert sorted(exact["rows"]) == [3, 4] and res["notes"] and "near-identical" in res["summary"]
    loose = tm.find_duplicates(texts, 1.0)
    assert loose["groups"] == 1                                         # a 1.0 solo quedan los idénticos
    assert tm.find_duplicates(corpus()[0].iloc[:12], 0.99)["groups"] == 0
    with pytest.raises(ValueError, match="threshold"):
        tm.find_duplicates(texts, 0.1)


def test_similar_texts_finds_the_closest_by_query_or_by_row():
    texts, labels = corpus()
    by_text = tm.similar_texts(texts, "package delayed at the warehouse with the courier", n=5)
    assert (labels[by_text["row"].to_numpy()] == "ship").all() and by_text["similarity"].is_monotonic_decreasing
    by_row = tm.similar_texts(texts, 1, n=5)
    assert 1 not in set(by_row["row"]) and (labels[by_row["row"].to_numpy()] == labels[1]).mean() >= 0.8
    with pytest.raises(ValueError, match="None of the words"):
        tm.similar_texts(texts, "zzzz qqqq")
    with pytest.raises(ValueError, match="must be a text"):
        tm.similar_texts(texts, 99999)


# ── Patrones ─────────────────────────────────────────────────────────────────
def test_known_patterns_are_found_and_personal_data_is_flagged():
    texts = pd.Series(["Write to ana.perez+shop@example.co.uk or call +34 612 345 678 before 2024-05-03.", "Paid $1,250.50 (15% off) at https://shop.example.com/offer?id=3 #sale @ana",
                       "Meeting on 03/05/2024, total 300 EUR, ask @bo", "Nothing to see here", None, "   "])
    res = tm.extract_patterns(texts)
    found = res["found"]
    values = lambda kind: set(found.loc[found["kind"] == kind, "value"])         # noqa: E731
    assert values("email") == {"ana.perez+shop@example.co.uk"} and values("phone") == {"+34 612 345 678"}
    assert values("url") == {"https://shop.example.com/offer?id=3"} and values("hashtag") == {"#sale"} and values("mention") == {"@ana", "@bo"}
    assert values("money") == {"$1,250.50", "300 EUR"} and values("percent") == {"15%"} and values("date") == {"2024-05-03", "03/05/2024"}
    assert res["texts"] == 4 and any("Privacy" in n for n in res["notes"]) and res["table"].set_index("kind").loc["email", "texts"] == 1
    only = tm.extract_patterns(texts, ["hashtag"])
    assert list(only["table"]["kind"]) == ["hashtag"] and not only["notes"]
    with pytest.raises(ValueError, match="Unknown pattern"):
        tm.extract_patterns(texts, ["ssn"])
    with pytest.raises(ValueError, match="no texts"):
        tm.extract_patterns(pd.Series(["", None]))


# ── Frases clave, colocaciones y legibilidad ─────────────────────────────────
def test_keyphrases_and_collocations_surface_the_repeated_expressions():
    rng = np.random.default_rng(2)
    fillers = ["battery life", "screen", "price", "keyboard", "camera"]
    docs = [f"The battery life is great and the {rng.choice(fillers)} works. Customer service was slow, customer service never answered." for _ in range(60)]
    key = tm.keyphrases(pd.Series(docs), 6)
    assert "battery life" in set(key["phrase"]) and "customer service" in set(key["phrase"])
    col = tm.collocations(pd.Series(docs), 5, min_count=10)
    assert col.iloc[0]["words"] in ("customer service", "battery life") and col["npmi"].is_monotonic_decreasing
    with pytest.raises(ValueError, match="min_count"):
        tm.collocations(pd.Series(docs), min_count=1)
    with pytest.raises(ValueError, match="at least 500 times"):
        tm.collocations(pd.Series(docs), min_count=500)
    def word(i):                                                  # una palabra distinta por texto, solo letras
        return "".join(chr(97 + (i // 26 ** k) % 26) for k in range(4))
    with pytest.raises(ValueError, match="at least two texts"):
        tm.keyphrases(pd.Series([f"{word(2 * i)} {word(2 * i + 1)}." for i in range(30)]))


def test_readability_orders_simple_and_dense_texts_in_both_languages():
    simple = ["The cat sat on the mat. It was a sunny day. We all went out to play."] * 12
    dense = ["Notwithstanding considerable epistemological disagreements, interdisciplinary investigators systematically operationalised heterogeneous methodological "
             "frameworks concerning institutional characteristics."] * 12
    easy, hard = tm.readability(pd.Series(simple)), tm.readability(pd.Series(dense))
    assert easy["median"] > 80 > 40 > hard["median"] and easy["label"] == "easy" and hard["label"] in ("very difficult", "difficult")
    spanish = ["El gato se sentó en la alfombra. Era un día de sol. Todos salimos a jugar al parque."] * 12
    res = tm.readability(pd.Series(spanish))
    assert res["language"] == "spanish" and res["median"] > 70 and "Fernández-Huerta" in res["summary"] and res["notes"]
    assert int(res["distribution"].sum()) == 12


def test_python_api_and_terminal(tmp_path):
    import kikolens as kl
    from typer.testing import CliRunner

    from kikolens.cli.main import app
    texts, _ = corpus(120)
    df = pd.DataFrame({"review": texts, "when": pd.date_range("2024-01-01", periods=120, freq="D")})
    assert kl.topics(df, "review", k=3)["k"] == 3 and kl.topics(df, "review", date="when", k=3)["over_time"].shape[1] == 3
    assert kl.near_duplicates(df, "review")["kind"] == "near duplicates" and kl.similar(df, "review", "late package", n=3)["table"].shape[0] == 3
    assert kl.text_patterns(df, "review")["kind"] == "text patterns" and kl.keyphrases(df, "review", n=5)["table"].shape[0] <= 5
    assert kl.readability(df, "review")["kind"] == "readability"
    csv = tmp_path / "r.csv"
    df.to_csv(csv, index=False)
    runner = CliRunner()
    for args, text in ((["topics", str(csv), "review", "--topics", "3"], "3 topics"), (["near-duplicates", str(csv), "review"], "near-duplicate"),
                       (["similar", str(csv), "review", "late package", "--n", "3"], "similarity"), (["patterns", str(csv), "review"], "No known patterns"),
                       (["keyphrases", str(csv), "review"], "phrase"), (["readability", str(csv), "review"], "reading ease")):
        out = runner.invoke(app, args)
        assert out.exit_code == 0 and text in out.output.replace("\n", " "), (args, out.output)
    assert runner.invoke(app, ["topics", str(csv), "nope"]).exit_code == 1
