import numpy as np
import pandas as pd
import pytest
import requests
from typer.testing import CliRunner

from kikolens.analyze import text as tx
from kikolens.cli.main import app

runner = CliRunner()

SHIPPING = ["The delivery was late and the package arrived damaged", "Shipping took weeks and the courier was rude",
            "My parcel was delayed again and nobody answered", "Terrible delivery, the box was broken when it came"]
QUALITY = ["The battery life is excellent and the screen is great", "Great screen quality and a fast processor",
           "Love the design, the camera is amazing and perfect", "Excellent build quality, very happy with the display"]
PRICE = ["Too expensive for what it offers, poor value", "The price is high and discounts are rare",
         "Expensive compared with others, not worth the cost", "Overpriced and the refund process is a problem"]


@pytest.fixture
def reviews():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(240):
        theme = rng.choice(["shipping", "quality", "price"])
        base = {"shipping": SHIPPING, "quality": QUALITY, "price": PRICE}[theme]
        rating = {"shipping": 2, "quality": 5, "price": 3}[theme] + int(rng.integers(-1, 2)) * (rng.random() < 0.3)
        topic = {"shipping": "delivery courier parcel", "quality": "screen battery camera", "price": "price cost expensive"}[theme]
        rows.append({"review": f"{rng.choice(base)} {topic} {int(rng.integers(0, 999))} ok", "theme": theme, "rating": rating,
                     "id": f"r{len(rows)}"})
    return pd.DataFrame(rows)


def test_free_text_columns_ignores_categories_and_ids(reviews):
    assert tx.free_text_columns(reviews) == ["review"]
    assert tx.free_text_columns(reviews.head(5)) == []


def test_language_detection_and_stats():
    es = pd.Series(["el producto es muy bueno y llegó a tiempo", "no me gusta el servicio de la empresa"] * 10)
    assert tx.detect_language(es) == "spanish"
    assert tx.detect_language(pd.Series(["the product is good and it arrived on time"] * 12)) == "english"
    info = tx.text_stats(pd.Series(["hello world", "", None, "hello world"]))
    assert info["rows"] == 4 and info["empty"] == 2 and info["duplicates"] == 1 and info["mean_words"] == 2


def test_top_terms_ignore_stopwords_and_rank_by_frequency(reviews):
    terms = tx.top_terms(reviews["review"], n=10)
    assert "the" not in set(terms["term"]) and "and" not in set(terms["term"])
    assert terms["texts"].is_monotonic_decreasing and terms["share"].between(0, 1).all()
    pairs = tx.top_terms(reviews["review"], n=5, ngram=2)
    assert all(" " in t for t in pairs["term"])
    with pytest.raises(ValueError, match="ngram"):
        tx.top_terms(reviews["review"], ngram=5)
    with pytest.raises(ValueError, match="At least"):
        tx.top_terms(reviews["review"].head(4))
    with pytest.raises(ValueError, match="Language"):
        tx.top_terms(reviews["review"], language="klingon")


def test_terms_by_group_find_the_distinctive_words(reviews):
    out = tx.terms_by_group(reviews, "review", "theme", n=6)
    by = out.groupby("group")["term"].apply(set)
    assert {"delivery", "package", "shipping"} & by["shipping"]
    assert {"screen", "excellent", "great", "camera"} & by["quality"]
    assert {"expensive", "price", "refund"} & by["price"]
    assert (out["log odds"] > 0).all()
    with pytest.raises(ValueError, match="between 2 and 12"):
        tx.terms_by_group(reviews, "review", "id")
    with pytest.raises(ValueError, match="not found"):
        tx.terms_by_group(reviews, "review", "ghost")


def test_search_shows_context_and_validates(reviews):
    found = tx.search_texts(reviews["review"], "BATTERY", context=10)
    assert len(found) > 0 and found["excerpt"].str.contains("battery", case=False).all()
    assert len(tx.search_texts(reviews["review"], "the", limit=5)) == 5
    assert tx.search_texts(reviews["review"], "zzzz").empty
    with pytest.raises(ValueError, match="at least two"):
        tx.search_texts(reviews["review"], "a")


def test_lexicon_sentiment_in_both_languages():
    out = tx.lexicon_sentiment(pd.Series(["great and excellent product", "terrible and awful", "it is a table", None]), "english")
    assert list(out["tone"]) == ["positive", "negative", "no signal", "no signal"]
    es = tx.lexicon_sentiment(pd.Series(["muy bueno y genial", "horrible, el peor"]), "spanish")
    assert list(es["tone"]) == ["positive", "negative"]
    with pytest.raises(ValueError, match="Language"):
        tx.lexicon_sentiment(pd.Series(["x"]), "french")


def test_clusters_recover_the_three_themes(reviews):
    res = tx.cluster_texts(reviews["review"], k=3)
    labels = res["labels"].reindex(reviews.index)
    purity = pd.crosstab(labels, reviews["theme"]).max(axis=1).sum() / len(reviews)
    assert purity > 0.9 and res["summary"]["texts"].sum() == len(reviews)
    assert res["summary"]["key terms"].str.len().gt(0).all()
    with pytest.raises(ValueError, match="between 2 and"):
        tx.cluster_texts(reviews["review"], k=1)


def test_text_classifier_beats_the_baseline_and_explains_itself(reviews):
    res = tx.train_text_classifier(reviews, "review", "theme")
    assert res["accuracy"] > 0.95 and res["accuracy"] > res["baseline"] + 0.3
    assert set(res["drivers"]["class"]) == {"shipping", "quality", "price"}
    out = tx.classify_texts(res, pd.Series(["the screen and camera are excellent", "my parcel never arrived"]))
    assert list(out["prediction"]) == ["quality", "shipping"] and out["confidence"].between(0, 1).all()
    with pytest.raises(ValueError, match="at least two"):
        tx.train_text_classifier(reviews.assign(one="x"), "review", "one")
    with pytest.raises(ValueError, match="more than 20"):
        tx.train_text_classifier(reviews, "review", "id")
    with pytest.raises(ValueError, match="Need at least 30"):
        tx.train_text_classifier(reviews.head(20), "review", "theme")


def test_terms_associated_with_a_number(reviews):
    out = tx.association_with_numeric(reviews["review"], reviews["rating"])
    top_pos = out[out["rho"] > 0]["term"].tolist()
    top_neg = out[out["rho"] < 0]["term"].tolist()
    assert {"excellent", "great", "screen", "camera", "amazing"} & set(top_pos)
    assert {"delivery", "late", "package", "broken"} & set(top_neg)
    with pytest.raises(ValueError, match="At least 30"):
        tx.association_with_numeric(reviews["review"].head(10), reviews["rating"].head(10))


def test_llm_summary_uses_the_local_model_and_reports_failures(reviews, monkeypatch):
    captured = {}

    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "  Shipping is the main complaint.  "}

    def fake_post(url, json, timeout):
        captured.update(url=url, prompt=json["prompt"])
        return Reply()

    monkeypatch.setattr(requests, "post", fake_post)
    assert tx.summarize_with_llm(reviews["review"], n=5) == "Shipping is the main complaint."
    assert "localhost" in captured["url"] and captured["prompt"].count("\n- ") == 5

    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", down)
    with pytest.raises(ValueError, match="Is Ollama running"):
        tx.summarize_with_llm(reviews["review"])

    def slow(*a, **k):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(requests, "post", slow)
    with pytest.raises(ValueError, match="too long"):
        tx.summarize_with_llm(reviews["review"])


def test_cli_text(tmp_path, reviews):
    source = tmp_path / "r.csv"
    reviews.to_csv(source, index=False)
    out = runner.invoke(app, ["text", str(source), "review", "--group", "theme", "--themes", "3"])
    assert out.exit_code == 0 and "Most common terms" in out.output and "Themes" in out.output
    assert runner.invoke(app, ["text", str(source), "ghost"]).exit_code == 1
    assert runner.invoke(app, ["text", str(source), "review", "--ngram", "7"]).exit_code == 1
