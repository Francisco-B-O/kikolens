import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import recommend as rc
from kikolens.cli.main import app


def shop(users=300, seed=0):
    """Tres gustos: cada persona compra sobre todo artículos de su grupo (A, B o C) y algo de lo demás."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(users):
        taste = u % 3
        for item in rng.choice(30, 8, replace=False):
            own = item // 10 == taste
            if own or rng.random() < 0.12:
                rows.append((f"u{u}", f"item{item:02d}", int(rng.integers(3, 6) if own else rng.integers(1, 3))))
    return pd.DataFrame(rows, columns=["user", "item", "rating"])


def test_recommendations_follow_the_users_taste_and_skip_what_they_have():
    df = shop()
    model = rc.fit(df, "user", "item")
    seen = set(df[df["user"] == "u0"]["item"])
    recs = model.recommend("u0", 5)
    assert not set(recs["item"]) & seen and len(recs) == 5
    group_of = lambda name: int(name[4:]) // 10
    assert np.mean([group_of(i) == 0 for i in recs["item"]]) >= 0.8        # u0 es del gusto 0
    assert (np.diff(recs["score"]) <= 1e-12).all()
    assert len(model.recommend("u0", 5, exclude_seen=False)) == 5


def test_similar_items_are_from_the_same_group_and_show_the_overlap():
    model = rc.fit(shop(), "user", "item")
    sim = model.similar_items("item03", 5)
    assert all(int(i[4:]) // 10 == 0 for i in sim["item"]) and (sim["people in common"] > 0).all()
    with pytest.raises(ValueError, match="not one of"):
        model.similar_items("nope")


def test_unknown_users_get_the_popular_items_and_ratings_change_the_scores():
    df = shop()
    model = rc.fit(df, "user", "item")
    pop = model.recommend("stranger", 3)
    assert pop["item"].tolist() == df["item"].value_counts().head(3).index.tolist() or len(pop) == 3
    weighted = rc.fit(df, "user", "item", "rating")
    assert not np.allclose(weighted.similarity, model.similarity)


def test_honest_evaluation_beats_popularity():
    res = rc.evaluate(shop(500), "user", "item", k=5)
    assert res["hit_rate"] > res["popular_hit_rate"] * 1.5 and res["people"] > 100 and "times better" in res["summary"]


def test_validation():
    df = shop()
    with pytest.raises(ValueError, match="not found"):
        rc.fit(df, "user", "nope")
    with pytest.raises(ValueError, match="At least 100"):
        rc.fit(df.head(50), "user", "item")
    with pytest.raises(ValueError, match="numeric"):
        rc.fit(df.assign(rating="x"), "user", "item", "rating")
    with pytest.raises(ValueError, match="10 people"):
        rc.fit(pd.DataFrame({"user": ["a"] * 120, "item": [f"i{k}" for k in range(120)]}), "user", "item")


def test_cli(tmp_path):
    csv = tmp_path / "s.csv"
    shop().to_csv(csv, index=False)
    out = CliRunner().invoke(app, ["recommend", str(csv), "user", "item", "--for", "u0", "--top", "3"])
    assert out.exit_code == 0 and "item" in out.output
    out = CliRunner().invoke(app, ["recommend", str(csv), "user", "item", "--similar-to", "item03"])
    assert out.exit_code == 0
    out = CliRunner().invoke(app, ["recommend", str(csv), "user", "item", "--evaluate"])
    assert out.exit_code == 0 and "times better" in " ".join(out.output.split())
    assert CliRunner().invoke(app, ["recommend", str(csv), "user", "nope", "--for", "u0"]).exit_code == 1
