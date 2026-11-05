import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import attribution as at

T0 = pd.Timestamp("2024-01-01")


def touches(paths, prefix="u"):
    """paths: lista de (canales, ¿convierte?, repeticiones). Un contacto cada dos días."""
    rows = []
    n = 0
    for channels, converted, times in paths:
        for _ in range(times):
            for k, ch in enumerate(channels):
                rows.append({"user": f"{prefix}{n}", "time": T0 + pd.Timedelta(days=2 * k), "channel": ch, "converted": bool(converted and k == len(channels) - 1)})
            n += 1
    return pd.DataFrame(rows)


# ── El ejemplo resuelto a mano ───────────────────────────────────────────────
def worked_example():
    return touches([(["A", "B"], True, 100), (["B"], False, 100)])


def test_every_model_matches_the_hand_computed_shares():
    res = at.attribute(worked_example(), "user", "time", "channel", "converted")
    credit = res["credit"]
    assert credit.loc["A", "first touch"] == 100 and credit.loc["B", "last touch"] == 100 and credit.loc["B", "first touch"] == 0
    assert credit.loc["A", "linear"] == pytest.approx(50) and credit.loc["A", "position based"] == pytest.approx(50)
    # Markov: P(convertir) = 0,5; sin A baja a 0,25 (efecto 0,5); sin B baja a 0 (efecto 1) -> reparto 1/3 y 2/3
    assert res["markov_conversion_probability"] == pytest.approx(0.5)
    removal = res["table"].set_index("channel")["removal effect"]
    assert removal["A"] == pytest.approx(0.5) and removal["B"] == pytest.approx(1.0)
    assert credit.loc["A", "markov"] == pytest.approx(100 / 3) and credit.loc["B", "markov"] == pytest.approx(200 / 3)
    assert credit.loc["A", "shapley"] == pytest.approx(50) and credit.loc["B", "shapley"] == pytest.approx(50)
    assert res["conversions"] == 100 and res["journeys_total"] == 200 and res["conversion_rate"] == pytest.approx(0.5)


def test_each_model_shares_out_exactly_the_conversions_it_was_given():
    rng = np.random.default_rng(0)
    channels = list("ABCDE")
    rows = []
    for i in range(600):
        k = int(rng.integers(1, 6))
        path = rng.choice(channels, k)
        convert = rng.random() < 0.15 + 0.1 * ("C" in path)
        for j, ch in enumerate(path):
            rows.append({"user": i, "time": T0 + pd.Timedelta(days=int(j) * 3 + int(rng.integers(0, 2))), "channel": ch, "converted": convert and j == len(path) - 1, "revenue": float(rng.integers(20, 90))})
    d = pd.DataFrame(rows)
    plain = at.attribute(d, "user", "time", "channel", "converted")
    assert (plain["credit"].sum() - plain["conversions"]).abs().max() < 1e-6
    worth = at.attribute(d, "user", "time", "channel", "converted", value="revenue")
    assert (worth["credit"].sum() - worth["total_value"]).abs().max() < 1e-6 and worth["total_value"] > worth["conversions"] * 20
    assert (plain["shares"].sum() - 1).abs().max() < 1e-9 and set(plain["credit"].columns) == set(at.MODELS)


def test_markov_and_shapley_credit_the_assist_that_last_touch_ignores():
    rng = np.random.default_rng(1)
    rows = []
    n = 0
    spec = [(["display", "search"], 0.5, 600), (["search"], 0.08, 600), (["display"], 0.01, 600), (["social", "search"], 0.12, 300)]
    for channels, p, times in spec:
        for _ in range(times):
            convert = rng.random() < p
            for k, ch in enumerate(channels):
                rows.append({"user": n, "time": T0 + pd.Timedelta(days=k), "channel": ch, "converted": convert and k == len(channels) - 1})
            n += 1
    res = at.attribute(pd.DataFrame(rows), "user", "time", "channel", "converted")
    shares = res["shares"]
    assert shares.loc["display", "last touch"] < 0.05 and shares.loc["display", "markov"] > 0.2 and shares.loc["display", "shapley"] > 0.3
    assert shares.loc["search", "last touch"] > 0.9 and shares.loc["search", "markov"] < shares.loc["search", "last touch"] - 0.2
    assert any("'search' gets less credit" in n for n in res["notes"])                 # la mayor discrepancia es el canal que cierra: último contacto le da casi todo


# ── Reglas ───────────────────────────────────────────────────────────────────
def test_time_decay_and_position_rules():
    d = pd.DataFrame({"user": ["x"] * 4 + ["y"] * 4, "time": [T0, T0 + pd.Timedelta(days=1), T0 + pd.Timedelta(days=2), T0 + pd.Timedelta(days=30)] * 2,
                      "channel": ["a", "b", "c", "d"] * 2, "converted": [False, False, False, True] * 2})
    filler = touches([(["a", "d"], True, 20), (["b"], False, 20)], prefix="f")
    res = at.attribute(pd.concat([d, filler], ignore_index=True), "user", "time", "channel", "converted", half_life_days=3)
    decay, position = res["credit"]["time decay"], res["credit"]["position based"]
    assert decay["d"] > 100 * decay["c"] and decay["c"] < 0.05                    # a 30 días del final, con vida media de 3, un contacto casi no cuenta
    assert position["b"] == pytest.approx(0.2) and position["c"] == pytest.approx(0.2)         # 4 contactos: 40 % primero, 40 % último, 10 % cada uno de los de en medio, en 2 recorridos
    assert position.sum() == pytest.approx(res["conversions"])
    slow = at.attribute(pd.concat([d, filler], ignore_index=True), "user", "time", "channel", "converted", half_life_days=100)
    assert slow["credit"]["time decay"]["c"] > res["credit"]["time decay"]["c"]


def test_touches_after_the_conversion_do_not_count_and_value_comes_from_the_conversion_row():
    d = pd.DataFrame({"user": ["x"] * 3, "time": [T0, T0 + pd.Timedelta(days=1), T0 + pd.Timedelta(days=5)], "channel": ["a", "b", "late"], "converted": [False, True, False], "amount": [0, 70, 0]})
    j = at.build_journeys(pd.concat([d, touches([(["a", "b"], True, 20), (["a"], False, 20)], prefix="p").assign(amount=0.0)], ignore_index=True), "user", "time", "channel", "converted", "amount")
    row = j[j["user"] == "x"].iloc[0]
    assert row["channels"] == ("a", "b") and row["converted"] and row["value"] == 70


# ── Estadísticas del recorrido ───────────────────────────────────────────────
def test_journey_statistics_and_the_top_paths():
    res = at.attribute(worked_example(), "user", "time", "channel", "converted")
    assert res["median_days_to_convert"] == pytest.approx(2.0) and res["mean_touches"] == pytest.approx(2.0) and res["top_paths"].iloc[0] == 100 and res["top_paths"].index[0] == "A > B"
    assert "Last touch gives most credit to 'B'" in res["summary"] and len(res["figure"].data) == 4 and res["notes"][-1].startswith("Attribution splits")


def test_single_touch_journeys_make_the_models_agree_and_the_page_says_so():
    d = touches([(["A"], True, 60), (["A"], False, 140), (["B"], True, 30), (["B"], False, 170)])
    res = at.attribute(d, "user", "time", "channel", "converted")
    shares = res["shares"]
    assert (shares["last touch"] - shares["markov"]).abs().max() < 0.02 and any("single contact" in n for n in res["notes"])


def test_many_channels_are_grouped_as_other():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(500):
        for k, ch in enumerate(rng.choice([f"c{j}" for j in range(14)], int(rng.integers(1, 4)))):
            rows.append({"user": i, "time": T0 + pd.Timedelta(days=k), "channel": ch, "converted": False})
        rows[-1]["converted"] = bool(rng.random() < 0.2)
    res = at.attribute(pd.DataFrame(rows), "user", "time", "channel", "converted")
    assert len(res["channels"]) == at.MAX_SHAPLEY_CHANNELS and "other" in res["channels"] and (res["credit"].sum() - res["conversions"]).abs().max() < 1e-6


def test_input_problems_are_explained():
    ok = worked_example()
    for kwargs, fragment in (({"user": "nope"}, "not found"), ({"half_life_days": 0}, "positive")):
        args = {"user": "user", "time": "time", "channel": "channel", "converted": "converted", **kwargs}
        with pytest.raises(ValueError, match=fragment):
            at.attribute(ok, **args)
    with pytest.raises(ValueError, match="Nobody converted"):
        at.attribute(ok.assign(converted=False), "user", "time", "channel", "converted")
    with pytest.raises(ValueError, match="At least 30"):
        at.attribute(ok.head(20), "user", "time", "channel", "converted")
    with pytest.raises(ValueError, match="At least two channels"):
        at.attribute(ok.assign(channel="A"), "user", "time", "channel", "converted")
    with pytest.raises(ValueError, match="numeric"):
        at.attribute(ok.assign(v="x"), "user", "time", "channel", "converted", value="v")
    text = ok.assign(converted=ok["converted"].map({True: "yes", False: "no"}))
    assert at.attribute(text, "user", "time", "channel", "converted")["conversions"] == 100


def test_python_api_and_terminal(tmp_path):
    import kikolens as kl
    from typer.testing import CliRunner

    from kikolens.cli.main import app
    d = worked_example()
    res = kl.attribution(d, "user", "time", "channel", "converted")
    assert res["kind"] == "attribution" and res["credit"].loc["A", "markov"] == pytest.approx(100 / 3)
    csv = tmp_path / "t.csv"
    d.to_csv(csv, index=False)
    runner = CliRunner()
    out = runner.invoke(app, ["attribution", str(csv), "user", "time", "channel", "converted", "--output", str(tmp_path / "credit.csv")])
    assert out.exit_code == 0 and "conversions out of" in " ".join(out.output.split()) and (tmp_path / "credit.csv").exists()
    assert runner.invoke(app, ["attribution", str(csv), "user", "time", "channel", "nope"]).exit_code == 1
