import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kikolens.analyze import privacy as pv
from kikolens.cli.main import app
from kikolens.pipeline.recipe import Recipe


def people(n=300, seed=0):
    rng = np.random.default_rng(seed)
    letters = "TRWAGMYFPDXBNJZSQVHLCKE"
    numbers = rng.integers(10_000_000, 99_999_999, n)
    return pd.DataFrame({
        "contact": [f"user{i}@example.com" for i in range(n)],
        "tax_id": [f"{v}{letters[v % 23]}" for v in numbers],
        "card": ["4111 1111 1111 1111" if i % 2 else "5500-0000-0000-0004" for i in range(n)],
        "account": ["ES9121000418450200051332"] * n,
        "ip": [f"192.168.{i % 250}.{(i * 7) % 250}" for i in range(n)],
        "mobile": [f"+34 6{rng.integers(10_000_000, 99_999_999)}" for _ in range(n)],
        "full_name": [f"Person {i}" for i in range(n)],
        "age": rng.integers(20, 70, n), "city": rng.choice(["A", "B", "C"], n), "spend": rng.uniform(10, 500, n),
        "comment": ["great"] * n,
    })


def test_scan_finds_each_kind_by_content_and_by_name_without_false_alarms():
    found = pv.scan(people()).set_index("column")
    assert found.loc["contact", "kind"] == "email" and found.loc["tax_id", "kind"] == "Spanish ID (DNI/NIE)"
    assert found.loc["card", "kind"] == "credit card" and found.loc["account", "kind"] == "IBAN"
    assert found.loc["ip", "kind"] == "IP address" and found.loc["mobile", "kind"] == "phone number"
    assert found.loc["full_name", "reason"] == "name" and found.loc["card", "risk"] == "high"
    assert not {"age", "city", "spend", "comment"} & set(found.index)
    assert "example.com" not in " ".join(found["example"]) and found.loc["contact", "example"].count("*") > 3


def test_validators_reject_look_alikes():
    assert not pv._card("4111 1111 1111 1112") and pv._card("4111111111111111")      # falla el dígito de control (Luhn)
    assert not pv._iban("ES9121000418450200051333") and pv._iban("GB82 WEST 1234 5698 7654 32")
    assert not pv._dni("12345678A") and pv._dni("12345678Z") and pv._dni("X1234567L")
    df = pd.DataFrame({"n": [f"{4000000000000000 + i}" for i in range(100)]})        # números largos que no son tarjetas
    assert pv.scan(df).empty


def test_mask_hash_and_drop_keep_missing_and_are_stable():
    df = pd.DataFrame({"id": ["abcdef", None, "xy"], "other": [1, 2, 3]})
    masked, _ = pv.mask_column(df, "id", keep=2)
    assert masked["id"].tolist()[0] == "****ef" and pd.isna(masked["id"][1]) and masked["id"][2] == "**"
    a, _ = pv.hash_column(df, "id", salt="s")
    b, _ = pv.hash_column(df, "id", salt="s")
    c, _ = pv.hash_column(df, "id", salt="other")
    assert a["id"][0] == b["id"][0] != c["id"][0] and len(a["id"][0]) == 12 and "abcdef" not in a["id"][0] and pd.isna(a["id"][1])
    with pytest.raises(ValueError):
        pv.mask_column(df, "nope")
    with pytest.raises(ValueError):
        pv.mask_column(df, "id", keep=20)
    with pytest.raises(ValueError):
        pv.hash_column(df, "id", length=2)


def test_anonymize_steps_replay_through_a_recipe():
    df = people()
    new, steps = pv.anonymize(df, {"contact": "hash", "card": "mask", "full_name": "drop"}, salt="pepper")
    assert "full_name" not in new.columns and new["card"].str.endswith("04").sum() + new["card"].str.endswith("11").sum() == len(df)
    recipe = Recipe()
    for op, params, message in steps:
        recipe.add(op, params, message)
    again, _ = Recipe.from_json(recipe.to_json()).apply(df)
    pd.testing.assert_frame_equal(again, new)
    assert "mask_column" in recipe.to_python()
    with pytest.raises(ValueError):
        pv.anonymize(df, {"card": "shred"})


def test_k_anonymity_counts_group_sizes():
    df = pd.DataFrame({"age": [30] * 6 + [40] * 2 + [50], "city": ["A"] * 6 + ["B"] * 2 + ["C"]})
    res = pv.k_anonymity(df, ["age", "city"], k=3)
    assert res["min_k"] == 1 and res["unique_rows"] == 1 and res["groups"] == 3
    assert res["share_below_k"] == pytest.approx(3 / 9) and res["smallest_groups"]["people in group"].iloc[0] == 1
    safe = pv.k_anonymity(df.iloc[:6], ["age", "city"], k=3)
    assert safe["share_below_k"] == 0 and "safe" in safe["summary"]
    with pytest.raises(ValueError):
        pv.k_anonymity(df, [])
    with pytest.raises(ValueError):
        pv.k_anonymity(df, ["age"], k=1)
    assert set(pv.suggest_quasi_identifiers(people())) == {"age", "city"}


def test_cli_privacy(tmp_path):
    csv = tmp_path / "p.csv"
    people().to_csv(csv, index=False)
    out = CliRunner().invoke(app, ["privacy", str(csv), "--scan"])
    assert out.exit_code == 0 and "IBAN" in out.output
    res = tmp_path / "anon.csv"
    rec = tmp_path / "r.json"
    out = CliRunner().invoke(app, ["privacy", str(csv), "--hash", "contact", "--mask", "card", "--drop", "full_name",
                                   "--salt", "x", "--output", str(res), "--recipe", str(rec)])
    assert out.exit_code == 0, out.output
    cleaned = pd.read_csv(res)
    assert "full_name" not in cleaned.columns and not cleaned["contact"].str.contains("@").any() and rec.exists()
    out = CliRunner().invoke(app, ["privacy", str(csv), "--k", "5", "--quasi", "age,city"])
    assert out.exit_code == 0 and "smallest group" in out.output
    assert CliRunner().invoke(app, ["privacy", str(csv), "--hash", "nope"]).exit_code == 1
