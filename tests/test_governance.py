import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from kikolens.analyze import governance as gv
from kikolens.analyze import predict as pr
from kikolens.analyze.modellab import run_model_lab


def make_data(n=300, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n), "g": rng.choice(["a", "b"], n)})
    df["y"] = 3 * df["x"] + df["z"] + rng.normal(scale=noise, size=n)
    return df


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("KIKOLENS_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def save(name, frame, model="Ridge"):
    bundle = pr.build_bundle(run_model_lab(frame, "y", models=[model], folds=3, tune=False), frame, name=name)
    return pr.save_bundle(bundle)


@pytest.fixture
def good(home):
    return save("Good model", make_data()).stem


def documented(stem, **kw):
    return gv.register(stem, owner="Ana", description="Predicts y.", intended_use="Weekly planning.", limitations="Not for new regions.", actor="ana", **kw)


# ── Alta y ficha ─────────────────────────────────────────────────────────────
def test_registering_needs_an_owner_and_starts_as_a_draft_with_its_lineage(good):
    with pytest.raises(ValueError, match="owner"):
        gv.register(good, owner=" ")
    with pytest.raises(ValueError, match="risk"):
        gv.register(good, owner="Ana", risk="extreme")
    record = documented(good)
    assert record["stage"] == "draft" and record["target"] == "y" and record["task"] == "regression" and len(record["lineage"]["sha256"]) == 64
    assert record["lineage"]["metrics"]["R2"] > 0.9 and record["lineage"]["inputs"]["x"] == "numeric"
    with pytest.raises(ValueError, match="already registered"):
        documented(good)
    with pytest.raises(ValueError, match="not registered"):
        gv.get("nothing-here")
    gv.update_card(good, actor="ana", limitations="Not for new regions or promotions.", risk="high")
    assert gv.get(good)["risk"] == "high" and "promotions" in gv.get(good)["limitations"]
    with pytest.raises(ValueError, match="cannot be changed"):
        gv.update_card(good, stage="production")
    with pytest.raises(ValueError, match="risk"):
        gv.update_card(good, risk="meh")


# ── Puertas ──────────────────────────────────────────────────────────────────
def test_transitions_follow_the_stage_graph_and_staging_needs_documentation(good):
    gv.register(good, owner="Ana", actor="ana")
    with pytest.raises(ValueError, match="cannot go to 'production'"):
        gv.promote(good, "production", actor="ana")
    with pytest.raises(ValueError, match=r"\[documented\] Missing: a description, the intended use, the known limitations"):
        gv.promote(good, "staging", actor="ana")
    gv.update_card(good, actor="ana", description="d", intended_use="u", limitations="l")
    out = gv.promote(good, "staging", actor="ana")
    assert out["from"] == "draft" and out["to"] == "staging" and not out["overridden"] and gv.get(good)["stage"] == "staging"
    with pytest.raises(ValueError, match="stage must be"):
        gv.promote(good, "retired")


def test_production_needs_a_validation_on_new_data_that_does_not_degrade(good):
    documented(good)
    gv.promote(good, "staging", actor="ana")
    with pytest.raises(ValueError, match=r"\[validated on new data\] No passed validation"):
        gv.promote(good, "production", actor="ana")
    with pytest.raises(ValueError, match="column named 'y'"):
        gv.validate(good, make_data().drop(columns="y"), actor="bo")
    with pytest.raises(ValueError, match="At least 30 rows"):
        gv.validate(good, make_data(10, seed=1), actor="bo")
    drifted = make_data(200, seed=2)
    drifted["y"] = drifted["y"] * 4 + 10 * drifted["z"]                       # la relación cambió: el modelo ya no acierta
    bad = gv.validate(good, drifted, actor="bo")
    assert bad["passed"] is False and bad["degradation"] > 0.1 and bad["reference_metric"] == "MAE"
    with pytest.raises(ValueError, match="validated on new data"):
        gv.promote(good, "production", actor="ana")
    ok = gv.validate(good, make_data(200, seed=3), actor="bo")
    assert ok["passed"] is True and ok["actor"] == "bo" and ok["rows"] == 200 and ok["metrics"]["R2"] > 0.9
    out = gv.promote(good, "production", actor="ana")
    assert out["to"] == "production" and out["replaced"] is None
    assert all(g["passed"] for g in gv.checks(good, "production"))


def test_a_stale_validation_stops_counting(good, home):
    documented(good)
    gv.promote(good, "staging", actor="ana")
    gv.validate(good, make_data(100, seed=4), actor="bo")
    record = gv.get(good)
    record["validations"][-1]["ts"] = (datetime.now(timezone.utc) - timedelta(days=gv.VALIDATION_DAYS + 5)).isoformat(timespec="seconds")
    (home / "governance" / f"{good}.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="validated on new data"):
        gv.promote(good, "production", actor="ana")
    assert gv.overview().loc[0, "review due"] == True      # noqa: E712


def test_a_high_risk_model_needs_someone_other_than_its_owner_to_approve_that_exact_file(good, home):
    documented(good, risk="high")
    gv.promote(good, "staging", actor="ana")
    gv.validate(good, make_data(100, seed=5), actor="bo")
    with pytest.raises(ValueError, match="approved by someone else"):
        gv.promote(good, "production", actor="ana")
    with pytest.raises(ValueError, match="cannot approve their own"):
        gv.approve(good, "Ana")
    gv.approve(good, "carla", "looks right")
    with pytest.raises(ValueError, match="already approved"):
        gv.approve(good, "Carla")
    assert gv.promote(good, "production", actor="ana")["to"] == "production"
    assert gv.get(good)["approvals"][0]["comment"] == "looks right"


def test_overriding_a_failed_gate_needs_a_reason_and_is_recorded(good):
    gv.register(good, owner="Ana", actor="ana")
    with pytest.raises(ValueError, match="needs a reason"):
        gv.promote(good, "staging", actor="ana", force=True)
    out = gv.promote(good, "staging", actor="ana", force=True, reason="Demo for the board, docs to follow")
    assert out["overridden"] == ["documented"]
    log = gv.audit_log(good)
    assert "promote_override" in set(log["action"]) and log[log["action"] == "promote_override"].iloc[0]["details"]["reason"].startswith("Demo")


# ── Sustitución, comparación y vuelta atrás ──────────────────────────────────
def promote_to_production(stem, frame):
    documented(stem)
    gv.promote(stem, "staging", actor="ana")
    gv.validate(stem, frame, actor="bo")
    return gv.promote(stem, "production", actor="ana")


def test_a_new_production_model_archives_the_old_one_and_can_be_rolled_back(home):
    first = save("Model one", make_data(seed=0), "Random Forest").stem
    second = save("Model two", make_data(seed=1)).stem
    fresh = make_data(200, seed=9)
    promote_to_production(first, fresh)
    documented(second)
    gv.promote(second, "staging", actor="ana")
    gv.validate(second, fresh, actor="bo")
    with pytest.raises(ValueError, match="real results so both can be compared"):
        gv.promote(second, "production", actor="ana")
    out = gv.promote(second, "production", actor="ana", data=fresh)
    assert out["replaced"] == first and gv.get(first)["stage"] == "archived" and gv.get(second)["stage"] == "production"
    stages = gv.overview().set_index("model")["stage"].to_dict()
    assert stages == {first: "archived", second: "production"} and list(gv.overview()["stage"])[0] == "production"
    back = gv.rollback("y", actor="ana", reason="second model misbehaves")
    assert back == {"restored": first, "demoted": second} and gv.get(first)["stage"] == "production" and gv.get(second)["stage"] == "staging"
    assert "rollback" in set(gv.audit_log()["action"])
    with pytest.raises(ValueError, match="no earlier production"):
        gv.rollback("other-target")


def test_a_clearly_worse_challenger_is_blocked(home):
    champion = save("Champion", make_data(seed=0)).stem
    weak = make_data(seed=1)
    weak["y"] = np.random.default_rng(3).permutation(weak["y"].to_numpy())             # un modelo entrenado con el resultado desordenado: no sabe nada
    challenger = save("Challenger", weak).stem
    fresh = make_data(300, seed=11)
    promote_to_production(champion, fresh)
    documented(challenger)
    gv.promote(challenger, "staging", actor="ana")
    with pytest.raises(ValueError, match="validated on new data"):
        gv.promote(challenger, "production", actor="ana", data=fresh)               # primero tendría que validarse
    gv.validate(challenger, fresh, actor="bo", max_drop=1.0)
    gates = {g["name"]: g for g in gv.checks(challenger, "production", fresh)}
    assert not gates["not worse than production"]["passed"] and "comes first" in gates["not worse than production"]["detail"]
    with pytest.raises(ValueError, match="not worse than production"):
        gv.promote(challenger, "production", actor="ana", data=fresh)
    assert gv.get(champion)["stage"] == "production"


# ── Integridad y auditoría ───────────────────────────────────────────────────
def test_a_model_file_changed_after_registration_cannot_be_promoted_or_served(good, home):
    documented(good)
    assert gv.integrity(good)["ok"]
    gv.promote(good, "staging", actor="ana")
    path = home / "models" / f"{good}.klmodel"
    path.write_bytes(path.read_bytes() + b"tampered")
    assert not gv.integrity(good)["ok"] and not gv.overview().loc[0, "file unchanged"]
    with pytest.raises(ValueError, match="file unchanged"):
        gv.promote(good, "production", actor="ana", force=False)
    with pytest.raises(ValueError, match="changed after it was registered"):
        gv.resolve_reference("y@staging")
    assert "CHANGED" in gv.model_card(good)


def test_the_audit_log_is_a_hash_chain_that_notices_edits_and_removals(good, home):
    documented(good)
    gv.promote(good, "staging", actor="ana")
    gv.validate(good, make_data(100, seed=6), actor="bo")
    assert gv.verify_audit() == {"ok": True, "entries": 3, "broken_at": None, "message": "3 entries, the chain is intact."}
    log = gv.audit_log()
    assert list(log["action"]) == ["register", "promote", "validate"] and list(log["actor"]) == ["ana", "ana", "bo"] and log["prev"].iloc[0] == ""
    path = home / "governance" / "audit.jsonl"
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    edited = json.loads(lines[1])
    edited["actor"] = "someone-else"
    path.write_text("\n".join([lines[0], json.dumps(edited), lines[2]]) + "\n", encoding="utf-8")
    assert gv.verify_audit()["broken_at"] == 2
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    assert not gv.verify_audit()["ok"] and gv.verify_audit()["broken_at"] == 2
    path.write_text(original + "not json\n", encoding="utf-8")
    assert gv.verify_audit()["broken_at"] == 4
    path.write_text(original, encoding="utf-8")
    assert gv.verify_audit()["ok"]


# ── Referencias y ficha ──────────────────────────────────────────────────────
def test_models_can_be_referred_to_by_the_stage_they_are_in(good):
    documented(good)
    gv.promote(good, "staging", actor="ana")
    assert pr.resolve_model("y@staging").stem == good and pr.resolve_model(f"{good}@staging").stem == good
    with pytest.raises(ValueError, match="No model called or predicting"):
        pr.resolve_model("y@production")
    with pytest.raises(ValueError, match="No model called"):
        pr.resolve_model("who@knows")
    gv.validate(good, make_data(100, seed=7), actor="bo")
    gv.promote(good, "production", actor="ana")
    assert pr.resolve_model("y@production").stem == good


def test_the_model_card_collects_everything(good):
    documented(good, risk="high")
    gv.promote(good, "staging", actor="ana")
    gv.validate(good, make_data(100, seed=8), actor="bo")
    gv.approve(good, "carla", "ok to go")
    card = gv.model_card(good)
    for text in ("# Model card: Good model", "**staging**", "Owner | Ana", "Risk | high", "Weekly planning.", "Not for new regions.", "R2 ", "passed", "carla", "ok to go",
                 "-> **staging** by ana", "unchanged"):
        assert text in card, text
    bare = pr.save_bundle(pr.build_bundle(run_model_lab(make_data(), "y", models=["Ridge"], folds=3, tune=False), make_data(), name="Bare")).stem
    gv.register(bare, owner="Dev")
    assert "_Not written yet._" in gv.model_card(bare) and "_No validation yet._" in gv.model_card(bare)


def test_a_classifier_is_validated_on_accuracy(home):
    df = make_data()
    df["c"] = np.where(df["y"] > 0, "hi", "lo")
    frame = df.drop(columns="y")
    bundle = pr.build_bundle(run_model_lab(frame, "c", models=["Logistic Regression"], folds=3, tune=False), frame, name="Clf")
    stem = pr.save_bundle(bundle).stem
    documented(stem)
    fresh = make_data(200, seed=12)
    fresh["c"] = np.where(fresh["y"] > 0, "hi", "lo")
    result = gv.validate(stem, fresh.drop(columns="y"), actor="bo")
    assert result["reference_metric"] == "accuracy" and result["passed"] is True


# ── Terminal, API y vista ────────────────────────────────────────────────────
def test_python_api_and_terminal(good, home):
    import kikolens as kl
    from typer.testing import CliRunner

    from kikolens.cli.main import app
    runner = CliRunner()
    out = runner.invoke(app, ["governance", "register", good, "--owner", "Ana", "--description", "d", "--intended-use", "u", "--limitations", "l", "--risk", "low", "--actor", "ana"])
    assert out.exit_code == 0 and "draft" in out.output
    assert runner.invoke(app, ["governance", "register", good, "--owner", "Ana"]).exit_code == 1
    assert runner.invoke(app, ["governance", "promote", good, "production", "--actor", "ana"]).exit_code == 1
    assert runner.invoke(app, ["governance", "promote", good, "staging", "--actor", "ana"]).exit_code == 0
    fresh = home / "fresh.csv"
    make_data(150, seed=13).to_csv(fresh, index=False)
    out = runner.invoke(app, ["governance", "validate", good, str(fresh), "--actor", "bo"])
    assert out.exit_code == 0 and "Passed on 150 rows" in out.output
    out = runner.invoke(app, ["governance", "checks", good, "production"])
    assert out.exit_code == 0 and "validated on new data" in out.output
    assert runner.invoke(app, ["governance", "promote", good, "production", "--actor", "ana"]).exit_code == 0
    out = runner.invoke(app, ["governance"])
    assert out.exit_code == 0 and "Model governance" in out.output
    card = home / "card.md"
    assert runner.invoke(app, ["governance", "card", good, "--output", str(card)]).exit_code == 0 and "# Model card" in card.read_text(encoding="utf-8")
    assert "intact" in runner.invoke(app, ["governance", "audit"]).output
    assert runner.invoke(app, ["governance", "rollback", "y"]).exit_code == 1
    assert kl.governance()["stage"].tolist() == ["production"]
    assert kl.model_card(good).startswith("# Model card")


def test_the_governance_page_walks_through_a_promotion(good):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from kikolens.dashboard import app as dashboard_app
    df = make_data(150, seed=14)
    at = AppTest.from_file(str(Path(dashboard_app.__file__)), default_timeout=60)
    at.session_state["df"] = df
    at.session_state["original_df"] = df.copy()
    at.session_state["filename"] = "s.csv"
    at.session_state["nav"] = "Model governance"
    at.run()
    assert not at.exception
    at.selectbox(key="mg_model").set_value(good).run()
    at.text_input(key="mg_owner").set_value("Ana").run()
    at.text_area(key="mg_description").set_value("Predicts y").run()
    at.text_input(key="mg_intended").set_value("Planning").run()
    at.text_input(key="mg_limits").set_value("New regions").run()
    at.text_input(key="mg_actor").set_value("ana").run()
    at.button(key="mg_register").click().run()
    assert not at.exception and gv.get(good)["owner"] == "Ana"
    at.button(key="mg_validate").click().run()
    assert not at.exception and gv.get(good)["validations"] and gv.get(good)["validations"][-1]["passed"]
    at.selectbox(key="mg_to").set_value("staging").run()
    at.button(key="mg_promote").click().run()
    assert not at.exception
    assert gv.get(good)["stage"] == "staging" and gv.get(good)["stage_history"][-1]["actor"] == "ana"
