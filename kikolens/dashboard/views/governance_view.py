import pandas as pd
import streamlit as st

from kikolens.analyze import governance as gv
from kikolens.analyze.predict import list_models
from kikolens.dashboard import components as ui
from kikolens.dashboard.i18n import t

TITLE = "Model governance"
SUBTITLE = "Who owns each saved model, what stage it is in, what was checked before it went live, and a log nobody can quietly edit."

STAGE_NOTE = {"draft": "accent", "staging": "medium", "production": "ok", "archived": "low"}


def _actor() -> str:
    return st.text_input("Who is doing this", key="mg_actor", help="Recorded in the audit log. It is a name, not a login: this is a record for the team, not access control.") or ""


def _overview() -> None:
    table = gv.overview()
    if table.empty:
        ui.note("No models are registered yet. Open the Model tab, choose a saved model and register it with an owner.", "low")
    else:
        due = int(table["review due"].sum())
        ui.stats([("Registered", f"{len(table)}"), ("In production", f"{int((table['stage'] == 'production').sum())}"), ("Review due", f"{due}"),
                  ("File changed", f"{int((~table['file unchanged']).sum())}")])
        ui.table(table)
        if due:
            ui.note("Some models in staging or production have no passed validation in the last 90 days.", "medium")
        if (~table["file unchanged"]).any():
            ui.note("A model file differs from the one that was registered. It cannot be promoted or served by stage until it is registered again.", "high")


def _register(choice: str) -> None:
    ui.note("This model is not registered yet. Give it an owner and describe it; it starts as a draft.", "accent")
    c1, c2 = st.columns(2)
    owner = c1.text_input("Owner", key="mg_owner")
    risk = c2.selectbox("Risk", list(gv.RISKS), index=1, key="mg_risk", help="High-risk models need the approval of someone other than the owner before production.")
    description = st.text_area("What is it for?", key="mg_description")
    intended = st.text_input("Intended use", key="mg_intended")
    limits = st.text_input("Known limitations", key="mg_limits")
    actor = _actor()
    if st.button("Register", key="mg_register", type="primary"):
        try:
            gv.register(choice, owner, description, intended, limits, risk, actor or None)
        except ValueError as e:
            ui.note(str(e), "high")
        else:
            st.rerun()


def _card(record: dict, actor: str) -> None:
    with st.expander(t("Model card")):
        c1, c2 = st.columns(2)
        owner = c1.text_input("Owner", record["owner"], key=f"mg_c_owner_{record['model']}")
        risk = c2.selectbox("Risk", list(gv.RISKS), index=gv.RISKS.index(record["risk"]), key=f"mg_c_risk_{record['model']}")
        description = st.text_area("What is it for?", record["description"], key=f"mg_c_desc_{record['model']}")
        intended = st.text_input("Intended use", record["intended_use"], key=f"mg_c_use_{record['model']}")
        limits = st.text_input("Known limitations", record["limitations"], key=f"mg_c_lim_{record['model']}")
        if st.button("Save the card", key="mg_save_card"):
            try:
                gv.update_card(record["model"], actor or None, owner=owner, risk=risk, description=description, intended_use=intended, limitations=limits)
            except ValueError as e:
                ui.note(str(e), "high")
            else:
                st.rerun()
        st.download_button("Download the card (Markdown)", gv.model_card(record["model"]).encode("utf-8"), f"{record['model']}_card.md", "text/markdown", key="mg_dl_card")


def _validation(record: dict, df: pd.DataFrame, actor: str) -> None:
    st.subheader(t("Validation on new data"))
    target = record["target"]
    if target not in df.columns:
        ui.note(f"The open data has no column '{target}', so it cannot be used to validate this model. Open rows with the real results.", "low")
    else:
        c1, c2 = st.columns(2)
        drop = c1.slider("Tolerated worsening against training", 0.0, 1.0, 0.2, 0.05, format="%.2f", key="mg_drop")
        raw = c2.checkbox("The data is uncleaned (repeat the training cleaning steps)", key="mg_raw")
        if st.button("Validate with the open data", key="mg_validate"):
            try:
                result = gv.validate(record["model"], df, actor or None, raw, drop)
            except ValueError as e:
                ui.note(str(e), "high")
            else:
                ui.note(result["note"], "ok" if result["passed"] else "high" if result["passed"] is False else "medium")
    if record["validations"]:
        ui.table(pd.DataFrame(record["validations"])[["ts", "actor", "rows", "reference_metric", "new_value", "degradation", "passed"]].iloc[::-1])


def _promotion(record: dict, df: pd.DataFrame, actor: str) -> None:
    st.subheader(t("Stage"))
    ui.note(f"{record['model']} is in {record['stage']}.", STAGE_NOTE[record["stage"]])
    options = sorted(gv.TRANSITIONS[record["stage"]], key=gv.STAGES.index)
    c1, c2 = st.columns(2)
    to = c1.selectbox("Move to", options, key="mg_to")
    reason = c2.text_input("Reason (needed to override a check)", key="mg_reason")
    data = df if record["target"] in df.columns else None
    try:
        gates = gv.checks(record["model"], to, data)
    except ValueError as e:
        ui.note(str(e), "high")
        return
    for gate in gates:
        ui.note(gate["detail"], "ok" if gate["passed"] else "medium", bold=gate["name"])
    force = st.checkbox("Override the checks that fail (recorded in the audit log)", key="mg_force")
    if st.button("Move", key="mg_promote", type="primary"):
        try:
            out = gv.promote(record["model"], to, actor or None, data, reason, force)
        except ValueError as e:
            ui.note(str(e), "high")
        else:
            ui.note(f"{out['model']} is now in {out['to']}." + (f" {out['replaced']} was archived." if out["replaced"] else ""), "ok")
            st.rerun()
    if record["risk"] == "high":
        a, b = st.columns([2, 3])
        who = a.text_input("Approver", key="mg_approver")
        comment = b.text_input("Comment", key="mg_comment")
        if st.button("Approve this version", key="mg_approve"):
            try:
                gv.approve(record["model"], who, comment)
            except ValueError as e:
                ui.note(str(e), "high")
            else:
                st.rerun()
    if record["stage"] == "production" and st.button("Roll back to the previous production model", key="mg_rollback"):
        try:
            out = gv.rollback(record["target"], actor or None, reason)
        except ValueError as e:
            ui.note(str(e), "high")
        else:
            ui.note(f"{out['restored']} is back in production.", "ok")
            st.rerun()


def _model(df: pd.DataFrame) -> None:
    saved = list_models()
    if saved.empty:
        ui.note("There are no saved models. Train one in the Model lab and save it, then come back.", "low")
        return
    choice = st.selectbox("Saved model", list(saved["file"]), key="mg_model")
    try:
        record = gv.get(choice)
    except ValueError:
        _register(choice)
        return
    actor = _actor()
    check = gv.integrity(choice)
    if not check["ok"]:
        ui.note(check["message"], "high")
    ui.stats([("Stage", record["stage"]), ("Owner", record["owner"]), ("Risk", record["risk"]), ("Predicts", str(record["target"]))])
    _card(record, actor)
    _validation(record, df, actor)
    _promotion(gv.get(choice), df, actor)
    if record["approvals"]:
        st.caption(t("Approved by") + " " + ", ".join(a["actor"] for a in record["approvals"]))
    with st.expander(t("History")):
        ui.table(pd.DataFrame(record["stage_history"]))


def _audit() -> None:
    check = gv.verify_audit()
    ui.note(check["message"], "ok" if check["ok"] else "high")
    log = gv.audit_log()
    if len(log):
        view = log[["ts", "actor", "action", "model", "details"]].copy()
        view["details"] = view["details"].astype(str)
        ui.table(view.iloc[::-1])


def render(df: pd.DataFrame) -> None:
    tabs = st.tabs(["Overview", "Model", "Audit log"])
    with tabs[0]:
        _overview()
    with tabs[1]:
        _model(df)
    with tabs[2]:
        _audit()
