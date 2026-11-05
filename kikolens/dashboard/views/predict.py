import io

import joblib
import pandas as pd
import streamlit as st

from kikolens.analyze import predict as pr
from kikolens.dashboard import components as ui
from kikolens.utils.loaders import load_data

TITLE = "Predict"
SUBTITLE = "Use a trained model on new data, keep it for later and see what it expects."

SESSION_MODELS = {"lab_res": "Model lab", "automl_res": "Classical model", "nn_res": "Neural network"}


def _session_bundles(df: pd.DataFrame) -> dict:
    """Modelos entrenados en esta sesión, empaquetados contra el dataset actual (que es el de entrenamiento)."""
    bundles = {}
    for key, label in SESSION_MODELS.items():
        res = st.session_state.get(key)
        if not res or res.get("target") not in df.columns:
            continue
        try:
            bundles[f"This session · {label} → {res['target']}"] = pr.build_bundle(
                res, df, name=f"{label} for {res['target']}", recipe=st.session_state.get("recipe"))
        except (ValueError, KeyError):
            continue
    for label, bundle in st.session_state.get("restored_models", {}).items():
        if isinstance(bundle, dict) and pr.BUNDLE_KEY in bundle:
            bundles[f"Opened project · {bundle['name']}"] = bundle
    return bundles


def _registry_choices() -> dict:
    frame = pr.list_models()
    return {f"Saved · {row['name']} ({row['file']})": row["file"] for _, row in frame.iterrows()}


def _pick_model(df: pd.DataFrame):
    options = _session_bundles(df)
    saved = _registry_choices()
    labels = list(options) + list(saved) + ["Open a model file..."]
    choice = st.selectbox("Model", labels, key="pd_model")
    if choice in options:
        return options[choice], "session"
    if choice in saved:
        try:
            return pr.load_bundle(pr.resolve_model(saved[choice])), "registry"
        except ValueError as e:
            ui.note(str(e), "high")
            return None, ""
    file = st.file_uploader("Model file (.klmodel)", type=[pr.EXTENSION.lstrip(".")], key="pd_file")
    st.caption("Only open model files you created or trust: they can run code when opened.")
    trust = st.checkbox("I trust this file", False, key="pd_trust")
    if file is not None and trust:
        try:
            bundle = joblib.load(io.BytesIO(file.getvalue()))
            if not isinstance(bundle, dict) or pr.BUNDLE_KEY not in bundle:
                raise ValueError("This is not a KikoLens model package.")
            return bundle, "file"
        except Exception as e:
            ui.note(f"Could not open it: {e}", "high")
    return None, ""


def _summary(bundle: dict) -> None:
    meta = pr.metadata(bundle)
    score = next(iter(meta["metrics"].items()), None)
    ui.stats([("Predicts", meta["target"], meta["task"]), ("Algorithm", meta["label"]),
              ("Trained on", f"{meta['rows_trained']:,} rows"),
              (score[0] if score else "Score", f"{score[1]:.3f}" if score else "n/a", "held-out test")])
    with st.expander("What this model expects"):
        rows = []
        for col, info in bundle["schema"].items():
            if info["kind"] == "numeric":
                span = f"{info['min']:.4g} to {info['max']:.4g}"
            elif info["kind"] == "datetime":
                span = f"{info['min'][:10]} to {info['max'][:10]}"
            else:
                span = ", ".join(info["values"][:6]) + (" ..." if info["n_unique"] > 6 else "")
            rows.append({"column": col, "type": info["kind"], "seen in training": span, "empty %": info["missing_pct"]})
        ui.table(pd.DataFrame(rows), hide_index=True)
        st.download_button("Download model card (Markdown)", pr.model_card(bundle).encode("utf-8"),
                           "model_card.md", "text/markdown", key="pd_card")


def _show_warnings(report: dict) -> None:
    notes = pr.input_warnings(report)
    if not notes:
        ui.note("The new data matches what the model saw in training.", "ok")
    for severity, message in notes:
        ui.note(message, severity)


def _batch(bundle: dict, df: pd.DataFrame, origin: str) -> None:
    source = st.radio("Data to predict", ["The dataset open now", "Another file"], horizontal=True, key="pd_src")
    new = df
    if source == "Another file":
        file = st.file_uploader("File with the same columns", key="pd_new")
        if file is None:
            return
        try:
            new = load_data(file)
        except Exception as e:
            ui.note(str(e), "high")
            return
    apply_recipe = False
    if bundle.get("recipe") and source == "Another file":
        apply_recipe = st.checkbox("Apply the cleaning steps used for training first", True, key="pd_recipe",
                                   help="Use this when the file is raw, as it was before any cleaning.")
    if st.button("Predict", key="pd_go", type="primary"):
        try:
            out, report = pr.predict_frame(bundle, new, apply_recipe=apply_recipe)
        except ValueError as e:
            ui.note(str(e), "high")
            return
        st.session_state.pd_result = (out, report, bundle["target"])

    result = st.session_state.get("pd_result")
    if not result or result[2] != bundle["target"]:
        return
    out, report, target = result
    _show_warnings(report)
    column = f"predicted_{target}"
    if bundle["task"] == "classification":
        counts = out[column].value_counts().rename_axis("class").reset_index(name="rows")
        c1, c2 = st.columns([1, 2])
        with c1:
            ui.table(counts, hide_index=True)
        with c2:
            if "confidence" in out:
                low = int((out["confidence"] < 0.6).sum())
                ui.note(f"{low:,} of {len(out):,} rows have a confidence under 60%; review those first.",
                        "medium" if low else "ok")
    else:
        ui.stats([("Rows", f"{len(out):,}"), ("Mean prediction", f"{out[column].mean():,.4g}"),
                  ("Lowest", f"{out[column].min():,.4g}"), ("Highest", f"{out[column].max():,.4g}")])
    ui.table(out.head(500), height=340)
    ui.download_csv(out, f"{target}_predictions.csv", "Download predictions (CSV)")


def _single(bundle: dict) -> None:
    cols = st.columns(3)
    values = {}
    for i, (col, info) in enumerate(bundle["schema"].items()):
        box = cols[i % 3]
        if info["kind"] == "numeric":
            values[col] = box.number_input(col, value=info["median"], key=f"pd_in_{col}",
                                           help=f"Seen in training: {info['min']:.4g} to {info['max']:.4g}")
        elif info["kind"] == "datetime":
            values[col] = box.date_input(col, value=pd.Timestamp(info["max"]).date(), key=f"pd_in_{col}")
        else:
            values[col] = box.selectbox(col, info["values"] or [""], key=f"pd_in_{col}")
    if st.button("Predict this row", key="pd_one", type="primary"):
        try:
            out, report = pr.predict_frame(bundle, pd.DataFrame([values]))
        except ValueError as e:
            ui.note(str(e), "high")
            return
        target = bundle["target"]
        value = out[f"predicted_{target}"].iloc[0]
        st.metric("Prediction", f"{value:,.4f}" if bundle["task"] == "regression" else str(value))
        if f"{target}_low" in out:
            st.caption(f"Likely range: {out[f'{target}_low'].iloc[0]:,.4g} to {out[f'{target}_high'].iloc[0]:,.4g} "
                       "(90% of test errors fell inside it).")
        if "confidence" in out:
            probs = out.filter(like="probability_").iloc[0].rename_axis("class").reset_index(name="probability")
            probs["class"] = probs["class"].str.replace("probability_", "", regex=False)
            ui.table(probs.sort_values("probability", ascending=False).round(4), hide_index=True)
        for severity, message in pr.input_warnings(report):
            if severity != "low":
                ui.note(message, severity)


def _library(bundle: dict, origin: str) -> None:
    name = st.text_input("Name", bundle["name"], key="pd_name")
    if origin != "registry" and st.button("Save to my models", key="pd_save", type="primary"):
        try:
            path = pr.save_bundle({**bundle, "name": name.strip() or bundle["name"]})
        except OSError as e:
            ui.note(f"Could not save the model: {e}", "high")
        else:
            ui.note(f"Saved as {path.name} in {path.parent}. Saving the same name again creates a new version.", "ok")
            st.rerun()
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer)
    st.download_button("Download model package", buffer.getvalue(), f"{pr.slugify(bundle['name'])}{pr.EXTENSION}",
                       key="pd_dl", help="Open it later in this page or with: kikolens score data.csv model.klmodel")

    st.markdown("#### Saved models")
    frame = pr.list_models()
    if frame.empty:
        st.caption("Nothing saved yet.")
        return
    ui.table(frame, hide_index=True)
    victim = st.selectbox("Delete a saved model", ["(none)"] + frame["file"].tolist(), key="pd_del")
    if victim != "(none)" and st.button("Delete", key="pd_del_go"):
        pr.delete_model(victim)
        st.rerun()


def render(df: pd.DataFrame) -> None:
    bundle, origin = _pick_model(df)
    if bundle is None:
        ui.note("Train a model first (Models, Model lab or Neural network), save one, or open a model file.", "low")
        return
    _summary(bundle)
    tabs = st.tabs(["Predict a file", "Predict one row", "Save and share"])
    with tabs[0]:
        _batch(bundle, df, origin)
    with tabs[1]:
        _single(bundle)
    with tabs[2]:
        _library(bundle, origin)
