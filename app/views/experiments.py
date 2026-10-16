import json

import pandas as pd
import streamlit as st
import yaml
from components import ui
from components.backend import get_backend

backend = get_backend()
exp = backend.experiments()

ui.page_title(
    "Operations · model registry",
    "Experiment Lab",
    "Every training run is a versioned, checksummed artifact with its config, metrics and promotion gates. "
    "Only a version that passes every gate can be promoted, and promotion is an explicit, logged admin action.",
)

lifecycle = exp["lifecycle"]
st.markdown(
    " → ".join(ui.badge(s.replace("_", " "), "accent" if s == "demo_approved" else "") for s in lifecycle),
    unsafe_allow_html=True,
)
st.write("")

if not exp["versions"]:
    ui.empty_state(
        "No model versions yet",
        "Run a training job to create the first version.",
        "loan-approval train            # real HMDA data\nloan-approval train --synthetic --fast",
    )
    st.stop()

rows = []
for v in exp["versions"]:
    m = v.get("headline_metrics", {})
    rows.append(
        {
            "Version": v["version"],
            "Active": v["version"] == exp["active_version"],
            "Status": v["status"],
            "Model": v.get("selected_algorithm"),
            "Data": "synthetic" if v.get("synthetic") else "HMDA",
            "Rows": v.get("rows"),
            "PR-AUC": m.get("pr_auc"),
            "ROC-AUC": m.get("roc_auc"),
            "Gates": "all passed" if v.get("gates_passed") else "failed",
            "Train (s)": v.get("training_seconds"),
            "Commit": v.get("git_commit"),
            "Created": v.get("created_at"),
        }
    )
with st.container(border=True, key="card_versions"):
    st.markdown("##### Registered versions")
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Active": st.column_config.CheckboxColumn(),
            "Rows": st.column_config.NumberColumn(format="localized"),
            "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
            "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
        },
    )

pick = st.selectbox("Inspect version", [v["version"] for v in exp["versions"]], key="exp_pick")
entry = next(v for v in exp["versions"] if v["version"] == pick)
with st.container(border=True, key="card_gates"):
    st.markdown("##### Promotion gates")
    gates = pd.DataFrame(entry.get("gates", []))
    if not gates.empty:
        gates["passed"] = gates["passed"].map({True: "pass", False: "FAIL"})
        st.dataframe(
            gates.rename(columns={"name": "Gate", "passed": "Result", "detail": "Detail"}),
            hide_index=True,
            width="stretch",
        )
with st.container(border=True, key="card_artifacts"):
    st.markdown("##### Artifacts and provenance")
    ui.kv(
        [
            ("Status", entry["status"]),
            ("Dataset", entry.get("dataset", "")),
            ("Manifest SHA-256", entry.get("manifest_sha256", "")[:24] + "…"),
            *[(f"{k}.joblib", v[:24] + "…") for k, v in entry.get("artifact_sha256", {}).items()],
            ("Git commit", entry.get("git_commit", "")),
        ]
    )
    st.caption("Artifacts are loaded only if their SHA-256 matches the registry and the manifest.")

if pick == exp["active_version"]:
    info = backend.model_info()
    metrics = backend.report("metrics")
    with st.expander("Hyper-parameter search (cross-validation on the training split)"):
        for algo, cv in metrics["cv"].items():
            st.markdown(
                f"**{info['algorithms'][algo]}** · best {cv['best_params']} · "
                f"PR-AUC {cv['mean']:.4f} ± {cv['std']:.4f} over {cv['folds']} folds ({cv['rows']:,} rows)"
            )
            st.dataframe(
                pd.DataFrame(
                    [{**g["params"], "mean PR-AUC": g["mean"], "std": g["std"]} for g in cv["grid"]]
                ),
                hide_index=True,
                width="stretch",
            )
    with st.expander("Training configuration"):
        st.code(yaml.safe_dump(info["training_config"], sort_keys=False), language="yaml")
    with st.expander("Package versions"):
        st.code(json.dumps(info["provenance"]["packages"], indent=2), language="json")

with st.container(border=True, key="card_audit"):
    st.markdown("##### Registry audit log")
    log = pd.DataFrame(exp["audit_log"])
    if log.empty:
        st.caption("No entries yet.")
    else:
        st.dataframe(log.fillna("").astype(str), hide_index=True, width="stretch")

with st.container(border=True, key="card_howto"):
    st.markdown("##### Train and promote (admins only)")
    st.caption(
        "Training and promotion are not available to dashboard visitors. Use the CLI or the admin API "
        "(requires ADMIN_TOKEN)."
    )
    st.code(
        "loan-approval train                         # real HMDA data (after fetch + prepare)\n"
        "loan-approval list\n"
        'loan-approval promote <version> --actor "Your Name" --reason "Reviewed gates and model card"\n\n'
        'curl -X POST $API/api/v1/train-jobs -H "X-Admin-Token: $ADMIN_TOKEN" '
        '-H \'Idempotency-Key: run-42\' -d \'{"dataset": "synthetic", "rows": 20000}\'',
        language="bash",
    )
