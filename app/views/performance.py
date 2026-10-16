import pandas as pd
import streamlit as st
from components import charts, ui
from components.backend import cached_model_info, cached_report
from components.page import require_model

status = require_model()
version = status["active_version"]
info = cached_model_info(version)
metrics = cached_report(version, "metrics")
curves = cached_report(version, "curves")
names = info["algorithms"]
sel = metrics["selected_algorithm"]
test = metrics["test"]
n_test = test[sel]["probability"]["n"]
prevalence = test[sel]["probability"]["prevalence"]

ui.page_title(
    "Model · held-out evaluation",
    "Model Performance",
    "Three candidates trained with the same protocol and compared on one locked test split. The model was "
    f"selected by {metrics['primary_metric'].upper().replace('_', '-')} on a separate selection split before "
    "the test split was scored, so these numbers are not cherry-picked.",
)
st.markdown(
    ui.badge(f"{n_test:,} test applications", "accent")
    + " "
    + ui.badge(f"{test[sel]['probability']['positives']:,} denied ({ui.pct(prevalence)})")
    + " "
    + ui.badge(f"split {info['dataset']['split_hash'][:12]}", "mono")
    + " "
    + ui.badge(f"{test[sel]['ci']['n_boot']} bootstrap resamples"),
    unsafe_allow_html=True,
)
st.write("")

rows = []
for algo, r in test.items():
    p, t, ci = r["probability"], r["at_threshold"], r["ci"]
    cv = metrics["cv"][algo]
    rows.append(
        {
            "Model": r["name"] + (" ★" if algo == sel else ""),
            "ROC-AUC": p["roc_auc"],
            "ROC-AUC 95% CI": f"{ci['roc_auc']['low']:.3f}–{ci['roc_auc']['high']:.3f}",
            "PR-AUC": p["pr_auc"],
            "PR-AUC 95% CI": f"{ci['pr_auc']['low']:.3f}–{ci['pr_auc']['high']:.3f}",
            "Brier": p["brier"],
            "ECE": p["ece"],
            "Log loss": p["log_loss"],
            "Recall @ thr": t["recall"],
            "Precision @ thr": t["precision"],
            "F1 @ thr": t["f1"],
            "Balanced acc.": t["balanced_accuracy"],
            "CV PR-AUC": f"{cv['mean']:.3f} ± {cv['std']:.3f}",
            "Fit (s)": metrics["fit_seconds"][algo],
        }
    )
d = metrics["dummy_baseline"]
df = pd.DataFrame(rows)
fmt3 = st.column_config.NumberColumn(format="%.3f")
fmt4 = st.column_config.NumberColumn(format="%.4f")
pctc = st.column_config.NumberColumn(format="percent")
with st.container(border=True, key="card_table"):
    st.markdown("##### Test-split comparison")
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "ROC-AUC": fmt3,
            "PR-AUC": fmt3,
            "Brier": fmt4,
            "ECE": fmt4,
            "Log loss": fmt4,
            "Recall @ thr": pctc,
            "Precision @ thr": pctc,
            "F1 @ thr": pctc,
            "Balanced acc.": pctc,
        },
    )
    st.caption(
        f"★ selected model. A prevalence-only baseline (every application gets {ui.pct(d['prevalence'])}) "
        f"scores ROC-AUC {d['roc_auc']:.3f}, PR-AUC {d['pr_auc']:.3f} and Brier {d['brier']:.4f}. Threshold metrics use each model's research threshold (max F1 on the selection "
        "split). They illustrate trade-offs; no threshold here is a lending policy."
    )

tabs = st.tabs(
    ["ROC", "Precision-recall", "Calibration", "Score distribution", "Confusion matrix", "Protocol"]
)
with tabs[0], st.container(border=True, key="card_roc"):
    st.altair_chart(charts.curve_chart(curves, names, sel, "roc"), width="stretch")
    st.caption(f"Dashed line: random ranking. Source: {version}, test split (n = {n_test:,}).")
with tabs[1], st.container(border=True, key="card_pr"):
    st.altair_chart(charts.curve_chart(curves, names, sel, "pr", prevalence), width="stretch")
    st.caption(
        f"Denials are the minority class ({ui.pct(prevalence)}), so precision-recall is more informative "
        "than ROC. Dashed line: prevalence-only baseline."
    )
with tabs[2], st.container(border=True, key="card_cal"):
    st.altair_chart(charts.calibration_chart(curves, names, sel), width="stretch")
    rows_cal = [
        {
            "Model": test[a]["name"],
            "ECE (calibrated)": test[a]["probability"]["ece"],
            "ECE (raw model)": test[a]["uncalibrated"]["ece"],
            "Brier (calibrated)": test[a]["probability"]["brier"],
            "Brier (raw model)": test[a]["uncalibrated"]["brier"],
        }
        for a in test
    ]
    st.dataframe(
        pd.DataFrame(rows_cal),
        hide_index=True,
        width="stretch",
        column_config={
            k: fmt4
            for k in ["ECE (calibrated)", "ECE (raw model)", "Brier (calibrated)", "Brier (raw model)"]
        },
    )
    st.caption(
        "Points are deciles of estimated probability. On the diagonal = an estimate of 30% means about 30% "
        "of those applications were denied. Isotonic calibration was fitted on its own split."
    )
with tabs[3], st.container(border=True, key="card_hist"):
    pick = (
        st.segmented_control("Model", list(curves), format_func=names.get, default=sel, key="hist_model")
        or sel
    )
    st.altair_chart(
        charts.histogram_chart(curves[pick]["histogram"], metrics["thresholds"][pick]), width="stretch"
    )
    st.caption(
        "Stacked counts of test applications by estimated probability. Dashed line: research threshold."
    )
with tabs[4], st.container(border=True, key="card_cm"):
    pick = (
        st.segmented_control("Model", list(test), format_func=names.get, default=sel, key="cm_model") or sel
    )
    t = test[pick]["at_threshold"]
    cm = t["confusion_matrix"]
    total = sum(cm.values())
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Threshold {t['threshold']:.3f}** · {total:,} applications")
        grid = pd.DataFrame(
            [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            index=["Actually approved", "Actually denied"],
            columns=["Estimated below threshold", "Estimated above threshold"],
        )
        st.dataframe(grid, width="stretch")
    with b:
        ui.kv(
            [
                (
                    "Recall (denials caught)",
                    f"{ui.pct(t['recall'])}  ({cm['tp']:,} / {cm['tp'] + cm['fn']:,})",
                ),
                ("Precision", f"{ui.pct(t['precision'])}  ({cm['tp']:,} / {cm['tp'] + cm['fp']:,})"),
                (
                    "False positive rate",
                    f"{ui.pct(t['false_positive_rate'])}  ({cm['fp']:,} / {cm['fp'] + cm['tn']:,})",
                ),
                ("Flagged share", ui.pct(t["flagged_rate"])),
                ("Balanced accuracy", ui.pct(t["balanced_accuracy"])),
            ]
        )
with tabs[5], st.container(border=True, key="card_protocol"):
    splits = pd.DataFrame(metrics["splits"])
    st.dataframe(
        splits,
        hide_index=True,
        width="stretch",
        column_config={
            "rows": st.column_config.NumberColumn("Rows", format="localized"),
            "positives": st.column_config.NumberColumn("Denied", format="localized"),
            "prevalence": st.column_config.NumberColumn("Denial rate", format="percent"),
        },
    )
    for i, (title, body) in enumerate(
        [
            (
                "Split before fitting",
                "Stratified 60/10/10/20 split (train / calibration / selection / test), seed "
                f"{info['dataset']['split_seed']}, hash recorded in the manifest.",
            ),
            (
                "Tune on train only",
                f"{info['training_config']['tuning']['cv_folds']}-fold stratified CV on a "
                f"{info['training_config']['tuning']['max_rows']:,}-row training subsample, scored by PR-AUC.",
            ),
            (
                "Calibrate separately",
                "Isotonic regression fitted on the calibration split, never on train or test.",
            ),
            (
                "Select, then test once",
                "Candidate and threshold chosen on the selection split; every candidate scored "
                "once on the test split afterwards.",
            ),
        ],
        1,
    ):
        ui.step(i, title, body)
        st.write("")
    params = pd.DataFrame(
        [
            {"Model": names[a], "Parameters": ", ".join(f"{k}={v}" for k, v in p.items())}
            for a, p in metrics["params"].items()
        ]
    )
    st.dataframe(params, hide_index=True, width="stretch")
