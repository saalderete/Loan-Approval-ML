import streamlit as st
from components import ui
from components.backend import cached_report
from components.page import require_model

status = require_model()
version = status["active_version"]
metrics = cached_report(version, "metrics")
fairness = cached_report(version, "fairness")
importance = cached_report(version, "importance")
profile = cached_report(version, "profile")
sel = status["selected_algorithm"]
test = metrics["test"][sel]
p = test["probability"]
ci = test["ci"]

ui.hero(
    "Mortgage decision research lab · 2025 Texas HMDA",
    f"How did Texas lenders decide on {profile['rows']:,} mortgage applications?",
    "An explainable model of historical approve/deny decisions, trained on public federal HMDA data, "
    "with calibrated estimates, a fair-lending audit and a reproducible pipeline you can inspect end to end.",
)
st.write("")
ui.notice(
    "Estimates describe patterns in past lender decisions for fictional scenarios. They are not "
    "lending decisions, credit scores or advice, and the model has never been validated for real use."
)
st.write("")

cols = st.columns(5)
cards = [
    ("Applications modelled", f"{profile['rows']:,}", "Texas 2025, after filters"),
    ("Historical denial rate", ui.pct(profile["denial_rate"]), "of decided applications"),
    (
        "Test ROC-AUC",
        ui.num(p["roc_auc"]),
        f"95% CI {ui.num(ci['roc_auc']['low'])}–{ui.num(ci['roc_auc']['high'])}",
    ),
    ("Test PR-AUC", ui.num(p["pr_auc"]), f"baseline {ui.num(metrics['dummy_baseline']['pr_auc'])}"),
    ("Calibration error", ui.num(p["ece"], 4), "ECE on the test split"),
]
for i, (col, (label, value, sub)) in enumerate(zip(cols, cards, strict=True)):
    with col, st.container(border=True, key=f"card_stat_{i}"):
        ui.stat(label, value, sub)

st.markdown(
    ui.badge(f"Selected: {status['algorithm_name']}", "accent")
    + " "
    + ui.badge(version, "mono")
    + " "
    + ui.badge(f"{p['n']:,} held-out test applications")
    + " "
    + ui.badge(f"Trained {status['created_at'][:10]}"),
    unsafe_allow_html=True,
)
st.write("")

c1, c2, c3 = st.columns(3)
with c1, st.container(border=True, key="card_cta_scenario"):
    st.markdown("#### Explore a scenario")
    st.caption(
        "Pick a fictional applicant, change the inputs and see how the model's estimate and its "
        "Shapley explanation respond."
    )
    st.page_link("views/scenario.py", label="Open Scenario Explorer", icon=":material/arrow_forward:")
with c2, st.container(border=True, key="card_cta_perf"):
    st.markdown("#### Model performance")
    st.caption(
        "Three candidate models, one locked test split: ROC and PR curves, calibration, "
        "confusion matrices and bootstrap confidence intervals."
    )
    st.page_link("views/performance.py", label="Open Model Performance", icon=":material/arrow_forward:")
with c3, st.container(border=True, key="card_cta_fair"):
    st.markdown("#### Fairness audit")
    st.caption(
        "Historical denial-rate disparities and model error rates by race, ethnicity, sex and age. "
        "Protected attributes are never model inputs."
    )
    st.page_link(
        "views/responsible.py", label="Open Fairness & Responsible AI", icon=":material/arrow_forward:"
    )

st.write("")
left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown("### What the data shows")
    top = importance["permutation"][sel]["features"][:3]
    race = next(t for t in fairness["model_audit"] if t["attribute"] == "derived_race")
    reportable = [g for g in race["groups"] if g["status"] == "ok" and not g["not_reported"]]
    widest = max(reportable, key=lambda g: g["unexplained_gap"])
    with st.container(border=True, key="card_findings"):
        st.markdown(
            f"- **{top[0]['label']}** is the input the model relies on most, followed by "
            f"{top[1]['label'].lower()} and {top[2]['label'].lower()} (permutation importance on held-out data).\n"
            f"- Calibrated estimates track observed denial rates closely: expected calibration error is "
            f"**{p['ece']:.4f}** on {p['n']:,} test applications.\n"
            f"- **{widest['group']}** applicants were denied {ui.pct(widest['observed_denial_rate'])} of the time "
            f"in the test split, while the model's financial inputs average {ui.pct(widest['mean_estimated_probability'])}: "
            f"a **{widest['unexplained_gap'] * 100:.1f}-point gap** the 13 HMDA fields don't account for. "
            "HMDA has no credit scores, so this is a signal for review, not proof of discrimination.\n"
            f"- At the research threshold the selected model catches {ui.pct(test['at_threshold']['recall'])} of "
            f"denials with {ui.pct(test['at_threshold']['precision'])} precision: decent ranking, but far from "
            "a decision-grade model."
        )
with right:
    st.markdown("### How it works")
    steps = [
        ("Download and verify", "Public HMDA file from the FFIEC Data Browser, recorded with its SHA-256."),
        (
            "Filter and validate",
            "Approved/denied, complete, in-domain applications; every dropped row counted.",
        ),
        (
            "Train three models",
            "Stratified split fixed up front; CV tuning on train only; isotonic calibration.",
        ),
        ("Select, then test once", "Chosen by PR-AUC on a selection split; the test split is scored once."),
        ("Audit and explain", "Fairness by group, permutation importance and per-scenario Shapley values."),
    ]
    for i, (t, d) in enumerate(steps, 1):
        ui.step(i, t, d)
        st.write("")

ui.footer(
    f"Source: {status['dataset']['source']} · Model {version} · "
    "Code and methodology: github.com/saalderete/Loan-Approval-ML"
)
