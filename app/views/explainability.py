import pandas as pd
import streamlit as st
from components import charts, ui
from components.backend import cached_model_info, cached_report
from components.page import require_model

from loan_approval.data import hmda
from loan_approval.inference.explain import EXPLANATION_GROUPS, LOCAL_METHOD

status = require_model()
version = status["active_version"]
info = cached_model_info(version)
importance = cached_report(version, "importance")
leakage = cached_report(version, "leakage_screen")
names = info["algorithms"]
sel = info["selected_algorithm"]

ui.page_title(
    "Model · explanations",
    "Explainability",
    "Global explanations show what each model relies on across held-out applications. Local explanations "
    "(in the Scenario Explorer) show what moved one estimate. Neither describes real-world causes.",
)

left, right = st.columns([3, 2], gap="large")
with left, st.container(border=True, key="card_global"):
    st.markdown("##### Global importance")
    pick = (
        st.segmented_control(
            "Model", list(importance["permutation"]), format_func=names.get, default=sel, key="imp_model"
        )
        or sel
    )
    perm = importance["permutation"][pick]
    st.altair_chart(charts.importance_chart(perm["features"]), width="stretch")
    st.caption(
        perm["method"] + ". Correlated inputs (loan amount, property value, LTV) share credit, so shuffling "
        "one of them understates the group's joint importance."
    )
with right:
    with st.container(border=True, key="card_methods"):
        st.markdown("##### Methods")
        st.markdown(
            "**Global — permutation importance.** Shuffle one input across held-out rows and measure how much "
            "PR-AUC falls. Model-agnostic, so all three candidates are comparable.\n\n"
            f"**Local — group Shapley values.** {LOCAL_METHOD}\n\n"
            "**Why groups?** Loan amount, property value and LTV only make sense together. Explaining them jointly "
            "avoids asking the model about impossible applications."
        )
        st.dataframe(
            pd.DataFrame([{"Group": v[0], "Inputs": ", ".join(v[1])} for v in EXPLANATION_GROUPS.values()]),
            hide_index=True,
            width="stretch",
        )
    with st.container(border=True, key="card_caveats"):
        st.markdown("##### Read with care")
        st.markdown(
            "- Importance is association, not causation.\n"
            "- An attribution is not a legally sufficient adverse-action reason.\n"
            "- Explanations are specific to one model version and are regenerated with every training run."
        )

coef = importance.get("logistic_coefficients") or []
c1, c2 = st.columns(2, gap="large")
with c1, st.container(border=True, key="card_coef"):
    st.markdown("##### Logistic regression coefficients (baseline)")
    if coef:
        df = pd.DataFrame(coef[:20])
        st.dataframe(
            df,
            hide_index=True,
            width="stretch",
            column_config={
                "term": "Encoded term",
                "coefficient": st.column_config.NumberColumn("Coefficient (log-odds)", format="%.3f"),
                "odds_ratio": st.column_config.NumberColumn("Odds ratio", format="%.2f"),
            },
        )
        st.caption(
            "Numeric terms are standardised (per 1 SD); categorical terms are one-hot vs. no category, "
            "with correlated terms sharing weight. Read signs and rough size, not exact effects. "
            "Coefficients come from the uncalibrated baseline."
        )
with c2, st.container(border=True, key="card_leak"):
    st.markdown("##### Leakage screen")
    df = pd.DataFrame(leakage["univariate_auc"])
    df["feature"] = df["feature"].map(hmda.FEATURE_LABELS)
    st.dataframe(
        df[["feature", "auc"]],
        hide_index=True,
        width="stretch",
        column_config={
            "feature": "Input",
            "auc": st.column_config.ProgressColumn(
                "Single-input ROC-AUC", min_value=0.5, max_value=1.0, format="%.3f"
            ),
        },
    )
    st.caption(
        "ROC-AUC of each input on its own (selection split). A single field near 1.0 usually means it "
        "leaks the outcome; the promotion gate requires every input to stay below 0.95."
    )
