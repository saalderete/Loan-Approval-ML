import pandas as pd
import streamlit as st
from components import charts, ui
from components.backend import cached_model_info, cached_report
from components.page import require_model

from loan_approval.data import hmda
from loan_approval.data.validate import DOMAIN

status = require_model()
version = status["active_version"]
info = cached_model_info(version)
profile = cached_report(version, "profile")
quality = profile["quality"]

ui.page_title(
    "Data · provenance and quality",
    "Dataset Explorer",
    "Aggregate views of the 2025 Texas HMDA applications behind the model. No row-level records are stored "
    "in the app; every chart is computed from the training run's saved profile.",
)
cols = st.columns(4)
for i, (label, value, sub) in enumerate(
    [
        ("Modelled applications", f"{profile['rows']:,}", "after all filters"),
        ("Denied", f"{profile['denied']:,}", ui.pct(profile["denial_rate"]) + " of decided applications"),
        ("Raw file rows", f"{quality['funnel'][0]['rows']:,}", "approved + denied, Texas 2025"),
        ("Duplicates removed", f"{quality['duplicates_removed']:,}", "identical records"),
    ]
):
    with cols[i], st.container(border=True, key=f"card_ds_{i}"):
        ui.stat(label, value, sub)

st.markdown(
    ui.badge(info["dataset"]["source"])
    + " "
    + ui.badge(f"sha256 {info['dataset']['raw_sha256'][:16]}…", "mono"),
    unsafe_allow_html=True,
)
st.write("")

tab_funnel, tab_features, tab_reasons, tab_dict = st.tabs(
    ["Filtering & quality", "Feature explorer", "Denial reasons", "Data dictionary"]
)
with tab_funnel:
    with st.container(border=True, key="card_funnel"):
        st.markdown("##### From raw file to modelled population")
        a, b = st.columns([3, 2], gap="large")
        with a:
            st.altair_chart(charts.funnel_chart(quality["funnel"]), width="stretch")
        with b:
            funnel = pd.DataFrame(quality["funnel"])[["description", "rows", "denial_rate"]]
            st.dataframe(
                funnel,
                hide_index=True,
                width="stretch",
                column_config={
                    "description": "Rule",
                    "rows": st.column_config.NumberColumn("Rows", format="localized"),
                    "denial_rate": st.column_config.NumberColumn("Denied", format="percent"),
                },
            )
    a, b = st.columns([3, 2], gap="large")
    with a, st.container(border=True, key="card_missing"):
        st.markdown("##### Missing values leak the outcome")
        for w in quality["warnings"]:
            st.warning(w, icon=":material/warning:")
        miss = pd.DataFrame([m for m in quality["missingness"] if m["missing_rate"] > 0])
        miss["feature"] = miss["feature"].map(hmda.FEATURE_LABELS)
        for c in ("missing_rate", "denial_rate_when_missing", "denial_rate_when_present"):
            miss[c] = miss[c] * 100
        st.dataframe(
            miss,
            hide_index=True,
            width="stretch",
            column_config={
                "feature": "Input",
                "missing_rate": st.column_config.NumberColumn("Missing", format="%.1f%%"),
                "denial_rate_when_missing": st.column_config.NumberColumn(
                    "Denied if missing", format="%.1f%%"
                ),
                "denial_rate_when_present": st.column_config.NumberColumn(
                    "Denied if present", format="%.1f%%"
                ),
            },
        )
        st.caption(
            "Measured on site-built 1-4 unit applications before the completeness filter. Inputs with no "
            "missing values are omitted."
        )
    with b, st.container(border=True, key="card_ood"):
        st.markdown("##### Out-of-domain values removed")
        labels = {
            **hmda.FEATURE_LABELS,
            "loan_amount_vs_property_value": "Loan > 2.5× property value",
            "debt_to_income_ratio_unknown_code": "Unknown DTI code",
        }
        od = pd.DataFrame(
            [{"Rule": labels.get(k, k), "Rows removed": v} for k, v in quality["out_of_domain"].items()]
        )
        st.dataframe(
            od,
            hide_index=True,
            width="stretch",
            column_config={"Rows removed": st.column_config.NumberColumn(format="localized")},
        )
        st.caption(
            "Domain: loan $5k–$3M, property $10k–$10M, income $1k–$5M, CLTV 1–250%, term up to 40 years."
        )

with tab_features:
    feature = st.selectbox(
        "Input", hmda.INPUT_FEATURES, format_func=hmda.FEATURE_LABELS.get, key="ds_feature"
    )
    st.caption(hmda.FEATURE_GLOSSARY[feature])
    a, b = st.columns(2, gap="large")
    if feature in hmda.NUMERIC_INPUTS:
        num = profile["numeric"][feature]
        with a, st.container(border=True, key="card_numhist"):
            st.markdown("##### Distribution (1st–99th percentile)")
            st.altair_chart(
                charts.numeric_hist_chart(num["bins"], hmda.FEATURE_LABELS[feature]), width="stretch"
            )
            ui.kv(
                [
                    ("Median", f"{num['median']:,.1f}"),
                    ("Mean", f"{num['mean']:,.1f}"),
                    ("Model domain", f"{DOMAIN[feature][0]:,.0f} – {DOMAIN[feature][1]:,.0f}"),
                ]
            )
        with b, st.container(border=True, key="card_numrate"):
            st.markdown("##### Denial rate by decile")
            dec = pd.DataFrame(num["denial_by_decile"])
            st.dataframe(
                dec,
                hide_index=True,
                width="stretch",
                column_config={
                    "range": "Range",
                    "n": st.column_config.NumberColumn("Applications", format="localized"),
                    "denial_rate": st.column_config.ProgressColumn(
                        "Denial rate", min_value=0, max_value=0.6, format="percent"
                    ),
                },
            )
    else:
        rows = profile["categorical"][feature]
        with a, st.container(border=True, key="card_catrate"):
            st.markdown("##### Denial rate by category")
            st.altair_chart(charts.category_chart(rows, hmda.FEATURE_LABELS[feature]), width="stretch")
        with b, st.container(border=True, key="card_cattable"):
            st.markdown("##### Counts")
            st.dataframe(
                pd.DataFrame(rows)[["label", "n", "share", "denial_rate"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "label": "Category",
                    "n": st.column_config.NumberColumn("Applications", format="localized"),
                    "share": st.column_config.NumberColumn("Share", format="percent"),
                    "denial_rate": st.column_config.NumberColumn("Denial rate", format="percent"),
                },
            )
    st.caption("Raw rates mix many factors together; they are descriptive, not causal.")

with tab_reasons, st.container(border=True, key="card_reasons"):
    st.markdown("##### Primary denial reason reported by lenders")
    reasons = pd.DataFrame(profile["denial_reasons"])
    st.dataframe(
        reasons[["label", "n", "share"]],
        hide_index=True,
        width="stretch",
        column_config={
            "label": "Reason",
            "n": st.column_config.NumberColumn("Denied applications", format="localized"),
            "share": st.column_config.ProgressColumn(
                "Share of denials", min_value=0, max_value=1, format="percent"
            ),
        },
    )
    st.caption(
        "Denial reasons are recorded after the decision, so they are never model inputs. Shown here only "
        "to describe the data."
    )

with tab_dict, st.container(border=True, key="card_dict"):
    st.markdown("##### Model inputs")
    fields = info["input_schema"]["fields"]
    dd = pd.DataFrame(
        [
            {
                "Field": f["name"],
                "Label": f["label"],
                "Type": f["type"],
                "Description": f["description"],
                "Allowed": (
                    f"{f['min']:,.0f} – {f['max']:,.0f}"
                    if f["type"] == "numeric"
                    else ", ".join(f"{k} = {v}" for k, v in f["allowed"].items())
                ),
            }
            for f in fields
        ]
    )
    st.dataframe(dd, hide_index=True, width="stretch")
    st.download_button(
        "Download data dictionary (.csv)",
        dd.to_csv(index=False),
        file_name="data_dictionary.csv",
        icon=":material/download:",
    )
    st.markdown("##### Excluded on purpose")
    ex = info["excluded_columns"]
    st.dataframe(
        pd.DataFrame(
            [
                {"Group": "Post-decision (leaks the outcome)", "Columns": ", ".join(ex["post_decision"])},
                {"Group": "Process artifacts", "Columns": ", ".join(ex["process_artifacts"])},
                {
                    "Group": "Protected (fairness audit only)",
                    "Columns": ", ".join(ex["protected_audit_only"]),
                },
                {"Group": "Geography / lender (proxy risk)", "Columns": ", ".join(ex["proxy_risk"])},
            ]
        ),
        hide_index=True,
        width="stretch",
    )
