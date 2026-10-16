import pandas as pd
import streamlit as st
from components import charts, ui
from components.backend import BackendError, get_backend
from components.page import require_model, show_error

from loan_approval.data import hmda
from loan_approval.inference.explain import EXPLANATION_GROUPS

status = require_model()
backend = get_backend()
presets = {s["id"]: s for s in backend.demo_scenarios()}
TERMS = [120, 180, 240, 300, 360, 480]
CASTS = {
    "loan_amount": int,
    "property_value": int,
    "loan_term": int,
    "income": float,
    "combined_loan_to_value_ratio": float,
}


def load_preset(preset_id: str) -> None:
    for k, v in presets[preset_id]["inputs"].items():
        st.session_state[f"f_{k}"] = CASTS.get(k, str)(v)
    st.session_state.pop("last_error", None)


if "preset" not in st.session_state:
    st.session_state["preset"] = next(iter(presets))
if "f_loan_amount" not in st.session_state:
    load_preset(st.session_state["preset"])

ui.page_title(
    "Explore · fictional scenarios only",
    "Scenario Explorer",
    "Describe a fictional application and see how the trained model responds. The estimate is the share of "
    "similar historical Texas applications that lenders denied, not a decision about anyone.",
)

form_col, main_col = st.columns([4, 8], gap="large")

with form_col, st.container(border=True, key="card_form"):
    st.markdown("##### Start from a preset")
    st.selectbox(
        "Fictional preset",
        options=list(presets),
        format_func=lambda k: presets[k]["label"],
        key="preset",
        on_change=lambda: load_preset(st.session_state["preset"]),
        label_visibility="collapsed",
    )
    st.caption(presets[st.session_state["preset"]]["description"])
    st.markdown(
        ui.badge("Synthetic scenario", "amber") + " " + ui.badge("Not stored"), unsafe_allow_html=True
    )
    st.write("")
    with st.form("scenario_form", border=False):
        st.markdown("##### Loan")
        st.selectbox(
            "Loan purpose", list(hmda.LOAN_PURPOSE), format_func=hmda.LOAN_PURPOSE.get, key="f_loan_purpose"
        )
        st.selectbox("Loan type", list(hmda.LOAN_TYPE), format_func=hmda.LOAN_TYPE.get, key="f_loan_type")
        a, b = st.columns(2)
        a.number_input(
            "Loan amount ($)",
            min_value=5_000,
            max_value=3_000_000,
            step=5_000,
            key="f_loan_amount",
            help=hmda.FEATURE_GLOSSARY["loan_amount"],
        )
        b.selectbox("Term", TERMS, format_func=lambda m: f"{m // 12} years", key="f_loan_term")
        a, b = st.columns(2)
        a.selectbox("Lien", list(hmda.LIEN_STATUS), format_func=hmda.LIEN_STATUS.get, key="f_lien_status")
        b.selectbox(
            "Conforming",
            list(hmda.CONFORMING_LOAN_LIMIT),
            format_func=hmda.CONFORMING_LOAN_LIMIT.get,
            key="f_conforming_loan_limit",
            help=hmda.FEATURE_GLOSSARY["conforming_loan_limit"],
        )
        st.markdown("##### Property")
        a, b = st.columns(2)
        a.number_input(
            "Value ($)",
            min_value=10_000,
            max_value=10_000_000,
            step=5_000,
            key="f_property_value",
            help=hmda.FEATURE_GLOSSARY["property_value"],
        )
        b.number_input(
            "CLTV (%)",
            min_value=1.0,
            max_value=250.0,
            step=0.5,
            format="%.1f",
            key="f_combined_loan_to_value_ratio",
            help=hmda.FEATURE_GLOSSARY["combined_loan_to_value_ratio"],
        )
        st.selectbox(
            "Occupancy",
            list(hmda.OCCUPANCY_TYPE),
            format_func=hmda.OCCUPANCY_TYPE.get,
            key="f_occupancy_type",
        )
        st.radio(
            "Units",
            list(hmda.TOTAL_UNITS),
            format_func=hmda.TOTAL_UNITS.get,
            horizontal=True,
            key="f_total_units",
        )
        st.markdown("##### Applicant finances")
        st.number_input(
            "Annual income ($ thousands)",
            min_value=1.0,
            max_value=5_000.0,
            step=1.0,
            format="%.0f",
            key="f_income",
            help=hmda.FEATURE_GLOSSARY["income"],
        )
        st.select_slider(
            "Debt-to-income ratio",
            options=hmda.DTI_ORDER,
            format_func=lambda c: hmda.code_label("debt_to_income_ratio", c),
            key="f_debt_to_income_ratio",
            help=hmda.FEATURE_GLOSSARY["debt_to_income_ratio"],
        )
        st.radio(
            "Preapproval",
            list(hmda.PREAPPROVAL),
            format_func=lambda c: "Requested" if c == "1" else "Not requested",
            horizontal=True,
            key="f_preapproval",
        )
        submitted = st.form_submit_button(
            "Analyze scenario", type="primary", width="stretch", icon=":material/analytics:"
        )
    st.button(
        "Reset to preset",
        width="stretch",
        on_click=lambda: load_preset(st.session_state["preset"]),
        icon=":material/restart_alt:",
    )

inputs = {k: st.session_state[f"f_{k}"] for k in hmda.INPUT_FEATURES}
inputs["loan_term"] = int(inputs["loan_term"])

if submitted or "result" not in st.session_state:
    with main_col, st.spinner("Scoring with the saved pipeline…"):
        try:
            st.session_state["result"] = backend.predict(inputs)
            st.session_state["result_inputs"] = dict(inputs)
            st.session_state.pop("last_error", None)
            st.session_state.pop("compare", None)
        except BackendError as exc:
            st.session_state["last_error"] = exc

result = st.session_state.get("result")
stale = result is not None and st.session_state.get("result_inputs") != inputs

with main_col:
    if "last_error" in st.session_state:
        show_error(st.session_state["last_error"])
        if result:
            st.caption("Showing the previous valid result below.")
    if not result:
        st.stop()
    score_col, text_col = st.columns([1, 1], gap="medium")
    with score_col, st.container(border=True, key="card_score"):
        st.markdown('<div class="la-eyebrow">Illustrative estimate</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="la-score">{result["estimated_probability"] * 100:.1f}%</div>'
            '<div class="la-score-sub">estimated share of similar 2025 Texas applications that were '
            "denied</div>",
            unsafe_allow_html=True,
        )
        ui.meter(result["estimated_probability"], result["test_base_rate"], result["research_threshold"])
        st.markdown(
            ui.badge(f"Calibrated ({result['calibration_method']})", "accent")
            + " "
            + ui.badge(result["algorithm_name"])
            + " "
            + (ui.badge("Inputs edited since this result", "amber") if stale else ""),
            unsafe_allow_html=True,
        )
        st.write("")
        ui.kv(
            [
                ("Typical application", ui.pct(result["reference_probability"])),
                ("Test-set average", ui.pct(result["test_base_rate"])),
                ("Higher than", f"{result['percentile_among_test_applications']:.0f}% of test applications"),
                (
                    "Research threshold",
                    f"{ui.pct(result['research_threshold'])} "
                    f"({'above' if result['above_research_threshold'] else 'below'})",
                ),
            ]
        )
    with text_col:
        with st.container(border=True, key="card_interp"):
            st.markdown("##### In plain English")
            st.write(result["interpretation"])
            for w in result["warnings"]:
                st.warning(w, icon=":material/info:")
            st.page_link(
                "views/analyst.py", label="Ask the AI Analyst about this result", icon=":material/forum:"
            )
        with st.container(border=True, key="card_notmean"):
            st.markdown("##### What this does not mean")
            st.markdown(
                "- Not an approval, denial, rate or credit decision.\n"
                "- Not a prediction for a real person: HMDA has no credit scores or credit history.\n"
                "- Contributions are associations in past decisions, not causes or adverse-action reasons."
            )

    with st.container(border=True, key="card_contrib"):
        st.markdown("##### What moved the estimate")
        moved = [c for c in result["all_contributions"] if c["direction"] != "neutral"]
        diff = (result["estimated_probability"] - result["reference_probability"]) * 100
        st.caption(
            f"Shapley values over input groups versus a typical Texas application "
            f"({ui.pct(result['reference_probability'])}). Bars add up to {diff:+.1f} percentage points."
        )
        if moved:
            st.altair_chart(charts.contribution_chart(moved), width="stretch")
        else:
            st.info("This scenario matches the typical application on every input group.")
        with st.expander("Values compared for each group"):
            df = pd.DataFrame(result["all_contributions"])[
                ["label", "value_display", "reference_display", "contribution"]
            ]
            df["contribution"] = (df["contribution"] * 100).round(2)
            st.dataframe(
                df.rename(
                    columns={
                        "label": "Inputs",
                        "value_display": "This scenario",
                        "reference_display": "Typical application",
                        "contribution": "Points",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

    sens_col, cmp_col = st.columns([3, 2], gap="medium")
    with sens_col, st.container(border=True, key="card_sensitivity"):
        st.markdown("##### Model sensitivity")
        st.caption(
            "Change one input and hold the rest fixed. Shows how this model responds; it is not advice."
        )
        options = [
            "debt_to_income_ratio",
            "combined_loan_to_value_ratio",
            "income",
            "loan_purpose",
            "loan_type",
            "occupancy_type",
        ]
        feature = (
            st.segmented_control(
                "Input to vary",
                options,
                format_func=hmda.FEATURE_LABELS.get,
                default="debt_to_income_ratio",
                key="sens_feature",
                label_visibility="collapsed",
            )
            or "debt_to_income_ratio"
        )
        try:
            sens = backend.explain(st.session_state["result_inputs"], feature)
            st.altair_chart(
                charts.sensitivity_chart(sens["points"], sens["label"], feature in hmda.NUMERIC_INPUTS),
                width="stretch",
            )
            if feature in ("income", "combined_loan_to_value_ratio"):
                st.caption(
                    "Other inputs stay fixed, so some combinations on this curve rarely occur in real data."
                )
        except BackendError as exc:
            show_error(exc)
    with cmp_col, st.container(border=True, key="card_compare"):
        st.markdown("##### Compare models")
        st.caption("Score this scenario with every candidate model.")
        if st.button("Score with all three models", width="stretch", icon=":material/compare_arrows:"):
            try:
                st.session_state["compare"] = backend.compare(st.session_state["result_inputs"])
            except BackendError as exc:
                show_error(exc)
        if "compare" in st.session_state:
            df = pd.DataFrame(st.session_state["compare"]["results"])
            df = df[["algorithm_name", "estimated_probability", "test_pr_auc"]]
            st.dataframe(
                df,
                hide_index=True,
                width="stretch",
                column_config={
                    "algorithm_name": "Model",
                    "estimated_probability": st.column_config.ProgressColumn(
                        "Estimate", min_value=0, max_value=1, format="percent"
                    ),
                    "test_pr_auc": st.column_config.NumberColumn("Test PR-AUC", format="%.3f"),
                },
            )
            st.caption("Models with similar overall accuracy can still disagree on individual scenarios.")

    with st.expander("Technical details (JSON response)"):
        st.json({k: v for k, v in result.items() if k != "all_contributions"}, expanded=False)
        st.caption(f"Explanation method: {result['explanation_method']}")
        st.caption(
            "Explanation groups: "
            + "; ".join(f"{v[0]} = {', '.join(v[1])}" for v in EXPLANATION_GROUPS.values())
        )
    ui.footer(result["disclaimer"])
