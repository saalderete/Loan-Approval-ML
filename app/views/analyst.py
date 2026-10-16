import streamlit as st
from components import ui
from components.backend import BackendError, get_backend
from components.page import require_model, show_error

from loan_approval.ai.analyst import SUGGESTED_QUESTIONS
from loan_approval.config import get_settings

require_model()
backend = get_backend()
ai_on = get_settings().ai_enabled or bool(get_settings().api_url)

ui.page_title(
    "Explore · evidence-grounded assistant",
    "AI Model Analyst",
    "Ask about the model's measured results. The assistant reads saved metrics, model cards and audits through "
    "read-only tools, cites its evidence, and has its numbers checked. It never makes or predicts lending decisions.",
)

scenario = st.session_state.get("result")
left, right = st.columns([3, 2], gap="large")

with right:
    with st.container(border=True, key="card_guard"):
        st.markdown("##### Guardrails")
        st.markdown(
            "- Read-only tools: model card, metrics, importance, dataset card, versions, drift, fairness, demo scenarios.\n"
            "- Every number with a decimal or % is matched against tool evidence.\n"
            "- Requests for real decisions or personal advice are redirected before any LLM call.\n"
            "- Personal identifiers in a question are refused.\n"
            "- Text inside data is treated as data, not instructions."
        )
    with st.container(border=True, key="card_ctx"):
        st.markdown("##### Context")
        use_scenario = st.toggle(
            "Include my current Scenario Explorer result",
            value=scenario is not None,
            disabled=scenario is None,
        )
        if scenario is None:
            st.caption("Run a scenario first to ask about it.")
        else:
            st.caption(
                f"Estimate {ui.pct(scenario['estimated_probability'])} from {scenario['model_version']}."
            )

with left:
    if not ai_on:
        st.info(
            "The LLM analyst is off on this deployment (no `ANTHROPIC_API_KEY`). Quick answers below are "
            "built from the same evidence without an LLM.",
            icon=":material/info:",
        )
    with st.container(border=True, key="card_quick"):
        st.markdown("##### Quick answers from saved evidence")
        kind = st.pills(
            "Question",
            list(SUGGESTED_QUESTIONS),
            format_func=SUGGESTED_QUESTIONS.get,
            key="quick_kind",
            label_visibility="collapsed",
            default="compare",
        )
        if kind:
            try:
                qa = backend.quick_answer(kind)
                st.markdown(qa["answer"])
                st.markdown(
                    " ".join(ui.badge(e, "mono") for e in qa["evidence"])
                    + " "
                    + ui.badge("No LLM used", "accent"),
                    unsafe_allow_html=True,
                )
            except BackendError as exc:
                show_error(exc)

    if ai_on:
        history = st.session_state.setdefault("chat", [])
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("meta"):
                    st.markdown(msg["meta"], unsafe_allow_html=True)
        question = st.chat_input("Ask about metrics, calibration, fairness, features or limitations…")
        if question:
            history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"), st.spinner("Checking the evidence…"):
                try:
                    reply = backend.ask(question, scenario if (scenario and use_scenario) else None)
                except BackendError as exc:
                    reply = {"status": "error", "answer": str(exc), "evidence": []}
                st.markdown(reply["answer"])
                meta = ""
                if reply["status"] == "ok":
                    g = reply["grounding"]
                    meta = (
                        ui.badge("Grounded in evidence", "accent")
                        if g["grounded"]
                        else ui.badge(
                            "Some numbers not verified: " + ", ".join(g["unverified_numbers"]), "amber"
                        )
                    )
                    meta += " " + " ".join(
                        ui.badge(e["evidence_id"], "mono") for e in reply["evidence"] if e["cited"]
                    )
                    if reply.get("hypotheses"):
                        st.caption("Hypotheses (not measured): " + " · ".join(reply["hypotheses"]))
                elif reply["status"] == "refused":
                    meta = ui.badge("Out of scope", "amber")
                st.markdown(meta, unsafe_allow_html=True)
                history.append({"role": "assistant", "content": reply["answer"], "meta": meta})
