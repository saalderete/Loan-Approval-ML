import json

import streamlit as st
from components import ui
from components.backend import get_backend

from loan_approval import __version__
from loan_approval.config import get_settings

backend = get_backend()
schema = backend.schema()
presets = backend.demo_scenarios()
api = get_settings().api_url or "http://localhost:8000"

ui.page_title(
    "Operations · developers",
    "API & Docs",
    "A typed, versioned FastAPI service exposes the same results as this dashboard. Interactive OpenAPI docs "
    "live at /docs on the API.",
)
st.markdown(
    ui.badge(f"package v{__version__}")
    + " "
    + ui.badge(schema["schema_version"], "mono")
    + " "
    + ui.badge(f"API base {api}", "mono"),
    unsafe_allow_html=True,
)
st.write("")

tab_quick, tab_api, tab_arch = st.tabs(["Quick start", "Endpoints", "Architecture"])
with tab_quick, st.container(border=True, key="card_qs"):
    st.markdown("##### Run everything locally")
    st.code(
        "git clone https://github.com/saalderete/Loan-Approval-ML.git && cd Loan-Approval-ML\n"
        "python -m venv .venv && source .venv/bin/activate\n"
        "pip install -r requirements.txt && pip install -e .\n\n"
        "streamlit run app/streamlit_app.py           # dashboard (uses the committed model)\n"
        "uvicorn loan_approval.api.app:app --port 8000 # API\n"
        "docker compose up --build                    # both, API-backed\n\n"
        "# Reproduce the model from public data\n"
        "loan-approval fetch && loan-approval prepare && loan-approval train",
        language="bash",
    )

with tab_api:
    endpoints = [
        ("GET", "/health", "Liveness", "public"),
        ("GET", "/ready", "Approved model loaded and verified", "public"),
        ("GET", "/api/v1/model", "Model card summary, schema, gates, provenance", "public"),
        ("GET", "/api/v1/metrics", "Saved evaluation report", "public"),
        (
            "GET",
            "/api/v1/reports/{name}",
            "curves, importance, fairness, profile, leakage_screen, drift_baseline",
            "public",
        ),
        ("GET", "/api/v1/demo-scenarios", "Fictional presets", "public"),
        ("GET", "/api/v1/schema", "Input contract", "public"),
        ("POST", "/api/v1/predict", "Estimate + Shapley explanation", "public, rate-limited"),
        ("POST", "/api/v1/explain", "One-input sensitivity curve", "public, rate-limited"),
        ("POST", "/api/v1/compare", "All candidate models on one scenario", "public, rate-limited"),
        ("POST", "/api/v1/ai/ask", "Evidence-grounded analyst", "rate-limited (10/min)"),
        ("GET", "/api/v1/experiments", "Registry versions and audit log", "public read-only"),
        ("GET", "/api/v1/monitoring", "Ops stats and input drift", "public read-only"),
        ("POST", "/api/v1/train-jobs", "Start a bounded training job (idempotent)", "admin token"),
        ("POST", "/api/v1/models/{version}/promote", "Promote a gated version", "admin token"),
    ]
    with st.container(border=True, key="card_endpoints"):
        st.dataframe(
            [{"Method": m, "Path": p, "Purpose": d, "Access": a} for m, p, d, a in endpoints],
            hide_index=True,
            width="stretch",
        )
    body = {"scenario": presets[0]["inputs"]}
    with st.container(border=True, key="card_curl"):
        st.markdown("##### Example request")
        st.code(
            f"curl -s -X POST {api}/api/v1/predict \\\n  -H 'Content-Type: application/json' \\\n"
            f"  -d '{json.dumps(body)}'",
            language="bash",
        )
        if st.button("Run this request against the model", icon=":material/play_arrow:"):
            result = backend.predict(body["scenario"])
            result.pop("all_contributions", None)
            st.json(result)
    with st.expander("Input schema"):
        st.json(schema, expanded=False)

with tab_arch, st.container(border=True, key="card_arch"):
    st.graphviz_chart(
        """
digraph G {
  rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#EAF4F4",
  color="#0F8B8D", fontname="Inter", fontsize=11]; edge [color="#8A9BB0"];
  hmda [label="FFIEC HMDA\\nData Browser"]; ingest [label="fetch + checksum\\n(data_manifest.yaml)"];
  validate [label="validate + filter\\nquality report"]; split [label="stratified split\\n(hash recorded)"];
  train [label="CV tuning → fit\\nLR · RF · HistGB"]; cal [label="isotonic\\ncalibration"];
  eval [label="select → test once\\nmetrics · CIs · fairness"]; reg [label="registry\\nmanifest + SHA-256 gates"];
  api [label="FastAPI\\n/api/v1"]; ui [label="Streamlit\\ndashboard"]; ai [label="AI analyst\\nread-only tools"];
  mon [label="SQLite\\nops + drift counts"];
  hmda -> ingest -> validate -> split -> train -> cal -> eval -> reg;
  reg -> api; reg -> ui; api -> ui [style=dashed, label="API_URL"]; ai -> reg [style=dashed];
  api -> mon; ui -> mon; ui -> ai;
}
""",
        width="stretch",
    )
    st.caption(
        "Training never runs inside a web request. The API and dashboard only load hash-verified artifacts."
    )
