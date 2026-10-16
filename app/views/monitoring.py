import pandas as pd
import streamlit as st
from components import charts, ui
from components.backend import get_backend

backend = get_backend()
report = backend.monitoring()
ops = report["operations"]
drift = report["drift"]

ui.page_title(
    "Operations · live demo traffic",
    "Monitoring",
    "Operational health and input drift for scenarios scored by this deployment. Demo traffic has no real "
    "outcomes, so this page never claims live accuracy, only whether inputs look different from training.",
)
if st.button("Refresh", icon=":material/refresh:"):
    st.rerun()

cols = st.columns(5)
cards = [
    ("Requests (7 days)", f"{ops['requests']:,}", "predict, explain, compare"),
    ("Invalid input rate", ui.pct(ops["invalid_rate"]), "rejected by schema validation"),
    ("Error rate", ui.pct(ops["error_rate"]), "unexpected backend errors"),
    (
        "Latency p50",
        "–" if ops["latency_ms_p50"] is None else f"{ops['latency_ms_p50']:.0f} ms",
        "successful calls",
    ),
    (
        "Latency p95",
        "–" if ops["latency_ms_p95"] is None else f"{ops['latency_ms_p95']:.0f} ms",
        "successful calls",
    ),
]
for i, (label, value, sub) in enumerate(cards):
    with cols[i], st.container(border=True, key=f"card_ops_{i}"):
        ui.stat(label, value, sub)

if ops["requests"] == 0:
    ui.empty_state(
        "No traffic yet",
        "Score a scenario in the Scenario Explorer and come back. Only aggregate "
        "counts are stored; scenario inputs are never saved.",
    )
else:
    with st.container(border=True, key="card_hourly"):
        st.markdown("##### Requests per hour")
        st.altair_chart(charts.hourly_chart(ops["hourly"]), width="stretch")

with st.container(border=True, key="card_drift"):
    st.markdown("##### Input drift (Population Stability Index)")
    if drift is None:
        st.caption("No active model.")
    elif drift["observations"] == 0:
        st.caption("No evidence yet: no scenarios have been scored against the active model.")
    else:
        df = pd.DataFrame(drift["features"])[["label", "psi", "n", "status"]]
        st.dataframe(
            df,
            hide_index=True,
            width="stretch",
            column_config={
                "label": "Input",
                "psi": st.column_config.NumberColumn("PSI", format="%.3f"),
                "n": st.column_config.NumberColumn("Scored scenarios", format="localized"),
                "status": "Status",
            },
        )
        st.caption(
            f"PSI ≥ {drift['thresholds']['warn']} = warn, ≥ {drift['thresholds']['alert']} = alert. Needs at "
            f"least {drift['min_samples']} scenarios per input. {drift['note']}"
        )

with st.container(border=True, key="card_ai_usage"):
    st.markdown("##### AI Analyst usage")
    usage = pd.DataFrame(report["ai_usage"]["rows"])
    if usage.empty:
        st.caption("No AI requests recorded.")
    else:
        st.dataframe(usage, hide_index=True, width="stretch")
ui.footer(
    "The monitoring database stores request counts, status, latency and binned input counts per model "
    "version. It does not store scenario values, IP addresses or AI prompts."
)
