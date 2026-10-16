"""Loan Approval ML dashboard. Run with: ``streamlit run app/streamlit_app.py``."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for p in (APP_DIR, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402
from components import ui  # noqa: E402
from components.backend import get_backend  # noqa: E402

st.set_page_config(
    page_title="Loan Approval ML · Mortgage decision research lab",
    page_icon=str(APP_DIR / "assets" / "icon.svg"),
    layout="wide",
    initial_sidebar_state="auto",
)
ui.inject_css()
st.logo(str(APP_DIR / "assets" / "logo.svg"), size="large", icon_image=str(APP_DIR / "assets" / "icon.svg"))

V = APP_DIR / "views"
pages = {
    "Explore": [
        st.Page(V / "overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
        st.Page(V / "scenario.py", title="Scenario Explorer", icon=":material/tune:", url_path="scenario"),
        st.Page(V / "analyst.py", title="AI Model Analyst", icon=":material/forum:", url_path="analyst"),
    ],
    "Model": [
        st.Page(
            V / "performance.py",
            title="Model Performance",
            icon=":material/monitoring:",
            url_path="performance",
        ),
        st.Page(
            V / "explainability.py",
            title="Explainability",
            icon=":material/insights:",
            url_path="explainability",
        ),
        st.Page(
            V / "responsible.py",
            title="Fairness & Responsible AI",
            icon=":material/balance:",
            url_path="responsible-ai",
        ),
    ],
    "Data": [
        st.Page(
            V / "dataset.py", title="Dataset Explorer", icon=":material/table_chart:", url_path="dataset"
        ),
    ],
    "Operations": [
        st.Page(
            V / "experiments.py", title="Experiment Lab", icon=":material/science:", url_path="experiments"
        ),
        st.Page(
            V / "monitoring.py", title="Monitoring", icon=":material/vital_signs:", url_path="monitoring"
        ),
        st.Page(V / "docs.py", title="API & Docs", icon=":material/code:", url_path="docs"),
    ],
}
nav = st.navigation(pages)

backend = get_backend()
status = backend.status()
with st.sidebar:
    if status.get("ready"):
        st.markdown(
            ui.badge("Model ready", "accent") + " " + ui.badge(status["active_version"], "mono"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(ui.badge("No model loaded", "amber"), unsafe_allow_html=True)
    st.markdown(
        '<p class="la-side-note">Educational research on public 2025 Texas HMDA data. Not a lending '
        "decision tool. Scenarios are fictional and are never stored.</p>"
        f'<p class="la-side-note">Backend: {ui.esc(backend.mode)}</p>',
        unsafe_allow_html=True,
    )

nav.run()
