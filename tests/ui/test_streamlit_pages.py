"""Streamlit smoke tests: every page renders against the synthetic model without exceptions."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
PAGES = [
    "views/overview.py",
    "views/scenario.py",
    "views/analyst.py",
    "views/performance.py",
    "views/explainability.py",
    "views/responsible.py",
    "views/dataset.py",
    "views/experiments.py",
    "views/monitoring.py",
    "views/docs.py",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(trained_env, page):
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.switch_page(page)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_scenario_flow_and_invalid_input(trained_env):
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.switch_page("views/scenario.py")
    at.run()
    assert "result" in at.session_state
    first = at.session_state["result"]["estimated_probability"]
    at.selectbox(key="preset").select("high-dti-cashout").run()
    at.button(key="FormSubmitter:scenario_form-Analyze scenario").click().run()
    assert at.session_state["result"]["estimated_probability"] != first
    # Invalid combination keeps the previous result and shows an error
    at.number_input(key="f_combined_loan_to_value_ratio").set_value(5.0)
    at.button(key="FormSubmitter:scenario_form-Analyze scenario").click().run()
    assert at.error, "expected a validation error banner"
    assert at.session_state["result"] is not None


def test_setup_state_without_model(tmp_path, monkeypatch):
    from loan_approval.inference import predict

    monkeypatch.setenv("LOAN_ML_MODELS_DIR", str(tmp_path))
    predict.clear_cache()
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("No approved model" in m.value for m in at.markdown)
    predict.clear_cache()
