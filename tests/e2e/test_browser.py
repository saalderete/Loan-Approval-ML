"""Opt-in browser test against a running dashboard: E2E_BASE_URL=http://localhost:8501 pytest -m e2e"""

import os

import pytest

pytestmark = pytest.mark.e2e
BASE = os.environ.get("E2E_BASE_URL")


@pytest.mark.skipif(not BASE, reason="set E2E_BASE_URL to run browser tests")
def test_scenario_explorer_in_browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.goto(f"{BASE}/scenario", wait_until="networkidle")
        page.get_by_text("Illustrative estimate").wait_for(timeout=30000)
        page.get_by_role("button", name="Analyze scenario").click()
        page.get_by_text("What moved the estimate").wait_for(timeout=30000)
        assert page.locator('[data-testid="stException"]').count() == 0
        page.goto(f"{BASE}/responsible-ai", wait_until="networkidle")
        page.get_by_text("Observed denial rate vs model estimate").wait_for(timeout=30000)
        browser.close()
