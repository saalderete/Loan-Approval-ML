"""Shared page helpers: model guard, error display and scenario state."""

from __future__ import annotations

import streamlit as st

from components import ui
from components.backend import BackendError, get_backend


def require_model() -> dict:
    status = get_backend().status()
    if not status.get("ready"):
        ui.empty_state(
            "No approved model yet",
            status.get("error")
            or "Train a model, review its promotion gates, then promote it. No metrics are shown until "
            "a real evaluated model exists.",
            "loan-approval fetch && loan-approval prepare && loan-approval train\n"
            'loan-approval list\nloan-approval promote <version> --actor "your name"',
        )
        st.stop()
    return status


def show_error(exc: BackendError) -> None:
    if exc.kind == "invalid":
        lines = "\n".join(f"- **{d.get('field') or 'scenario'}**: {d.get('message')}" for d in exc.details)
        st.error(f"{exc}\n\n{lines}", icon=":material/error:")
    elif exc.kind == "setup":
        st.warning(f"Model unavailable: {exc}", icon=":material/inventory_2:")
    elif exc.kind == "rate_limit":
        st.warning(str(exc), icon=":material/hourglass_top:")
    else:
        st.error(f"Backend problem: {exc}", icon=":material/cloud_off:")


def algo_names(info: dict) -> dict:
    return info["algorithms"]
