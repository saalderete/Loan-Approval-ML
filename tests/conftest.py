"""Shared fixtures. Tests train a small synthetic model once per session in a temporary directory,
so they never need the real HMDA file and never touch the committed models/ directory."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def fast_config() -> dict:
    from loan_approval.training.train import load_config

    cfg = load_config()
    cfg["tuning"]["max_rows"] = 2500
    cfg["tuning"]["cv_folds"] = 2
    cfg["bootstrap_resamples"] = 20
    cfg["importance_rows"] = 800
    cfg["candidates"]["random_forest"]["fixed"]["n_estimators"] = 20
    cfg["candidates"]["random_forest"]["grid"] = {"max_depth": [8]}
    cfg["candidates"]["hist_gradient_boosting"]["grid"] = {"learning_rate": [0.1], "max_leaf_nodes": [15]}
    cfg["candidates"]["logistic_regression"]["grid"] = {"C": [1.0]}
    return cfg


@pytest.fixture(scope="session")
def synthetic_frame():
    from loan_approval.data.synthetic import generate_raw
    from loan_approval.data.validate import build_model_frame

    frame, report = build_model_frame(generate_raw(6000, seed=3))
    return frame, report


@pytest.fixture(scope="session")
def trained_env(tmp_path_factory, synthetic_frame):
    """Train + promote a synthetic model in a temp models dir and point settings at it."""
    from loan_approval.inference import predict
    from loan_approval.registry import artifacts
    from loan_approval.registry.artifacts import sha256_bytes
    from loan_approval.training.train import DatasetInfo, train

    base = tmp_path_factory.mktemp("models")
    state = tmp_path_factory.mktemp("state")
    old = {
        k: os.environ.get(k)
        for k in ("LOAN_ML_MODELS_DIR", "LOAN_ML_STATE_DIR", "ADMIN_TOKEN", "ANTHROPIC_API_KEY", "API_URL")
    }
    os.environ["LOAN_ML_MODELS_DIR"] = str(base)
    os.environ["LOAN_ML_STATE_DIR"] = str(state)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("API_URL", None)
    frame, report = synthetic_frame
    info = DatasetInfo(
        name="Synthetic test data",
        synthetic=True,
        source="tests",
        raw_sha256=sha256_bytes(b"tests"),
        quality=report.to_dict(),
        outcome_definition="Synthetic outcome for tests.",
    )
    version = train(frame, info, fast_config(), out_base=base, log=lambda *_: None)
    manifest = artifacts.load_manifest(version, base)
    # Small synthetic data can miss strict gates (e.g. calibration); force them for test promotion only.
    if not all(g["passed"] for g in manifest["promotion_gates"]):
        import json

        for g in manifest["promotion_gates"]:
            g["passed"] = True
        (base / version / "manifest.json").write_text(json.dumps(manifest, indent=2))
        artifacts.register_version(version, manifest, "review_required", base)
    artifacts.promote(version, actor="pytest", reason="test fixture", base=base)
    predict.clear_cache()
    yield {"version": version, "base": base, "state": state, "frame": frame}
    predict.clear_cache()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def demo_inputs():
    from loan_approval.features.schema import load_demo_scenarios

    return load_demo_scenarios()[0]["inputs"]
