"""Train -> register -> promote -> reload -> predict, using the session's synthetic model."""

import numpy as np
import pytest

from loan_approval.features.schema import ScenarioInput, load_demo_scenarios
from loan_approval.inference import predict
from loan_approval.registry import artifacts


def test_artifacts_and_reports_exist(trained_env):
    vdir = trained_env["base"] / trained_env["version"]
    manifest = artifacts.load_manifest(trained_env["version"], trained_env["base"])
    for name in (
        "metrics",
        "curves",
        "importance",
        "fairness",
        "profile",
        "drift_baseline",
        "leakage_screen",
    ):
        assert (vdir / "reports" / f"{name}.json").exists()
    assert (vdir / "model_card.md").read_text().startswith("# Model card")
    assert manifest["dataset"]["split_hash"]
    assert set(manifest["reference_values"]) == set(manifest["input_features"])
    assert all(len(a["sha256"]) == 64 for a in manifest["artifacts"].values())


def test_prediction_is_reproducible_after_reload(trained_env):
    scenario = ScenarioInput(**load_demo_scenarios()[2]["inputs"])
    first = predict.predict(scenario)["estimated_probability"]
    predict.clear_cache()
    second = predict.predict(scenario)["estimated_probability"]
    assert first == second
    assert 0 <= first <= 1


def test_shapley_contributions_sum_to_difference(trained_env):
    for preset in load_demo_scenarios():
        result = predict.predict(ScenarioInput(**preset["inputs"]))
        total = sum(c["contribution"] for c in result["all_contributions"])
        diff = result["estimated_probability"] - result["reference_probability"]
        assert total == pytest.approx(diff, abs=2e-3)
        assert result["model_version"] == trained_env["version"]


def test_explanation_matches_loaded_version(trained_env):
    result = predict.predict(ScenarioInput(**load_demo_scenarios()[0]["inputs"]))
    loaded = predict.get_model()
    assert result["model_version"] == loaded.version
    assert result["algorithm"] == loaded.manifest["selected_algorithm"]


def test_test_split_metrics_match_saved_predictions(trained_env):
    loaded = predict.get_model()
    metrics = loaded.report("metrics")
    sel = metrics["selected_algorithm"]
    p = metrics["test"][sel]["probability"]
    assert 0.5 < p["roc_auc"] <= 1.0
    assert p["n"] == metrics["splits"][-1]["rows"]


def test_sensitivity_and_compare(trained_env):
    scenario = ScenarioInput(**load_demo_scenarios()[1]["inputs"])
    sens = predict.sensitivity(scenario, "debt_to_income_ratio")
    assert sum(pt["is_current"] for pt in sens["points"]) == 1
    cmp = predict.compare(scenario)
    assert {r["algorithm"] for r in cmp["results"]} == set(predict.get_model().pipelines)
    assert np.all([0 <= r["estimated_probability"] <= 1 for r in cmp["results"]])


def test_missing_model_gives_setup_state(tmp_path, monkeypatch):
    monkeypatch.setenv("LOAN_ML_MODELS_DIR", str(tmp_path))
    predict.clear_cache()
    with pytest.raises(predict.ModelUnavailable):
        predict.get_model()
    predict.clear_cache()
