"""Inference: load the approved model version once, validate inputs, score, explain.

Nothing is fitted here. Importing this module never trains or loads anything until it is asked to.
"""

from __future__ import annotations

import threading
import uuid
from bisect import bisect_left

import pandas as pd

from loan_approval.config import DISCLAIMER
from loan_approval.data import hmda
from loan_approval.features.schema import SCHEMA_VERSION, ScenarioInput
from loan_approval.inference import explain
from loan_approval.registry import artifacts


class ModelUnavailable(RuntimeError):
    """No approved model version is available (setup state)."""


_cache: dict[tuple, artifacts.LoadedVersion] = {}
_cache_lock = threading.Lock()


def get_model(version: str | None = None) -> artifacts.LoadedVersion:
    base = artifacts.models_dir()
    registry = artifacts.read_registry(base)
    version = version or registry.get("active_version")
    if not version:
        raise ModelUnavailable(
            "No model has been promoted yet. Train one with `loan-approval train` and promote it "
            "with `loan-approval promote <version>`."
        )
    entry = registry["versions"].get(version)
    if entry is None:
        raise ModelUnavailable(f"Model version {version!r} is not registered")
    if entry["status"] == "retired":
        raise ModelUnavailable(f"Model version {version!r} is retired")
    key = (str(base), version, entry["manifest_sha256"])
    with _cache_lock:
        if key not in _cache:
            try:
                _cache[key] = artifacts.load_version(version, base)
            except artifacts.ArtifactError as exc:
                raise ModelUnavailable(str(exc)) from exc
        return _cache[key]


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _percentile(quantiles: list[float], p: float) -> float:
    return round(100 * bisect_left(quantiles, p) / (len(quantiles) - 1), 1)


def _interpretation(
    probability: float, base_rate: float, percentile: float, top: list[dict], reference_p: float
) -> str:
    relative = probability / base_rate if base_rate else 0
    if relative >= 1.5:
        level = "higher than"
    elif relative <= 0.67:
        level = "lower than"
    else:
        level = "close to"
    up = [c["label"].lower() for c in top if c["direction"] == "up"][:2]
    down = [c["label"].lower() for c in top if c["direction"] == "down"][:2]
    parts = [
        f"The trained model estimates that {probability:.1%} of historical Texas applications like "
        f"this one were denied, {level} the {base_rate:.1%} average in the held-out data "
        f"(higher than {percentile:.0f}% of test applications)."
    ]
    parts.append(f"A typical application (median amounts, most common categories) gets {reference_p:.1%}.")
    if up:
        parts.append(
            f"Compared with that typical application, the model associates the "
            f"{' and '.join(up)} with a higher estimate."
        )
    if down:
        parts.append(f"It associates the {' and '.join(down)} with a lower estimate.")
    parts.append("These are patterns in past decisions, not causes, and not a prediction for a real person.")
    return " ".join(parts)


def predict(scenario: ScenarioInput, algorithm: str | None = None, version: str | None = None) -> dict:
    loaded = get_model(version)
    manifest = loaded.manifest
    algorithm = algorithm or manifest["selected_algorithm"]
    if algorithm not in loaded.pipelines:
        raise ModelUnavailable(f"Algorithm {algorithm!r} is not available in {loaded.version}")
    model = loaded.pipelines[algorithm]
    frame = scenario.to_frame()
    probability = float(model.predict_proba(frame)[0, 1])
    threshold = float(manifest["thresholds"][algorithm]["value"])
    metrics_report = loaded.report("metrics")
    base_rate = metrics_report["test"][algorithm]["probability"]["prevalence"]
    quantiles = loaded.report("test_quantiles").get(algorithm) or []
    percentile = _percentile(quantiles, probability) if quantiles else None
    reference = manifest["reference_values"]
    contributions = explain.local_contributions(model, frame, reference)
    top = [c for c in contributions if c["direction"] != "neutral"][:5]
    reference_p = explain.reference_probability(model, reference)
    return {
        "prediction_id": str(uuid.uuid4()),
        "model_version": loaded.version,
        "algorithm": algorithm,
        "algorithm_name": manifest["algorithms"][algorithm],
        "schema_version": SCHEMA_VERSION,
        "scenario_type": "synthetic",
        "outcome_definition": manifest["outcome_definition"],
        "estimated_probability": round(probability, 4),
        "calibrated": True,
        "calibration_method": manifest["calibration"]["method"],
        "test_base_rate": round(float(base_rate), 4),
        "percentile_among_test_applications": percentile,
        "research_threshold": round(threshold, 4),
        "above_research_threshold": probability >= threshold,
        "explanation_status": "available",
        "explanation_method": explain.LOCAL_METHOD,
        "reference_probability": round(reference_p, 4),
        "reference_profile": {f: explain.display_value(f, reference[f]) for f in hmda.INPUT_FEATURES},
        "top_contributors": top,
        "all_contributions": contributions,
        "warnings": scenario.warnings(),
        "interpretation": _interpretation(probability, base_rate, percentile or 0, top, reference_p),
        "disclaimer": DISCLAIMER,
    }


def compare(scenario: ScenarioInput, version: str | None = None) -> dict:
    loaded = get_model(version)
    frame = scenario.to_frame()
    rows = []
    for algorithm, model in loaded.pipelines.items():
        test = loaded.report("metrics")["test"][algorithm]
        rows.append(
            {
                "algorithm": algorithm,
                "algorithm_name": loaded.manifest["algorithms"][algorithm],
                "selected": algorithm == loaded.manifest["selected_algorithm"],
                "estimated_probability": round(float(model.predict_proba(frame)[0, 1]), 4),
                "test_pr_auc": test["probability"]["pr_auc"],
                "test_roc_auc": test["probability"]["roc_auc"],
            }
        )
    return {"model_version": loaded.version, "results": rows, "disclaimer": DISCLAIMER}


def sensitivity(scenario: ScenarioInput, feature: str, algorithm: str | None = None) -> dict:
    if feature not in hmda.INPUT_FEATURES:
        raise ValueError(f"Unknown feature {feature!r}")
    loaded = get_model()
    algorithm = algorithm or loaded.manifest["selected_algorithm"]
    model = loaded.pipelines[algorithm]
    return {
        "model_version": loaded.version,
        "algorithm": algorithm,
        "feature": feature,
        "label": hmda.FEATURE_LABELS[feature],
        "points": explain.sensitivity(model, scenario.to_frame(), feature),
        "note": "Model sensitivity only: how this model's estimate moves when one input changes and "
        "everything else is held fixed. It is not advice and not a guarantee of any real outcome.",
    }


def batch_predict(frame: pd.DataFrame, version: str | None = None) -> pd.Series:
    loaded = get_model(version)
    return pd.Series(loaded.selected.predict_proba(frame[hmda.INPUT_FEATURES])[:, 1], index=frame.index)
