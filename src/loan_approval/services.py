"""Application services shared by the FastAPI routes and the Streamlit dashboard.

Every function returns plain JSON-serialisable data built from saved artifacts, so the API and the
dashboard can never disagree about a number.
"""

from __future__ import annotations

import time

from pydantic import ValidationError

from loan_approval import monitoring
from loan_approval.config import DISCLAIMER, ROOT
from loan_approval.data import hmda
from loan_approval.features.schema import ScenarioInput, load_demo_scenarios, schema_summary
from loan_approval.inference import predict as inference
from loan_approval.inference.predict import ModelUnavailable
from loan_approval.registry import artifacts

DATASET_CARD = ROOT / "reports" / "dataset_card.md"

__all__ = ["ModelUnavailable", "ValidationError"]


def status() -> dict:
    registry = artifacts.read_registry()
    active = registry.get("active_version")
    info = {
        "active_version": active,
        "versions": len(registry["versions"]),
        "ready": False,
        "disclaimer": DISCLAIMER,
    }
    if active:
        try:
            loaded = inference.get_model()
            m = loaded.manifest
            info.update(
                ready=True,
                selected_algorithm=m["selected_algorithm"],
                algorithm_name=m["algorithms"][m["selected_algorithm"]],
                dataset=m["dataset"],
                headline_metrics=m["headline_metrics"],
                created_at=m["created_at"],
                outcome_definition=m["outcome_definition"],
                available_algorithms=list(loaded.pipelines),
            )
        except ModelUnavailable as exc:
            info["error"] = str(exc)
    return info


def model_info() -> dict:
    loaded = inference.get_model()
    m = loaded.manifest
    keep = [
        "version",
        "created_at",
        "selected_algorithm",
        "algorithms",
        "outcome_definition",
        "positive_class",
        "schema_version",
        "input_schema",
        "excluded_columns",
        "dataset",
        "calibration",
        "thresholds",
        "explanation",
        "headline_metrics",
        "promotion_gates",
        "provenance",
        "training_config",
    ]
    return {k: m[k] for k in keep} | {
        "artifacts": {k: {"size_mb": v["size_mb"], "sha256": v["sha256"]} for k, v in m["artifacts"].items()}
    }


def report(name: str) -> dict:
    allowed = {"metrics", "curves", "importance", "fairness", "profile", "leakage_screen", "drift_baseline"}
    if name not in allowed:
        raise KeyError(name)
    return inference.get_model().report(name)


def model_card() -> str:
    loaded = inference.get_model()
    path = loaded.base / loaded.version / "model_card.md"
    return path.read_text() if path.exists() else ""


def dataset_card() -> str:
    return DATASET_CARD.read_text() if DATASET_CARD.exists() else ""


def demo_scenarios() -> list[dict]:
    return load_demo_scenarios()


def input_schema() -> dict:
    return schema_summary()


def experiments() -> dict:
    registry = artifacts.read_registry()
    rows = []
    for version, entry in sorted(registry["versions"].items(), reverse=True):
        row = {"version": version, **entry}
        try:
            manifest = artifacts.load_manifest(version)
            row["training_seconds"] = manifest["provenance"]["training_seconds"]
            row["git_commit"] = manifest["provenance"]["git_commit"]
            row["synthetic"] = manifest["dataset"]["synthetic"]
            row["rows"] = manifest["dataset"]["rows"]
            row["gates"] = manifest["promotion_gates"]
        except artifacts.ArtifactError:
            row["missing_manifest"] = True
        rows.append(row)
    return {
        "active_version": registry.get("active_version"),
        "versions": rows,
        "audit_log": registry.get("audit_log", [])[-30:][::-1],
        "lifecycle": artifacts.LIFECYCLE,
    }


def _baseline(loaded) -> dict:
    return loaded.report("drift_baseline")


def _timed(endpoint: str, source: str, fn, inputs: dict | None):
    start = time.perf_counter()
    try:
        scenario = ScenarioInput(**(inputs or {}))
    except ValidationError:
        monitoring.record_inference(endpoint, "invalid", None, source)
        raise
    try:
        result = fn(scenario)
    except ModelUnavailable:
        monitoring.record_inference(endpoint, "unavailable", None, source)
        raise
    except Exception:
        monitoring.record_inference(endpoint, "error", None, source)
        raise
    latency = (time.perf_counter() - start) * 1000
    loaded = inference.get_model()
    monitoring.record_inference(
        endpoint,
        "ok",
        latency,
        source,
        loaded.version,
        inputs=scenario.model_dump() if endpoint == "predict" else None,
        baseline=_baseline(loaded) if endpoint == "predict" else None,
    )
    return result


def predict(inputs: dict, algorithm: str | None = None, source: str = "api") -> dict:
    return _timed("predict", source, lambda s: inference.predict(s, algorithm=algorithm), inputs)


def explain(inputs: dict, feature: str, algorithm: str | None = None, source: str = "api") -> dict:
    if feature not in hmda.INPUT_FEATURES:
        raise KeyError(feature)
    return _timed("explain", source, lambda s: inference.sensitivity(s, feature, algorithm), inputs)


def compare(inputs: dict, source: str = "api") -> dict:
    return _timed("compare", source, inference.compare, inputs)


def monitoring_report() -> dict:
    stats = monitoring.operational_stats()
    try:
        loaded = inference.get_model()
        drift_report = monitoring.drift_for(loaded.version, _baseline(loaded))
        version = loaded.version
    except ModelUnavailable:
        drift_report, version = None, None
    return {
        "model_version": version,
        "operations": stats,
        "drift": drift_report,
        "ai_usage": monitoring.ai_usage_summary(),
    }
