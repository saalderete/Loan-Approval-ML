"""Allow-listed, read-only evidence tools for the AI Model Analyst.

Each tool returns ``{"evidence_id", "title", "data"}`` built from saved artifacts. The assistant can
only cite evidence that a tool actually returned during the conversation.
"""

from __future__ import annotations

from collections.abc import Callable

from loan_approval import services
from loan_approval.data import hmda
from loan_approval.features.schema import ScenarioInput
from loan_approval.inference.explain import EXPLANATION_GROUPS


def _round(obj, digits: int = 4):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, digits) for v in obj]
    return obj


def get_model_card() -> dict:
    info = services.model_info()
    return {
        "evidence_id": f"model_card:{info['version']}",
        "title": f"Model card {info['version']}",
        "data": _round(
            {
                "version": info["version"],
                "selected_model": info["algorithms"][info["selected_algorithm"]],
                "outcome_definition": info["outcome_definition"],
                "dataset": info["dataset"],
                "calibration": info["calibration"],
                "headline_test_metrics": info["headline_metrics"],
                "inputs": [f["label"] for f in info["input_schema"]["fields"]],
                "excluded_column_groups": {k: len(v) for k, v in info["excluded_columns"].items()},
                "promotion_gates": [
                    {"name": g["name"], "passed": g["passed"]} for g in info["promotion_gates"]
                ],
                "limitations": [
                    "No credit scores, credit history, assets or employment in HMDA.",
                    "Outcome is historical denial vs approval, not default risk.",
                    "Texas 2025, complete applications, site-built 1-4 unit homes only.",
                    "Explanations describe model behaviour, not causes or adverse-action reasons.",
                ],
            }
        ),
    }


def get_run_metrics(algorithm: str | None = None) -> dict:
    m = services.report("metrics")
    algos = [algorithm] if algorithm else list(m["test"])
    data = {
        "primary_metric": m["primary_metric"],
        "selected_algorithm": m["selected_algorithm"],
        "test_split": {
            a: {
                "name": m["test"][a]["name"],
                "probability_metrics": m["test"][a]["probability"],
                "confidence_intervals_95": {
                    k: v for k, v in m["test"][a]["ci"].items() if isinstance(v, dict)
                },
                "at_research_threshold": {
                    k: v for k, v in m["test"][a]["at_threshold"].items() if k != "confusion_matrix"
                },
                "confusion_matrix": m["test"][a]["at_threshold"]["confusion_matrix"],
            }
            for a in algos
            if a in m["test"]
        },
        "prevalence_only_baseline": m["dummy_baseline"],
        "cross_validation": {
            a: {"mean": m["cv"][a]["mean"], "std": m["cv"][a]["std"]} for a in algos if a in m["cv"]
        },
    }
    return {
        "evidence_id": f"metrics:{algorithm or 'all'}:test",
        "title": "Test-split metrics" + (f" for {algorithm}" if algorithm else " for all candidates"),
        "data": _round(data),
    }


def get_feature_attributions(algorithm: str | None = None) -> dict:
    imp = services.report("importance")
    algorithm = algorithm or services.report("metrics")["selected_algorithm"]
    perm = imp["permutation"].get(algorithm, {})
    return {
        "evidence_id": f"importance:{algorithm}",
        "title": f"Permutation importance ({algorithm})",
        "data": _round({"method": perm.get("method"), "features": perm.get("features", [])}),
    }


def get_dataset_card() -> dict:
    profile = services.report("profile")
    info = services.model_info()
    return {
        "evidence_id": "dataset_card",
        "title": "Dataset card and quality report",
        "data": _round(
            {
                "name": info["dataset"]["name"],
                "source": info["dataset"]["source"],
                "rows": profile["rows"],
                "denial_rate": profile["denial_rate"],
                "funnel": profile["quality"]["funnel"],
                "missingness": profile["quality"]["missingness"],
                "warnings": profile["quality"]["warnings"],
                "splits": profile["splits"],
            }
        ),
    }


def compare_model_versions() -> dict:
    exp = services.experiments()
    return {
        "evidence_id": "registry:versions",
        "title": "Registered model versions",
        "data": _round(
            {
                "active_version": exp["active_version"],
                "versions": [
                    {
                        k: v
                        for k, v in row.items()
                        if k
                        in (
                            "version",
                            "status",
                            "selected_algorithm",
                            "headline_metrics",
                            "synthetic",
                            "rows",
                        )
                    }
                    for row in exp["versions"]
                ],
            }
        ),
    }


def get_drift_report() -> dict:
    rep = services.monitoring_report()
    return {
        "evidence_id": f"drift:{rep['model_version']}",
        "title": "Input drift and operational stats",
        "data": _round(
            {
                "drift": rep["drift"],
                "operations": {k: v for k, v in rep["operations"].items() if k != "hourly"},
            }
        ),
    }


def get_fairness_summary(attribute: str) -> dict:
    if attribute not in hmda.PROTECTED_AUDIT_COLUMNS:
        raise ValueError(f"attribute must be one of {list(hmda.PROTECTED_AUDIT_COLUMNS)}")
    rep = services.report("fairness")
    model_table = next(t for t in rep["model_audit"] if t["attribute"] == attribute)
    hist_table = next(t for t in rep["historical_disparities"] if t["attribute"] == attribute)
    return {
        "evidence_id": f"fairness:{attribute}",
        "title": f"Fairness audit: {hmda.PROTECTED_AUDIT_COLUMNS[attribute]}",
        "data": _round(
            {"model_audit_test_split": model_table, "historical_all_rows": hist_table, "notes": rep["notes"]}
        ),
    }


def get_demo_scenario(scenario_id: str) -> dict:
    presets = {s["id"]: s for s in services.demo_scenarios()}
    if scenario_id not in presets:
        raise ValueError(f"Unknown scenario; choose from {sorted(presets)}")
    preset = presets[scenario_id]
    result = services.predict(ScenarioInput(**preset["inputs"]).model_dump(), source="ai")
    return {
        "evidence_id": f"scenario:{scenario_id}",
        "title": f"Demo scenario: {preset['label']}",
        "data": _round(
            {
                "label": preset["label"],
                "inputs": preset["inputs"],
                "estimated_probability": result["estimated_probability"],
                "test_base_rate": result["test_base_rate"],
                "percentile": result["percentile_among_test_applications"],
                "top_contributors": result["top_contributors"],
                "model_version": result["model_version"],
            }
        ),
    }


TOOL_SPECS = [
    {
        "name": "get_model_card",
        "description": "Summary of the active model version: outcome, dataset, calibration, headline "
        "test metrics, inputs, promotion gates and limitations.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_run_metrics",
        "description": "Held-out test metrics (ROC-AUC, PR-AUC, Brier, ECE, 95% CIs, threshold metrics, "
        "confusion matrix) for one or all candidate models, plus the prevalence-only baseline.",
        "input_schema": {
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "enum": ["logistic_regression", "random_forest", "hist_gradient_boosting"],
                }
            },
        },
    },
    {
        "name": "get_feature_attributions",
        "description": "Global permutation importance for a model (drop in PR-AUC when an input is shuffled).",
        "input_schema": {
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "enum": ["logistic_regression", "random_forest", "hist_gradient_boosting"],
                }
            },
        },
    },
    {
        "name": "get_dataset_card",
        "description": "Dataset provenance, filtering funnel, missingness warnings and split sizes.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "compare_model_versions",
        "description": "All registered model versions with status and headline metrics.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_drift_report",
        "description": "Input-drift PSI per feature and operational stats for demo traffic.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_fairness_summary",
        "description": "Group-level fairness audit for one protected attribute (historical disparities "
        "and model error rates).",
        "input_schema": {
            "type": "object",
            "properties": {"attribute": {"type": "string", "enum": list(hmda.PROTECTED_AUDIT_COLUMNS)}},
            "required": ["attribute"],
        },
    },
    {
        "name": "get_demo_scenario",
        "description": "Score one of the fictional demo scenarios and return its explanation.",
        "input_schema": {
            "type": "object",
            "properties": {"scenario_id": {"type": "string"}},
            "required": ["scenario_id"],
        },
    },
]

TOOLS: dict[str, Callable[..., dict]] = {
    "get_model_card": get_model_card,
    "get_run_metrics": get_run_metrics,
    "get_feature_attributions": get_feature_attributions,
    "get_dataset_card": get_dataset_card,
    "compare_model_versions": compare_model_versions,
    "get_drift_report": get_drift_report,
    "get_fairness_summary": get_fairness_summary,
    "get_demo_scenario": get_demo_scenario,
}


def run_tool(name: str, arguments: dict | None) -> dict:
    if name not in TOOLS:
        raise ValueError(f"Tool {name!r} is not allowed")
    spec = next(s for s in TOOL_SPECS if s["name"] == name)
    allowed = set(spec["input_schema"].get("properties", {}))
    args = {k: v for k, v in (arguments or {}).items() if k in allowed and isinstance(v, str)}
    return TOOLS[name](**args)


def scenario_evidence(result: dict) -> dict:
    """Wrap a caller-supplied prediction as evidence. Only known, typed fields are kept."""
    keep = {
        "model_version": str,
        "algorithm": str,
        "estimated_probability": float,
        "test_base_rate": float,
        "percentile_among_test_applications": float,
        "research_threshold": float,
    }
    data = {k: t(result[k]) for k, t in keep.items() if k in result and result[k] is not None}
    contributors = []
    for c in (result.get("top_contributors") or [])[:8]:
        if c.get("feature") in EXPLANATION_GROUPS:
            contributors.append(
                {
                    "feature": c["feature"],
                    "label": EXPLANATION_GROUPS[c["feature"]][0],
                    "value_display": str(c.get("value_display", ""))[:40],
                    "contribution": float(c.get("contribution", 0.0)),
                    "direction": c.get("direction")
                    if c.get("direction") in ("up", "down", "neutral")
                    else "neutral",
                }
            )
    data["top_contributors"] = contributors
    return {"evidence_id": "scenario:current", "title": "Scenario currently on screen", "data": _round(data)}
