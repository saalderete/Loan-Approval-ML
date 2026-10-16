"""Input-drift baseline and Population Stability Index (PSI).

The demo has no real outcomes, so we only monitor whether *inputs* look different from training.
This is an input-drift indicator, not a measure of live accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loan_approval.data import hmda

PSI_WARN = 0.10
PSI_ALERT = 0.25
_EPS = 1e-4


def build_baseline(train_inputs: pd.DataFrame, bins: int = 10) -> dict:
    features = {}
    for col in hmda.INPUT_FEATURES:
        series = train_inputs[col]
        if col in hmda.NUMERIC_INPUTS:
            edges = np.unique(np.quantile(series.astype(float), np.linspace(0, 1, bins + 1)))
            inner = edges[1:-1].tolist()
            idx = np.searchsorted(inner, series.astype(float), side="right")
            counts = np.bincount(idx, minlength=len(inner) + 1)
            features[col] = {
                "kind": "numeric",
                "edges": [round(float(e), 4) for e in inner],
                "proportions": (counts / counts.sum()).round(6).tolist(),
            }
        else:
            share = series.astype(str).value_counts(normalize=True)
            features[col] = {"kind": "categorical", "proportions": share.round(6).to_dict()}
    return {"n": len(train_inputs), "features": features}


def bin_value(spec: dict, value) -> str:
    if spec["kind"] == "numeric":
        return str(int(np.searchsorted(spec["edges"], float(value), side="right")))
    return str(value)


def expected_proportions(spec: dict) -> dict[str, float]:
    if spec["kind"] == "numeric":
        return {str(i): p for i, p in enumerate(spec["proportions"])}
    return dict(spec["proportions"])


def psi(expected: dict[str, float], observed_counts: dict[str, int]) -> float:
    total = sum(observed_counts.values())
    if total == 0:
        return 0.0
    keys = set(expected) | set(observed_counts)
    value = 0.0
    for k in keys:
        e = max(expected.get(k, 0.0), _EPS)
        o = max(observed_counts.get(k, 0) / total, _EPS)
        value += (o - e) * np.log(o / e)
    return float(value)


def drift_report(baseline: dict, observed: dict[str, dict[str, int]], min_samples: int = 30) -> dict:
    rows = []
    n_obs = max((sum(c.values()) for c in observed.values()), default=0)
    for col, spec in baseline["features"].items():
        counts = observed.get(col, {})
        n = sum(counts.values())
        value = psi(expected_proportions(spec), counts) if n else None
        status = "no data"
        if n and n < min_samples:
            status = "too few samples"
        elif value is not None:
            status = "alert" if value >= PSI_ALERT else "warn" if value >= PSI_WARN else "stable"
        rows.append(
            {
                "feature": col,
                "label": hmda.FEATURE_LABELS[col],
                "psi": None if value is None else round(value, 4),
                "n": n,
                "status": status,
            }
        )
    return {
        "observations": n_obs,
        "min_samples": min_samples,
        "thresholds": {"warn": PSI_WARN, "alert": PSI_ALERT},
        "features": rows,
        "note": "Input drift compares scenario inputs submitted to this demo with the training "
        "distribution. Demo traffic has no outcomes, so this says nothing about live accuracy.",
    }
