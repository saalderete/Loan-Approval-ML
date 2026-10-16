"""Fair-lending style audit.

Protected attributes are never model inputs. They are joined back to held-out predictions here so we
can measure (a) disparities already present in the historical decisions and (b) how the model's
estimates and errors differ across groups. Small groups are reported as "insufficient data" instead
of being turned into a misleading rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loan_approval.data import hmda
from loan_approval.evaluation.metrics import wilson_interval

MIN_GROUP_SIZE = 300
AGE_LABELS = {"8888": "Not available"}
NOT_AVAILABLE = {
    "Race Not Available",
    "Ethnicity Not Available",
    "Sex Not Available",
    "Not available",
    "Free Form Text Only",
}


AGE_ORDER = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74", "Not available"]


def _order_key(attribute: str):
    if attribute == "applicant_age":
        return lambda g: (AGE_ORDER.index(g) if g in AGE_ORDER else len(AGE_ORDER), g)
    return lambda g: (g in NOT_AVAILABLE, g)


def _group_values(frame: pd.DataFrame, attribute: str) -> pd.Series:
    values = frame[attribute].astype(str)
    if attribute == "applicant_age":
        values = values.replace(AGE_LABELS)
    return values


def group_table(
    frame: pd.DataFrame,
    attribute: str,
    probability: np.ndarray | None = None,
    threshold: float | None = None,
) -> dict:
    values = _group_values(frame, attribute)
    y = frame["denied"].to_numpy()
    rows = []
    for group in sorted(values.unique(), key=_order_key(attribute)):
        mask = (values == group).to_numpy()
        n = int(mask.sum())
        row: dict = {"group": group, "n": n, "not_reported": group in NOT_AVAILABLE}
        if n < MIN_GROUP_SIZE:
            row["status"] = "insufficient data"
            rows.append(row)
            continue
        row["status"] = "ok"
        positives = int(y[mask].sum())
        lo, hi = wilson_interval(positives, n)
        row.update(
            observed_denial_rate=positives / n,
            observed_ci_low=lo,
            observed_ci_high=hi,
        )
        if probability is not None:
            p = probability[mask]
            row["mean_estimated_probability"] = float(p.mean())
            # Positive gap = more denials than the model's financial inputs account for.
            row["unexplained_gap"] = row["observed_denial_rate"] - row["mean_estimated_probability"]
            if threshold is not None:
                flagged = p >= threshold
                yg = y[mask]
                row["flagged_rate"] = float(flagged.mean())
                row["recall"] = float(flagged[yg == 1].mean()) if (yg == 1).any() else None
                row["false_positive_rate"] = float(flagged[yg == 0].mean()) if (yg == 0).any() else None
                row["precision"] = float(yg[flagged].mean()) if flagged.any() else None
        rows.append(row)

    reportable = [r for r in rows if r["status"] == "ok" and not r["not_reported"]]
    reference = max(reportable, key=lambda r: r["n"])["group"] if reportable else None
    if reference is not None:
        ref = next(r for r in rows if r["group"] == reference)
        for r in rows:
            if r["status"] != "ok":
                continue
            r["observed_rate_ratio_vs_reference"] = (
                r["observed_denial_rate"] / ref["observed_denial_rate"]
                if ref["observed_denial_rate"]
                else None
            )
            for key in ("false_positive_rate", "recall", "unexplained_gap"):
                if r.get(key) is not None and ref.get(key) is not None:
                    r[f"{key}_diff_vs_reference"] = r[key] - ref[key]
    return {
        "attribute": attribute,
        "label": hmda.PROTECTED_AUDIT_COLUMNS[attribute],
        "reference_group": reference,
        "min_group_size": MIN_GROUP_SIZE,
        "groups": rows,
    }


def audit(
    population: pd.DataFrame,
    test_frame: pd.DataFrame,
    test_probability: np.ndarray,
    threshold: float,
) -> dict:
    return {
        "historical_disparities": [group_table(population, attr) for attr in hmda.PROTECTED_AUDIT_COLUMNS],
        "model_audit": [
            group_table(test_frame, attr, test_probability, threshold)
            for attr in hmda.PROTECTED_AUDIT_COLUMNS
        ],
        "notes": [
            "Protected attributes are used only for this audit and are never model inputs.",
            "Historical disparities are computed on all modelled applications; the model audit uses "
            "the held-out test split.",
            "The 'unexplained gap' is the observed denial rate minus the model's average estimate "
            "for the group. Because the model sees only the 13 HMDA application fields (no credit "
            "scores, assets or employment), a gap is a signal for further review, not evidence of "
            "discrimination, and a small gap is not proof of fairness.",
            f"Groups with fewer than {MIN_GROUP_SIZE} applications are marked as insufficient data.",
            "Fairness metrics are not the same as legal compliance. Real lending models require "
            "fair-lending, model-risk and legal review.",
        ],
    }
