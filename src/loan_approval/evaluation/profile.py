"""Aggregate dataset profile for the Dataset Explorer (no row-level data is stored)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from loan_approval.data import hmda


def _numeric_hist(series: pd.Series, bins: int = 30) -> dict:
    s = series.astype(float)
    lo, hi = np.percentile(s, [1, 99])
    counts, edges = np.histogram(s.clip(lo, hi), bins=bins, range=(lo, hi))
    return {
        "p01": float(lo),
        "p99": float(hi),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "bins": [
            {"start": float(edges[i]), "end": float(edges[i + 1]), "count": int(counts[i])}
            for i in range(bins)
        ],
    }


def _binned_denial(series: pd.Series, y: pd.Series, bins: int = 10) -> list[dict]:
    s = series.astype(float)
    q = pd.qcut(s, bins, duplicates="drop")
    grouped = pd.DataFrame({"bin": q, "y": y}).groupby("bin", observed=True)["y"].agg(["size", "mean"])
    return [
        {
            "range": f"{iv.left:,.0f} – {iv.right:,.0f}",
            "n": int(row["size"]),
            "denial_rate": float(row["mean"]),
        }
        for iv, row in grouped.iterrows()
    ]


def build_profile(frame: pd.DataFrame) -> dict:
    y = frame["denied"]
    numeric = {}
    for col in hmda.NUMERIC_INPUTS:
        numeric[col] = _numeric_hist(frame[col]) | {"denial_by_decile": _binned_denial(frame[col], y)}
    categorical = {}
    for col in hmda.CATEGORICAL_INPUTS:
        g = frame.groupby(frame[col].astype(str))["denied"].agg(["size", "mean"])
        order = hmda.DTI_ORDER if col == "debt_to_income_ratio" else list(g.index)
        categorical[col] = [
            {
                "code": code,
                "label": hmda.code_label(col, code),
                "n": int(g.loc[code, "size"]),
                "share": float(g.loc[code, "size"] / len(frame)),
                "denial_rate": float(g.loc[code, "mean"]),
            }
            for code in order
            if code in g.index
        ]
    denied = frame[frame["denied"] == 1]
    reasons = denied["denial_reason_1"].astype(str).value_counts()
    return {
        "rows": len(frame),
        "denied": int(y.sum()),
        "denial_rate": float(y.mean()),
        "action_taken": {
            hmda.ACTION_TAKEN[k]: int(v) for k, v in frame["action_taken"].astype(str).value_counts().items()
        },
        "numeric": numeric,
        "categorical": categorical,
        "denial_reasons": [
            {"code": k, "label": hmda.DENIAL_REASON.get(k, k), "n": int(v), "share": float(v / len(denied))}
            for k, v in reasons.items()
        ],
    }
