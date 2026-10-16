"""Model explanations.

* Global: permutation importance on held-out data (model-agnostic), plus standardised
  coefficients for the logistic regression.
* Local: reference substitution - for each input, swap the scenario's value for the typical
  training value and measure how much the estimate moves. The changes do not add up to the
  total and they describe the model's behaviour, not causes in the real world.
* Sensitivity: vary one input across a range while holding the others fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from loan_approval.data import hmda

SENSITIVITY_GRID = {
    "loan_amount": np.linspace(50_000, 1_000_000, 20),
    "property_value": np.linspace(75_000, 1_500_000, 20),
    "income": np.linspace(20, 400, 20),
    "combined_loan_to_value_ratio": np.linspace(20, 110, 19),
    "loan_term": np.array([120, 180, 240, 300, 360]),
}


def reference_values(train_inputs: pd.DataFrame) -> dict:
    ref = {}
    for col in hmda.INPUT_FEATURES:
        if col in hmda.NUMERIC_INPUTS:
            ref[col] = float(train_inputs[col].astype(float).median())
        else:
            ref[col] = str(train_inputs[col].astype(str).mode().iloc[0])
    return ref


def _proba(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[hmda.INPUT_FEATURES])[:, 1]


# Inputs that only make sense together are explained together, so the explanation never asks the
# model about impossible applications (e.g. a new loan amount with the old loan-to-value ratio).
EXPLANATION_GROUPS = {
    "loan_size_collateral": (
        "Loan size and collateral",
        ["loan_amount", "property_value", "combined_loan_to_value_ratio", "conforming_loan_limit"],
    ),
    "debt_to_income_ratio": ("Debt-to-income ratio", ["debt_to_income_ratio"]),
    "income": ("Income", ["income"]),
    "loan_purpose": ("Loan purpose", ["loan_purpose"]),
    "loan_product": ("Loan type and term", ["loan_type", "loan_term"]),
    "lien_status": ("Lien position", ["lien_status"]),
    "property_use": ("Occupancy and units", ["occupancy_type", "total_units"]),
    "preapproval": ("Preapproval", ["preapproval"]),
}
LOCAL_METHOD = (
    "Exact Shapley values over 8 input groups, against a typical Texas application from the "
    "training split (median amounts, most common categories). The contributions add up to the "
    "difference between this estimate and the typical application's estimate."
)


def _group_display(features: list[str], values) -> str:
    return " · ".join(display_value(f, values[f]) for f in features)


def local_contributions(model, scenario: pd.DataFrame, reference: dict) -> list[dict]:
    """Shapley values over input groups versus the reference application (sum = p - p_ref)."""
    groups = list(EXPLANATION_GROUPS)
    m = len(groups)
    row = scenario.iloc[0]
    coalitions = []
    for mask in range(1 << m):
        values = dict(reference)
        for g_i, g in enumerate(groups):
            if mask >> g_i & 1:
                for f in EXPLANATION_GROUPS[g][1]:
                    values[f] = row[f]
        coalitions.append(values)
    v = _proba(model, pd.DataFrame(coalitions, columns=hmda.INPUT_FEATURES))
    fact = [float(np.prod(range(1, k + 1))) for k in range(m + 1)]
    phi = np.zeros(m)
    for mask in range(1 << m):
        size = bin(mask).count("1")
        for g_i in range(m):
            if mask >> g_i & 1:
                continue
            weight = fact[size] * fact[m - size - 1] / fact[m]
            phi[g_i] += weight * (v[mask | (1 << g_i)] - v[mask])
    rows = []
    for g_i, g in enumerate(groups):
        label, features = EXPLANATION_GROUPS[g]
        delta = float(phi[g_i])
        rows.append(
            {
                "feature": g,
                "label": label,
                "inputs": features,
                "value_display": _group_display(features, row),
                "reference_display": _group_display(features, reference),
                "contribution": round(delta, 5),
                "direction": "up" if delta > 0.0005 else "down" if delta < -0.0005 else "neutral",
            }
        )
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return rows


def reference_probability(model, reference: dict) -> float:
    return float(_proba(model, pd.DataFrame([reference], columns=hmda.INPUT_FEATURES))[0])


def sensitivity(model, scenario: pd.DataFrame, feature: str) -> list[dict]:
    if feature in SENSITIVITY_GRID:
        grid = list(SENSITIVITY_GRID[feature])
        current = float(scenario.iloc[0][feature])
        grid = sorted({*grid, current})
    elif feature in hmda.CATEGORICAL_INPUTS:
        grid = hmda.DTI_ORDER if feature == "debt_to_income_ratio" else list(hmda.CATEGORICAL_INPUTS[feature])
    else:
        raise ValueError(f"Unknown feature {feature!r}")
    frame = pd.concat([scenario] * len(grid), ignore_index=True)
    frame[feature] = grid
    probs = _proba(model, frame)
    current_value = scenario.iloc[0][feature]
    return [
        {
            "value": float(v) if feature in hmda.NUMERIC_INPUTS else str(v),
            "value_display": display_value(feature, v),
            "probability": round(float(p), 5),
            "is_current": (float(v) == float(current_value))
            if feature in hmda.NUMERIC_INPUTS
            else str(v) == str(current_value),
        }
        for v, p in zip(grid, probs, strict=True)
    ]


def global_importance(model, inputs: pd.DataFrame, y: np.ndarray, seed: int = 0, repeats: int = 5):
    result = permutation_importance(
        model,
        inputs[hmda.INPUT_FEATURES],
        y,
        scoring=lambda est, X, yy: average_precision_score(yy, est.predict_proba(X)[:, 1]),
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    rows = [
        {
            "feature": col,
            "label": hmda.FEATURE_LABELS[col],
            "importance": float(result.importances_mean[i]),
            "std": float(result.importances_std[i]),
        }
        for i, col in enumerate(hmda.INPUT_FEATURES)
    ]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return {
        "method": f"Permutation importance: drop in PR-AUC when one input is shuffled "
        f"({repeats} repeats, {len(y):,} held-out applications)",
        "features": rows,
    }


def logistic_coefficients(pipeline) -> list[dict]:
    """Standardised coefficients from an *uncalibrated* logistic-regression pipeline."""
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["encode"].get_feature_names_out()
    coefs = model.coef_[0]
    rows = [
        {"term": str(n), "coefficient": float(c), "odds_ratio": float(np.exp(c))}
        for n, c in zip(names, coefs, strict=True)
    ]
    rows.sort(key=lambda r: abs(r["coefficient"]), reverse=True)
    return rows


def display_value(feature: str, value) -> str:
    if feature in ("loan_amount", "property_value"):
        return f"${float(value):,.0f}"
    if feature == "income":
        return f"${float(value) * 1000:,.0f}/yr"
    if feature == "combined_loan_to_value_ratio":
        return f"{float(value):.1f}%"
    if feature == "loan_term":
        months = float(value)
        return f"{months / 12:.0f} years" if months % 12 == 0 else f"{months:.0f} months"
    return hmda.code_label(feature, str(value))
