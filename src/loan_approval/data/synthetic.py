"""Synthetic HMDA-format applications.

Used for CI, tests and offline development. The rows are generated from an explicit, documented
formula, so a model trained on them says nothing about real lending. Demographic columns are
random labels used only to exercise the fairness code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loan_approval.data import hmda

RACES = ["White", "Black or African American", "Asian", "Race Not Available", "Joint"]
ETHNICITIES = ["Not Hispanic or Latino", "Hispanic or Latino", "Ethnicity Not Available", "Joint"]
SEXES = ["Male", "Female", "Joint", "Sex Not Available"]
AGES = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]


def _dti_code(value: float) -> str:
    if value < 20:
        return "<20%"
    if value < 30:
        return "20%-<30%"
    if value < 36:
        return "30%-<36%"
    if value < 50:
        return str(int(value))
    if value <= 60:
        return "50%-60%"
    return ">60%"


def generate_raw(rows: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = np.round(np.exp(rng.normal(np.log(110), 0.55, rows)))
    income = np.clip(income, 15, 2000)
    property_value = np.round(np.exp(rng.normal(np.log(340_000), 0.5, rows)), -4) + 5000
    cltv = np.clip(rng.normal(82, 15, rows), 10, 120)
    loan_amount = np.clip(np.round(property_value * cltv / 100 * rng.uniform(0.85, 1.0, rows), -4) + 5000,
                          15_000, 2_500_000)
    dti = np.clip(rng.normal(38, 11, rows), 5, 75)
    loan_type = rng.choice(list(hmda.LOAN_TYPE), rows, p=[0.68, 0.2, 0.1, 0.02])
    purpose = rng.choice(list(hmda.LOAN_PURPOSE), rows, p=[0.66, 0.04, 0.12, 0.13, 0.05])
    lien = rng.choice(list(hmda.LIEN_STATUS), rows, p=[0.88, 0.12])
    occupancy = rng.choice(list(hmda.OCCUPANCY_TYPE), rows, p=[0.9, 0.02, 0.08])
    units = rng.choice(list(hmda.TOTAL_UNITS), rows, p=[0.97, 0.015, 0.008, 0.007])
    preapproval = rng.choice(list(hmda.PREAPPROVAL), rows, p=[0.03, 0.97])
    conforming = np.where(loan_amount > 806_500, "NC", "C")
    term = rng.choice([360, 360, 360, 300, 240, 180], rows)

    # Documented generating process (log-odds of denial).
    logit = (
        -2.4
        + 0.09 * (dti - 38)
        + 0.12 * np.maximum(dti - 50, 0)
        + 0.03 * (cltv - 82)
        - 0.6 * (np.log(income) - np.log(110))
        + np.where(purpose == "1", -0.7, 0.4)
        + np.where(lien == "2", 0.6, 0.0)
        + np.where(occupancy == "3", 0.3, 0.0)
        + rng.normal(0, 0.6, rows)
    )
    denied = rng.uniform(size=rows) < 1 / (1 + np.exp(-logit))
    approved_action = rng.choice(["1", "2"], rows, p=[0.93, 0.07])
    action = np.where(denied, "3", approved_action)
    reasons = rng.choice(["1", "3", "4", "9"], rows, p=[0.4, 0.3, 0.2, 0.1])

    frame = pd.DataFrame(
        {
            "action_taken": action,
            "reverse_mortgage": "2",
            "open-end_line_of_credit": "2",
            "business_or_commercial_purpose": "2",
            "derived_dwelling_category": "Single Family (1-4 Units):Site-Built",
            "loan_amount": loan_amount.astype(int).astype(str),
            "property_value": property_value.astype(int).astype(str),
            "income": income.astype(int).astype(str),
            "debt_to_income_ratio": [_dti_code(v) for v in dti],
            "loan_to_value_ratio": np.round(cltv, 2).astype(str),
            "loan_term": term.astype(str),
            "loan_type": loan_type,
            "loan_purpose": purpose,
            "lien_status": lien,
            "occupancy_type": occupancy,
            "total_units": units,
            "preapproval": preapproval,
            "conforming_loan_limit": conforming,
            "derived_race": rng.choice(RACES, rows, p=[0.6, 0.12, 0.1, 0.15, 0.03]),
            "derived_ethnicity": rng.choice(ETHNICITIES, rows, p=[0.55, 0.28, 0.14, 0.03]),
            "derived_sex": rng.choice(SEXES, rows, p=[0.36, 0.22, 0.34, 0.08]),
            "applicant_age": rng.choice(AGES, rows),
            "denial_reason-1": np.where(denied, reasons, "10"),
        }
    )
    # A small share of incomplete rows exercises the missing-value path.
    missing = rng.uniform(size=rows) < 0.03
    frame.loc[missing, "debt_to_income_ratio"] = "NA"
    return frame
