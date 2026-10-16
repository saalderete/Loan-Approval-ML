import numpy as np
import pandas as pd

from loan_approval.evaluation import drift, fairness


def test_small_groups_are_not_reported():
    frame = pd.DataFrame(
        {
            "denied": [0, 1] * 400 + [1] * 10,
            "derived_race": ["White"] * 800 + ["Tiny group"] * 10,
        }
    )
    table = fairness.group_table(frame, "derived_race", np.full(810, 0.5), 0.5)
    tiny = next(g for g in table["groups"] if g["group"] == "Tiny group")
    assert tiny["status"] == "insufficient data"
    assert "observed_denial_rate" not in tiny
    white = next(g for g in table["groups"] if g["group"] == "White")
    assert white["unexplained_gap"] == 0.0


def test_unexplained_gap_sign():
    frame = pd.DataFrame({"denied": [1] * 300 + [0] * 700, "derived_race": ["A"] * 1000})
    table = fairness.group_table(frame, "derived_race", np.full(1000, 0.2), 0.5)
    assert table["groups"][0]["unexplained_gap"] > 0


def test_psi_zero_for_identical_and_positive_for_shift():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "loan_amount": rng.uniform(1e5, 5e5, 5000),
            "property_value": rng.uniform(1e5, 8e5, 5000),
            "income": rng.uniform(40, 300, 5000),
            "combined_loan_to_value_ratio": rng.uniform(40, 100, 5000),
            "loan_term": 360.0,
            "debt_to_income_ratio": "41",
            "loan_type": "1",
            "loan_purpose": "1",
            "lien_status": "1",
            "occupancy_type": "1",
            "total_units": "1",
            "preapproval": "2",
            "conforming_loan_limit": "C",
        }
    )
    baseline = drift.build_baseline(train)
    spec = baseline["features"]["income"]
    same = {k: int(v * 1000) for k, v in drift.expected_proportions(spec).items()}
    assert drift.psi(drift.expected_proportions(spec), same) < 0.01
    shifted = {"0": 1000}
    assert drift.psi(drift.expected_proportions(spec), shifted) > drift.PSI_ALERT
    report = drift.drift_report(baseline, {"income": {"0": 5}})
    income = next(r for r in report["features"] if r["feature"] == "income")
    assert income["status"] == "too few samples"
