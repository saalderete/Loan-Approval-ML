import pandas as pd
import pytest

from loan_approval.data import hmda
from loan_approval.data.synthetic import generate_raw
from loan_approval.data.validate import SchemaError, build_model_frame


def test_funnel_is_monotonic_and_ends_at_output(synthetic_frame):
    frame, report = synthetic_frame
    rows = [s["rows"] for s in report.funnel]
    assert rows == sorted(rows, reverse=True)
    assert rows[-1] == len(frame)


def test_incomplete_rows_are_removed_and_flagged():
    raw = generate_raw(2000, seed=1)
    frame, report = build_model_frame(raw)
    assert (raw["debt_to_income_ratio"] == "NA").sum() > 0
    assert not frame["debt_to_income_ratio"].isin(["NA", "Exempt"]).any()
    dti = next(m for m in report.missingness if m["feature"] == "debt_to_income_ratio")
    assert dti["missing_rate"] > 0


def test_out_of_domain_values_are_dropped():
    raw = generate_raw(500, seed=2)
    raw.loc[0, "loan_to_value_ratio"] = "5000"
    raw.loc[1, "income"] = "-10"
    frame, report = build_model_frame(raw)
    assert report.out_of_domain["combined_loan_to_value_ratio"] >= 1
    assert report.out_of_domain["income"] >= 1
    assert frame["combined_loan_to_value_ratio"].max() <= 250


def test_missing_required_column_raises():
    raw = generate_raw(100).drop(columns=["loan_amount"])
    with pytest.raises(SchemaError):
        build_model_frame(raw)


def test_target_definition():
    raw = pd.concat([generate_raw(300, seed=5)], ignore_index=True)
    frame, _ = build_model_frame(raw)
    assert set(frame["denied"].unique()) <= {0, 1}
    assert ((frame["action_taken"] == "3") == (frame["denied"] == 1)).all()


def test_model_inputs_exclude_protected_post_decision_and_proxy_columns():
    banned = (
        set(hmda.POST_DECISION_COLUMNS)
        | set(hmda.PROTECTED_AUDIT_COLUMNS)
        | set(hmda.PROXY_RISK_COLUMNS)
        | set(hmda.PROCESS_ARTIFACT_COLUMNS)
    )
    assert not banned & set(hmda.INPUT_FEATURES)
    assert not banned & set(hmda.INPUT_COLUMN_MAP)
