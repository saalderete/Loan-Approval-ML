import numpy as np
import pandas as pd

from loan_approval.data import hmda
from loan_approval.features.preprocess import ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC, HmdaFeatureBuilder


def _row(**overrides):
    base = {
        "loan_amount": 250_000.0,
        "property_value": 320_000.0,
        "income": 100.0,
        "debt_to_income_ratio": "41",
        "combined_loan_to_value_ratio": 80.0,
        "loan_term": 360.0,
        "loan_type": "1",
        "loan_purpose": "1",
        "lien_status": "1",
        "occupancy_type": "1",
        "total_units": "1",
        "preapproval": "2",
        "conforming_loan_limit": "C",
    }
    return {**base, **overrides}


def test_builder_is_stateless_and_row_independent():
    rows = pd.DataFrame([_row(), _row(debt_to_income_ratio=">60%"), _row(income=50.0)])
    builder = HmdaFeatureBuilder().fit(rows)
    batch = builder.transform(rows)
    single = builder.transform(rows.iloc[[1]])
    pd.testing.assert_frame_equal(batch.iloc[[1]], single)
    assert list(batch.columns) == ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL


def test_formulas():
    out = HmdaFeatureBuilder().transform(pd.DataFrame([_row()]))
    assert out.loc[0, "dti_value"] == 41
    assert out.loc[0, "dti_band"] == "36-42%"
    assert np.isclose(out.loc[0, "ltv_this_loan"], 250_000 / 320_000 * 100)
    assert np.isclose(out.loc[0, "loan_to_income"], 2.5)


def test_every_dti_code_maps_to_a_band():
    out = HmdaFeatureBuilder().transform(pd.DataFrame([_row(debt_to_income_ratio=c) for c in hmda.DTI_ORDER]))
    assert out["dti_value"].notna().all()
    assert out["dti_value"].is_monotonic_increasing


def test_column_order_does_not_matter(trained_env):
    from loan_approval.inference.predict import get_model

    model = get_model().selected
    frame = pd.DataFrame([_row(), _row(debt_to_income_ratio="50%-60%")])
    shuffled = frame[list(reversed(frame.columns))]
    np.testing.assert_allclose(
        model.predict_proba(frame[hmda.INPUT_FEATURES]), model.predict_proba(shuffled[hmda.INPUT_FEATURES])
    )
