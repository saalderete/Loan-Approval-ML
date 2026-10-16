import pytest
from pydantic import ValidationError

from loan_approval.features.schema import ScenarioInput, load_demo_scenarios, schema_summary


def test_all_demo_scenarios_are_valid():
    scenarios = load_demo_scenarios()
    assert len(scenarios) >= 5
    assert all(s["synthetic"] for s in scenarios)


@pytest.mark.parametrize(
    "field,value",
    [
        ("loan_amount", -1),
        ("income", 0),
        ("combined_loan_to_value_ratio", 400),
        ("loan_term", 600),
        ("debt_to_income_ratio", "35"),
        ("loan_type", "9"),
        ("total_units", "5-24"),
    ],
)
def test_out_of_range_values_are_rejected(demo_inputs, field, value):
    with pytest.raises(ValidationError):
        ScenarioInput(**{**demo_inputs, field: value})


def test_cltv_cannot_be_far_below_own_ltv(demo_inputs):
    with pytest.raises(ValidationError, match="Combined LTV"):
        ScenarioInput(
            **{
                **demo_inputs,
                "loan_amount": 300_000,
                "property_value": 310_000,
                "combined_loan_to_value_ratio": 50,
            }
        )


def test_loan_far_above_property_value_rejected(demo_inputs):
    with pytest.raises(ValidationError):
        ScenarioInput(
            **{
                **demo_inputs,
                "loan_amount": 900_000,
                "property_value": 100_000,
                "combined_loan_to_value_ratio": 250,
            }
        )


def test_unknown_fields_are_rejected(demo_inputs):
    with pytest.raises(ValidationError):
        ScenarioInput(**{**demo_inputs, "applicant_race": "White"})


def test_schema_summary_lists_all_inputs():
    summary = schema_summary()
    assert len(summary["fields"]) == 13
    assert summary["schema_version"]
