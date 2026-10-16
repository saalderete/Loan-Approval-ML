"""The versioned input contract shared by the API, the dashboard and the smoke tests."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from loan_approval.config import CONFIGS_DIR
from loan_approval.data import hmda
from loan_approval.data.validate import DOMAIN, MAX_LOAN_TO_PROPERTY

SCHEMA_VERSION = "hmda-inputs-v1"

DTICode = Literal[
    "<20%",
    "20%-<30%",
    "30%-<36%",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "43",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50%-60%",
    ">60%",
]


class ScenarioInput(BaseModel):
    """A fictional application. Units follow HMDA: dollars, income in $ thousands, CLTV in %."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    loan_amount: float = Field(ge=DOMAIN["loan_amount"][0], le=DOMAIN["loan_amount"][1])
    property_value: float = Field(ge=DOMAIN["property_value"][0], le=DOMAIN["property_value"][1])
    income: float = Field(
        ge=DOMAIN["income"][0], le=DOMAIN["income"][1], description="Gross annual income, $ thousands"
    )
    debt_to_income_ratio: DTICode
    combined_loan_to_value_ratio: float = Field(
        ge=DOMAIN["combined_loan_to_value_ratio"][0], le=DOMAIN["combined_loan_to_value_ratio"][1]
    )
    loan_term: int = Field(ge=int(DOMAIN["loan_term"][0]), le=int(DOMAIN["loan_term"][1]))
    loan_type: Literal["1", "2", "3", "4"]
    loan_purpose: Literal["1", "2", "31", "32", "4"]
    lien_status: Literal["1", "2"]
    occupancy_type: Literal["1", "2", "3"]
    total_units: Literal["1", "2", "3", "4"]
    preapproval: Literal["1", "2"] = "2"
    conforming_loan_limit: Literal["C", "NC", "NA"] = "C"

    @model_validator(mode="after")
    def _cross_field(self) -> ScenarioInput:
        if self.loan_amount > self.property_value * MAX_LOAN_TO_PROPERTY:
            raise ValueError(
                f"Loan amount is more than {MAX_LOAN_TO_PROPERTY:g}x the property value, "
                "which is outside the model's domain."
            )
        own_ltv = self.loan_amount / self.property_value * 100
        # HMDA rounds amounts to $10k bands, so allow a tolerance before rejecting.
        if self.combined_loan_to_value_ratio + 10 < own_ltv:
            raise ValueError(
                f"Combined LTV ({self.combined_loan_to_value_ratio:.1f}%) cannot be lower than this "
                f"loan's own LTV ({own_ltv:.1f}%)."
            )
        return self

    def to_frame(self) -> pd.DataFrame:
        row = self.model_dump()
        return pd.DataFrame([{k: row[k] for k in hmda.INPUT_FEATURES}])

    def warnings(self) -> list[str]:
        notes = []
        if self.loan_type == "4" or self.total_units in ("3", "4"):
            notes.append("This combination is rare in the training data, so the estimate is less certain.")
        if self.loan_amount > 806_500 and self.conforming_loan_limit == "C":
            notes.append(
                "The amount is above the 2025 baseline conforming limit ($806,500) but marked as "
                "conforming. That is only valid in high-cost counties."
            )
        return notes


def load_demo_scenarios() -> list[dict]:
    data = yaml.safe_load((CONFIGS_DIR / "demo_scenarios.yaml").read_text())
    scenarios = []
    for item in data["scenarios"]:
        ScenarioInput(**item["inputs"])  # fail loudly if a preset drifts out of the schema
        scenarios.append({**item, "synthetic": True})
    return scenarios


def schema_summary() -> dict:
    fields = []
    json_schema = ScenarioInput.model_json_schema()["properties"]
    for name in hmda.INPUT_FEATURES:
        prop = json_schema[name]
        entry = {
            "name": name,
            "label": hmda.FEATURE_LABELS[name],
            "description": hmda.FEATURE_GLOSSARY[name],
            "type": "numeric" if name in hmda.NUMERIC_INPUTS else "categorical",
        }
        if name in hmda.NUMERIC_INPUTS:
            entry["min"], entry["max"] = DOMAIN[name]
        else:
            entry["allowed"] = {
                code: hmda.code_label(name, code)
                for code in (prop.get("enum") or [prop.get("const")])
                if code is not None
            }
        fields.append(entry)
    return {"schema_version": SCHEMA_VERSION, "fields": fields}
