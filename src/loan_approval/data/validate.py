"""Population filters, value checks and the data-quality report.

Everything here is deterministic. The report is saved with each model version and shown in the
Dataset Explorer, so every row that is dropped is accounted for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from loan_approval.data import hmda

MISSING_CODES = {"NA", "Exempt", "", "nan", "None"}
REQUIRED_RAW_COLUMNS = [
    "action_taken",
    "reverse_mortgage",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "derived_dwelling_category",
    *hmda.INPUT_COLUMN_MAP.keys(),
    *hmda.PROTECTED_AUDIT_COLUMNS.keys(),
    "denial_reason-1",
]

# The model's domain. Inference inputs are validated against the same ranges.
DOMAIN = {
    "loan_amount": (5_000.0, 3_000_000.0),
    "property_value": (10_000.0, 10_000_000.0),
    "income": (1.0, 5_000.0),
    "combined_loan_to_value_ratio": (1.0, 250.0),
    "loan_term": (1.0, 480.0),
}
MAX_LOAN_TO_PROPERTY = 2.5


class SchemaError(ValueError):
    """The raw file does not have the columns this pipeline was built for."""


@dataclass
class QualityReport:
    funnel: list[dict] = field(default_factory=list)
    missingness: list[dict] = field(default_factory=list)
    out_of_domain: dict[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0
    warnings: list[str] = field(default_factory=list)

    def add_stage(self, name: str, description: str, frame: pd.DataFrame) -> None:
        denied = (frame["action_taken"] == "3").mean() if len(frame) else float("nan")
        self.funnel.append(
            {
                "stage": name,
                "description": description,
                "rows": len(frame),
                "denial_rate": None if np.isnan(denied) else round(float(denied), 4),
            }
        )

    def to_dict(self) -> dict:
        return {
            "funnel": self.funnel,
            "missingness": self.missingness,
            "out_of_domain": self.out_of_domain,
            "duplicates_removed": self.duplicates_removed,
            "warnings": self.warnings,
        }


def check_schema(raw: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise SchemaError(f"Raw HMDA file is missing required columns: {missing}")


def _is_missing(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().isin(MISSING_CODES)


def build_model_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    """Filter the raw LAR to the modelled population and return model inputs + target + audit columns."""
    check_schema(raw)
    report = QualityReport()
    df = raw.copy()
    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    report.add_stage("raw", "All rows in the downloaded file", df)

    df = df[df["action_taken"].isin(["1", "2", "3"])]
    report.add_stage("decided", "Approved (originated or approved-not-accepted) or denied", df)

    df = df[
        (df["reverse_mortgage"] == "2")
        & (df["open-end_line_of_credit"] == "2")
        & (df["business_or_commercial_purpose"] == "2")
    ]
    report.add_stage(
        "consumer_closed_end",
        "Closed-end, non-reverse mortgages for a consumer (not business) purpose",
        df,
    )

    df = df[
        (df["derived_dwelling_category"] == "Single Family (1-4 Units):Site-Built")
        & df["total_units"].isin(list(hmda.TOTAL_UNITS))
        & df["loan_purpose"].isin(list(hmda.LOAN_PURPOSE))
    ]
    report.add_stage("site_built_1_4", "Site-built, 1-4 unit homes with a known loan purpose", df)

    # Missingness is measured before it is filtered, because it is strongly related to the outcome.
    denied = df["action_taken"] == "3"
    for raw_col, name in hmda.INPUT_COLUMN_MAP.items():
        miss = _is_missing(df[raw_col])
        report.missingness.append(
            {
                "feature": name,
                "missing_rate": round(float(miss.mean()), 4),
                "denial_rate_when_missing": round(float(denied[miss].mean()), 4)
                if miss.any()
                else None,
                "denial_rate_when_present": round(float(denied[~miss].mean()), 4),
            }
        )
    flagged = [
        m["feature"]
        for m in report.missingness
        if m["denial_rate_when_missing"] is not None
        and m["missing_rate"] > 0.001
        and abs(m["denial_rate_when_missing"] - m["denial_rate_when_present"]) > 0.05
    ]
    if flagged:
        report.warnings.append(
            "Missing values are strongly related to the outcome for: "
            + ", ".join(flagged)
            + ". Lenders often leave fields blank when an application never reached that step, "
            "so missingness leaks the decision. The model is trained on complete applications only."
        )

    any_missing = np.zeros(len(df), dtype=bool)
    for raw_col in hmda.INPUT_COLUMN_MAP:
        any_missing |= _is_missing(df[raw_col]).to_numpy()
    df = df[~any_missing]
    report.add_stage("complete", "All 13 model inputs reported (no NA / Exempt)", df)

    out = pd.DataFrame({"_row_id": df["_row_id"].to_numpy()})
    for raw_col, name in hmda.INPUT_COLUMN_MAP.items():
        values = df[raw_col].astype(str).str.strip().to_numpy()
        if name in hmda.NUMERIC_INPUTS:
            out[name] = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        else:
            out[name] = values
    out["denied"] = (df["action_taken"].to_numpy() == "3").astype(np.int8)
    out["action_taken"] = df["action_taken"].to_numpy()
    for col in hmda.PROTECTED_AUDIT_COLUMNS:
        out[col] = df[col].astype(str).to_numpy()
    out["denial_reason_1"] = df["denial_reason-1"].astype(str).to_numpy()

    keep = np.ones(len(out), dtype=bool)
    for name, (lo, hi) in DOMAIN.items():
        bad = ~out[name].between(lo, hi) | out[name].isna()
        report.out_of_domain[name] = int((bad & keep).sum())
        keep &= ~bad.to_numpy()
    ratio_bad = out["loan_amount"] > out["property_value"] * MAX_LOAN_TO_PROPERTY
    report.out_of_domain["loan_amount_vs_property_value"] = int((ratio_bad & keep).sum())
    keep &= ~ratio_bad.to_numpy()
    unknown_dti = ~out["debt_to_income_ratio"].isin(hmda.DTI_ORDER)
    report.out_of_domain["debt_to_income_ratio_unknown_code"] = int((unknown_dti & keep).sum())
    keep &= ~unknown_dti.to_numpy()
    out = out[keep]
    report.add_stage("in_domain", "Values inside the documented model domain", out)

    subset = [c for c in out.columns if c != "_row_id"]
    dup = out.duplicated(subset=subset, keep="first")
    report.duplicates_removed = int(dup.sum())
    out = out[~dup].reset_index(drop=True)
    report.add_stage("deduplicated", "Exact duplicate records removed", out)
    return out, report


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[hmda.INPUT_FEATURES].copy()
