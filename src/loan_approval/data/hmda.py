"""HMDA code tables and the leakage-safe feature contract.

Codes follow the FFIEC "Public HMDA - LAR Data Fields" documentation (2018+ format).
Only fields known when an application is submitted are allowed as model inputs.
"""

from __future__ import annotations

HMDA_SOURCE = {
    "title": "HMDA Loan/Application Register (public, modified LAR) - Texas, 2025",
    "publisher": "Federal Financial Institutions Examination Council (FFIEC) / CFPB",
    "landing_page": "https://ffiec.cfpb.gov/data-browser/",
    "api": "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv",
    "documentation": "https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields",
}

ACTION_TAKEN = {
    "1": "Loan originated",
    "2": "Application approved but not accepted",
    "3": "Application denied",
    "4": "Application withdrawn by applicant",
    "5": "File closed for incompleteness",
    "6": "Purchased loan",
    "7": "Preapproval request denied",
    "8": "Preapproval request approved but not accepted",
}

LOAN_TYPE = {"1": "Conventional", "2": "FHA", "3": "VA", "4": "USDA / RHS"}
LOAN_PURPOSE = {
    "1": "Home purchase",
    "2": "Home improvement",
    "31": "Refinance",
    "32": "Cash-out refinance",
    "4": "Other purpose",
}
LIEN_STATUS = {"1": "First lien", "2": "Subordinate lien"}
OCCUPANCY_TYPE = {"1": "Principal residence", "2": "Second residence", "3": "Investment property"}
PREAPPROVAL = {"1": "Preapproval requested", "2": "Preapproval not requested"}
CONFORMING_LOAN_LIMIT = {"C": "Conforming", "NC": "Nonconforming", "NA": "Not applicable"}
TOTAL_UNITS = {"1": "1 unit", "2": "2 units", "3": "3 units", "4": "4 units"}

DENIAL_REASON = {
    "1": "Debt-to-income ratio",
    "2": "Employment history",
    "3": "Credit history",
    "4": "Collateral",
    "5": "Insufficient cash",
    "6": "Unverifiable information",
    "7": "Credit application incomplete",
    "8": "Mortgage insurance denied",
    "9": "Other",
    "10": "Not applicable",
}

# HMDA publishes DTI in bands, with exact integers between 36 and 49.
DTI_VALUES: dict[str, float] = {
    "<20%": 15.0,
    "20%-<30%": 25.0,
    "30%-<36%": 33.0,
    **{str(v): float(v) for v in range(36, 50)},
    "50%-60%": 55.0,
    ">60%": 65.0,
}
DTI_ORDER = list(DTI_VALUES)

# Columns that only exist after a decision (or describe the decision) - never model inputs.
POST_DECISION_COLUMNS = [
    "action_taken",
    "purchaser_type",
    "interest_rate",
    "rate_spread",
    "hoepa_status",
    "total_loan_costs",
    "total_points_and_fees",
    "origination_charges",
    "discount_points",
    "lender_credits",
    "prepayment_penalty_term",
    "intro_rate_period",
    "denial_reason-1",
    "denial_reason-2",
    "denial_reason-3",
    "denial_reason-4",
]

# Columns excluded because they record process steps that leak the outcome
# (e.g. "not applicable" codes are used when a step never happened).
PROCESS_ARTIFACT_COLUMNS = [
    "applicant_credit_score_type",
    "co-applicant_credit_score_type",
    "aus-1",
    "aus-2",
    "aus-3",
    "aus-4",
    "aus-5",
    "submission_of_application",
    "initially_payable_to_institution",
]

# Protected characteristics (ECOA / Fair Housing Act). Used ONLY for the fairness audit.
PROTECTED_AUDIT_COLUMNS = {
    "derived_race": "Race (derived)",
    "derived_ethnicity": "Ethnicity (derived)",
    "derived_sex": "Sex (derived)",
    "applicant_age": "Applicant age",
}

# Excluded as proxies for protected classes or for lender identity.
PROXY_RISK_COLUMNS = [
    "lei",
    "census_tract",
    "county_code",
    "derived_msa-md",
    "tract_population",
    "tract_minority_population_percent",
    "ffiec_msa_md_median_family_income",
    "tract_to_msa_income_percentage",
    "tract_owner_occupied_units",
    "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
]

# Mapping from the HMDA data-browser column name to the model's input name.
INPUT_COLUMN_MAP = {
    "loan_amount": "loan_amount",
    "property_value": "property_value",
    "income": "income",
    "debt_to_income_ratio": "debt_to_income_ratio",
    "loan_to_value_ratio": "combined_loan_to_value_ratio",
    "loan_term": "loan_term",
    "loan_type": "loan_type",
    "loan_purpose": "loan_purpose",
    "lien_status": "lien_status",
    "occupancy_type": "occupancy_type",
    "total_units": "total_units",
    "preapproval": "preapproval",
    "conforming_loan_limit": "conforming_loan_limit",
}

NUMERIC_INPUTS = [
    "loan_amount",
    "property_value",
    "income",
    "combined_loan_to_value_ratio",
    "loan_term",
]
CATEGORICAL_INPUTS = {
    "debt_to_income_ratio": DTI_VALUES,
    "loan_type": LOAN_TYPE,
    "loan_purpose": LOAN_PURPOSE,
    "lien_status": LIEN_STATUS,
    "occupancy_type": OCCUPANCY_TYPE,
    "total_units": TOTAL_UNITS,
    "preapproval": PREAPPROVAL,
    "conforming_loan_limit": CONFORMING_LOAN_LIMIT,
}
INPUT_FEATURES = [
    "loan_amount",
    "property_value",
    "income",
    "debt_to_income_ratio",
    "combined_loan_to_value_ratio",
    "loan_term",
    "loan_type",
    "loan_purpose",
    "lien_status",
    "occupancy_type",
    "total_units",
    "preapproval",
    "conforming_loan_limit",
]

FEATURE_LABELS = {
    "loan_amount": "Loan amount",
    "property_value": "Property value",
    "income": "Gross annual income",
    "debt_to_income_ratio": "Debt-to-income ratio",
    "combined_loan_to_value_ratio": "Combined loan-to-value",
    "loan_term": "Loan term",
    "loan_type": "Loan type",
    "loan_purpose": "Loan purpose",
    "lien_status": "Lien position",
    "occupancy_type": "Occupancy",
    "total_units": "Dwelling units",
    "preapproval": "Preapproval",
    "conforming_loan_limit": "Conforming limit",
}

FEATURE_GLOSSARY = {
    "loan_amount": "Amount requested, in dollars. HMDA rounds it to the midpoint of a $10,000 band.",
    "property_value": "Value of the property relied on, in dollars (also rounded to a $10,000 band).",
    "income": "Gross annual income relied on, in thousands of dollars.",
    "debt_to_income_ratio": "Monthly debt payments divided by monthly income, as HMDA bands "
    "(exact values are published only between 36% and 49%).",
    "combined_loan_to_value_ratio": "All loans secured by the property divided by its value, in percent.",
    "loan_term": "Number of months until the loan would be repaid.",
    "loan_type": "Whether the loan is conventional or insured/guaranteed by FHA, VA or USDA.",
    "loan_purpose": "Purchase, refinance, cash-out refinance, home improvement or other.",
    "lien_status": "Whether the loan would be the first or a subordinate lien on the property.",
    "occupancy_type": "Principal residence, second home or investment property.",
    "total_units": "Number of dwelling units in the property (1-4 for this model).",
    "preapproval": "Whether the applicant requested a preapproval.",
    "conforming_loan_limit": "Whether the amount is within the GSE conforming loan limit for the county.",
}


def code_label(feature: str, code: str) -> str:
    if feature == "debt_to_income_ratio":
        return f"{code}%" if code.isdigit() else code
    table = CATEGORICAL_INPUTS.get(feature)
    if table is None:
        return code
    return table.get(code, code)
