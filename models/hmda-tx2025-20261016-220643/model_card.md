# Model card: hmda-tx2025-20261016-220643

> Generated automatically from this version's saved evaluation files. Educational research only; not for real credit decisions.

## Summary

- **Selected model:** Histogram Gradient Boosting (`hist_gradient_boosting`), chosen on the selection split by PR-AUC before the test split was scored
- **Outcome:** Estimated probability that a Texas mortgage application with these characteristics was denied by the lender, as recorded in 2025 HMDA data (action_taken = 3), out of applications that were either approved or denied.
- **Positive class:** denied. Prevalence in the modelled data: 16.8%
- **Dataset:** HMDA 2025 TX: approved or denied, complete, site-built 1-4 unit (483,685 applications)
- **Split hash:** `8d6a586640895e35…` (seed 20251)
- **Calibration:** isotonic, fitted on the calibration split
- **Created:** 2026-10-16T22:06:43+00:00 · commit `a4f38e6`

## Test-split results (all candidates, scored once)

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier | ECE | Recall @ thr | Precision @ thr | Threshold |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.820 (0.816–0.824) | 0.589 (0.582–0.596) | 0.0981 | 0.0036 | 56.0% | 52.6% | 0.275 |
| Random Forest | 0.848 (0.844–0.851) | 0.659 (0.653–0.666) | 0.0890 | 0.0035 | 55.2% | 64.9% | 0.322 |
| Histogram Gradient Boosting **(selected)** | 0.857 (0.853–0.860) | 0.685 (0.679–0.691) | 0.0850 | 0.0028 | 52.9% | 72.4% | 0.355 |
| Prevalence-only baseline | 0.500 | 0.168 | 0.1398 | – | – | – | – |

Test rows: 96,737. Confidence intervals: percentile bootstrap, 95% (200 resamples). Thresholds maximise F1 on the selection split and exist only to illustrate trade-offs.

## Inputs

The model uses 13 fields known when an application is submitted:

- **Loan amount** (`loan_amount`): Amount requested, in dollars. HMDA rounds it to the midpoint of a $10,000 band.
- **Property value** (`property_value`): Value of the property relied on, in dollars (also rounded to a $10,000 band).
- **Gross annual income** (`income`): Gross annual income relied on, in thousands of dollars.
- **Debt-to-income ratio** (`debt_to_income_ratio`): Monthly debt payments divided by monthly income, as HMDA bands (exact values are published only between 36% and 49%).
- **Combined loan-to-value** (`combined_loan_to_value_ratio`): All loans secured by the property divided by its value, in percent.
- **Loan term** (`loan_term`): Number of months until the loan would be repaid.
- **Loan type** (`loan_type`): Whether the loan is conventional or insured/guaranteed by FHA, VA or USDA.
- **Loan purpose** (`loan_purpose`): Purchase, refinance, cash-out refinance, home improvement or other.
- **Lien position** (`lien_status`): Whether the loan would be the first or a subordinate lien on the property.
- **Occupancy** (`occupancy_type`): Principal residence, second home or investment property.
- **Dwelling units** (`total_units`): Number of dwelling units in the property (1-4 for this model).
- **Preapproval** (`preapproval`): Whether the applicant requested a preapproval.
- **Conforming limit** (`conforming_loan_limit`): Whether the amount is within the GSE conforming loan limit for the county.

Excluded on purpose:

- *Post-decision fields* (would leak the outcome): `action_taken`, `purchaser_type`, `interest_rate`, `rate_spread`, `hoepa_status`, `total_loan_costs`, `total_points_and_fees`, `origination_charges`, `discount_points`, `lender_credits`, `prepayment_penalty_term`, `intro_rate_period`, `denial_reason-1`, `denial_reason-2`, `denial_reason-3`, `denial_reason-4`
- *Process artifacts* ("not applicable" codes reveal how far an application got): `applicant_credit_score_type`, `co-applicant_credit_score_type`, `aus-1`, `aus-2`, `aus-3`, `aus-4`, `aus-5`, `submission_of_application`, `initially_payable_to_institution`
- *Protected characteristics* (used only in the fairness audit): `derived_race`, `derived_ethnicity`, `derived_sex`, `applicant_age`
- *Geography and lender identity* (proxy risk): `lei`, `census_tract`, `county_code`, `derived_msa-md`, `tract_population`, `tract_minority_population_percent`, `ffiec_msa_md_median_family_income`, `tract_to_msa_income_percentage`, `tract_owner_occupied_units`, `tract_one_to_four_family_homes`, `tract_median_age_of_housing_units`

## Most important inputs (permutation importance, selected model)

| Input | PR-AUC drop when shuffled |
|---|---|
| Debt-to-income ratio | 0.3783 ± 0.0036 |
| Loan purpose | 0.2391 ± 0.0051 |
| Loan amount | 0.2315 ± 0.0027 |
| Property value | 0.2106 ± 0.0036 |
| Combined loan-to-value | 0.1776 ± 0.0040 |
| Loan type | 0.1503 ± 0.0020 |
| Gross annual income | 0.0386 ± 0.0011 |
| Loan term | 0.0216 ± 0.0015 |

## Fairness audit (selected model, test split)

**Race (derived)** (reference group: White)

| Group | n | Observed denial rate | Mean estimate | Unexplained gap | FPR | Recall |
|---|---|---|---|---|---|---|
| 2 or more minority races | 317 | 22.4% | 19.7% | 2.7% | 5.7% | 57.7% |
| American Indian or Alaska Native | 993 | 22.0% | 20.8% | 1.1% | 4.5% | 62.8% |
| Asian | 9,156 | 12.7% | 13.0% | -0.3% | 1.7% | 59.4% |
| Black or African American | 8,601 | 24.5% | 20.6% | 3.9% | 4.5% | 52.7% |
| Joint | 2,574 | 12.8% | 13.4% | -0.6% | 3.4% | 43.8% |
| Native Hawaiian or Other Pacific Islander | 182 | insufficient data | | | | |
| White | 60,098 | 15.4% | 16.6% | -1.2% | 4.4% | 52.5% |
| Free Form Text Only | 22 | insufficient data | | | | |
| Race Not Available | 14,794 | 20.6% | 18.3% | 2.3% | 4.2% | 52.0% |

**Ethnicity (derived)** (reference group: Not Hispanic or Latino)

| Group | n | Observed denial rate | Mean estimate | Unexplained gap | FPR | Recall |
|---|---|---|---|---|---|---|
| Hispanic or Latino | 26,032 | 20.1% | 19.6% | 0.6% | 5.1% | 55.8% |
| Joint | 4,784 | 13.3% | 14.2% | -1.0% | 3.3% | 49.7% |
| Not Hispanic or Latino | 53,997 | 14.9% | 15.7% | -0.8% | 3.7% | 52.2% |
| Ethnicity Not Available | 11,898 | 19.5% | 17.3% | 2.2% | 3.9% | 49.9% |
| Free Form Text Only | 26 | insufficient data | | | | |

**Sex (derived)** (reference group: Joint)

| Group | n | Observed denial rate | Mean estimate | Unexplained gap | FPR | Recall |
|---|---|---|---|---|---|---|
| Female | 19,928 | 20.0% | 19.8% | 0.2% | 5.4% | 56.6% |
| Joint | 36,782 | 13.1% | 14.4% | -1.3% | 3.2% | 51.5% |
| Male | 34,786 | 17.9% | 17.4% | 0.5% | 4.1% | 52.5% |
| Sex Not Available | 5,241 | 23.8% | 19.6% | 4.2% | 5.0% | 48.3% |

**Applicant age** (reference group: 35-44)

| Group | n | Observed denial rate | Mean estimate | Unexplained gap | FPR | Recall |
|---|---|---|---|---|---|---|
| <25 | 4,325 | 10.7% | 12.3% | -1.6% | 1.3% | 46.9% |
| 25-34 | 24,153 | 11.2% | 12.3% | -1.1% | 2.0% | 49.6% |
| 35-44 | 26,319 | 15.8% | 15.8% | 0.1% | 3.5% | 53.5% |
| 45-54 | 19,351 | 19.3% | 18.7% | 0.6% | 4.7% | 52.1% |
| 55-64 | 12,930 | 21.7% | 21.0% | 0.7% | 6.2% | 54.1% |
| 65-74 | 6,995 | 23.2% | 23.2% | 0.1% | 8.5% | 56.4% |
| >74 | 2,590 | 26.8% | 26.4% | 0.3% | 10.9% | 57.7% |
| Not available | 74 | insufficient data | | | | |

- Protected attributes are used only for this audit and are never model inputs.
- Historical disparities are computed on all modelled applications; the model audit uses the held-out test split.
- The 'unexplained gap' is the observed denial rate minus the model's average estimate for the group. Because the model sees only the 13 HMDA application fields (no credit scores, assets or employment), a gap is a signal for further review, not evidence of discrimination, and a small gap is not proof of fairness.
- Groups with fewer than 300 applications are marked as insufficient data.
- Fairness metrics are not the same as legal compliance. Real lending models require fair-lending, model-risk and legal review.

## Promotion gates

| Gate | Result | Detail |
|---|---|---|
| no_banned_inputs | pass | No post-decision, process-artifact, protected or proxy columns are model inputs |
| univariate_leakage_screen | pass | Strongest single input AUC = 0.688 (must be < 0.95) |
| beats_baseline | pass | Test PR-AUC 0.685 vs prevalence baseline 0.168 |
| calibration | pass | Test expected calibration error = 0.0028 (must be < 0.03) |
| cv_stability | pass | CV PR-AUC std = 0.0032 (must be < 0.03) |
| metrics_complete | pass | ROC-AUC, PR-AUC, Brier, ECE and log loss recorded on the test split |
| fairness_report | pass | Group-level audit produced for every protected attribute |
| smoke_predictions | pass | all demo scenarios scored |
| artifact_size | pass | Selected artifact 1.34 MB (budget 60 MB) |
| reference_values | pass | Reference values recorded for every input |

## Limitations and prohibited uses

- HMDA has no credit scores, credit history, assets, employment or cash reserves, which are central to real underwriting. The model reproduces patterns in *historical lender decisions* from limited fields, including any bias in those decisions.
- The outcome is denial versus approval, not default or repayment risk.
- Only complete applications for site-built 1-4 unit homes in Texas (2025) are modelled. Applications with missing fields had very different denial rates and are out of scope.
- Loan amounts and property values are rounded by HMDA to $10,000 bands; DTI is banded.
- Explanations describe how the model responds to inputs. They are not causes, not adverse-action reasons, and not advice on how to get approved.
- **Prohibited:** approving, denying or pricing real applications; marketing or targeting consumers; any use with real personal data.

## Reproduce

```bash
pip install -r requirements.txt && pip install -e .
loan-approval fetch      # downloads the HMDA file and checks its SHA-256
loan-approval prepare
loan-approval train
```

Raw file SHA-256: `3abc1ec4e60ad714a191f3b0b2da91008926a428999a30ddbf2a8a56dc8858da`
