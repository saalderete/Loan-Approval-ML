# Dataset card: HMDA 2025, Texas (approved or denied applications)

## Source

| Field | Value |
|---|---|
| Title | Home Mortgage Disclosure Act (HMDA) Loan/Application Register, public "modified LAR" |
| Publisher | Federal Financial Institutions Examination Council (FFIEC), on behalf of the CFPB and other agencies |
| Access | FFIEC Data Browser API: `https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=TX&years=2025&actions_taken=1,2,3` |
| Field documentation | https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields |
| Retrieved | 2026-10-16 (see `data/data_manifest.yaml`) |
| Raw file | `hmda_2025_TX_actions_1_2_3.csv`: 749,111 rows, 99 columns, 283.7 MB |
| SHA-256 | `3abc1ec4e60ad714a191f3b0b2da91008926a428999a30ddbf2a8a56dc8858da` |
| License / terms | Public data released by U.S. federal agencies under HMDA for public use. The raw file is **not** committed to this repository; `loan-approval fetch` downloads it and checks the checksum. |

HMDA requires most U.S. mortgage lenders to report every application they receive. The public
version is modified to protect applicant privacy: there are no names or exact dates, amounts are
rounded to $10,000 bands, ages are banded, and debt-to-income is banded outside 36–49%.

## Unit of analysis and target

- **Unit:** one mortgage application reported by one lender in 2025 for a property in Texas.
- **Target:** `denied` = 1 if `action_taken` = 3 (application denied), 0 if `action_taken` is
  1 (originated) or 2 (approved but not accepted). Withdrawn, incomplete and purchased loans are not
  in the file because the query requests action codes 1–3 only.
- This is a **historical lender-decision** label. It is not default, delinquency or repayment risk.

## Population filters

| Stage | Rule | Rows | Denial rate |
|---|---|---|---|
| raw | Approved or denied, Texas 2025 | 749,111 | 24.9% |
| consumer_closed_end | Closed-end, non-reverse, not for a business purpose | 622,545 | 23.3% |
| site_built_1_4 | Site-built, 1–4 unit homes, known loan purpose | 551,627 | 18.4% |
| complete | All 13 model inputs reported (no `NA` / `Exempt`) | 493,391 | 16.9% |
| in_domain | Values inside the documented model domain | 487,042 | 16.7% |
| deduplicated | Exact duplicate records removed (3,357) | **483,685** | **16.8%** |

Outcome counts in the modelled data: 386,871 originated, 15,511 approved but not accepted,
81,303 denied.

## Missing values and selection bias

Missing fields are strongly associated with denial. Lenders leave a field blank when the
application never reached that step:

| Input | Missing | Denied if missing | Denied if present |
|---|---|---|---|
| Loan term | 0.7% | 85.1% | 18.0% |
| Property value | 3.9% | 45.7% | 17.3% |
| Combined LTV | 5.7% | 29.6% | 17.8% |
| Debt-to-income | 6.5% | 27.2% | 17.8% |
| Income | 5.7% | 20.8% | 18.3% |

Imputing these values would let the model learn the decision from the *pattern of blanks*. The model
is therefore trained only on complete applications. Its scope is limited to applications where
lenders reported every input, and it should not be applied to incomplete ones.

## Fields

- **Model inputs (13):** loan amount, property value, income, debt-to-income ratio, combined
  loan-to-value ratio, loan term, loan type, loan purpose, lien status, occupancy, total units,
  preapproval, conforming-loan-limit flag.
- **Fairness audit only (never model inputs):** derived race, derived ethnicity, derived sex,
  applicant age.
- **Excluded as post-decision:** interest rate, rate spread, HOEPA status, loan costs, points and
  fees, origination charges, discount points, lender credits, prepayment penalty, intro-rate period,
  purchaser type, denial reasons.
- **Excluded as process artifacts:** credit-score model type, automated-underwriting system,
  submission channel, initially-payable flag. Their "not applicable" codes reveal how far an
  application progressed.
- **Excluded for proxy risk:** lender identifier (LEI), census tract, county, MSA, and tract
  demographics (minority share, income ratio, housing stock).

## Known biases and gaps

- No credit scores, credit history, assets, cash reserves or employment. These drive real
  underwriting decisions, so the model cannot separate them from the inputs it does see.
- Historical decisions can encode discrimination. The fairness audit measures disparities but cannot
  establish their cause.
- Demographic fields are self-reported, observed or "not available"; 15% of modelled applications
  have race not available.
- Texas only, a single year, and lender mix varies by market.
- Repeated applications by the same person to several lenders cannot be identified, because HMDA has
  no applicant ID. Exact duplicates are removed, but near-duplicates may remain across splits.

## Intended and prohibited uses

- **Intended:** education, portfolio demonstration, methodology research, fair-lending analytics
  practice on public data.
- **Prohibited:** approving, denying, pricing or pre-screening real applications; marketing or
  targeting; re-identification attempts; use with real personal data.

## Splits

Stratified by outcome with seed 20251: train 60%, calibration 10%, selection 10%, test 20%. The
assignment hash is stored in every model manifest.
