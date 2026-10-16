# Current state

_Last updated: 2026-10-16_

This file separates what is **verified and working** from **assumptions** and **future plans**, as
the development handoff required.

## Verified (built and tested in this repository)

| Area | Status | Evidence |
|---|---|---|
| Data | 2025 Texas HMDA file downloaded (749,111 rows, SHA-256 recorded); 483,685 applications modelled after documented filters | `data/data_manifest.yaml`, `reports/dataset_card.md` |
| Pipeline | Leakage-safe features, fixed stratified split with recorded hash, CV tuning on train only, isotonic calibration on its own split, selection before test | `src/loan_approval/training/train.py` |
| Models | Logistic regression, random forest and HistGradientBoosting, all scored on the same test split; HistGB selected (test ROC-AUC 0.857, PR-AUC 0.685, ECE 0.0028) | generated model card in `models/` |
| Fairness | Historical and model-level audit by race, ethnicity, sex and age with Wilson CIs and small-group suppression | `reports/fairness.json` in the model version |
| Explainability | Permutation importance, logistic coefficients, exact group-Shapley local explanations (sum verified in tests), sensitivity curves | `src/loan_approval/inference/explain.py` |
| Registry | Versioned artifacts with SHA-256, 10 promotion gates, audited promotion, tamper detection tests | `src/loan_approval/registry/artifacts.py` |
| API | FastAPI `/api/v1` with validation, rate limits, body limit, security headers, admin token, idempotent out-of-process training jobs | `tests/api/test_api.py` |
| Dashboard | 10-page Streamlit app, light/dark themes, in-process or API-backed | `tests/ui/test_streamlit_pages.py` |
| AI analyst | Read-only tools, citation and number grounding checks, scope/PII refusals, graceful failure; deterministic quick answers | `tests/unit/test_analyst.py` (scripted provider) |
| Monitoring | Ops stats, input-drift PSI per feature, AI usage; aggregate-only storage | `src/loan_approval/monitoring.py` |
| Quality | 95 tests passing on Python 3.12 and 3.13; ruff clean; `pip-audit` clean; Docker image built and smoke-tested. All green in GitHub Actions (run 38974407815) | `.github/workflows/ci.yml` |

## Not verified / assumptions

- The **LLM analyst** has been tested only with a scripted provider. It has not been run against a live
  Anthropic API key in this environment.
- **Hosting** (Streamlit Community Cloud or a container host) is not set up.
- HMDA terms of use were summarised from the FFIEC site; re-check them before redistributing raw files.

## Future plans (not implemented)

See [PRODUCT_ROADMAP.md](PRODUCT_ROADMAP.md): public hosting, multi-year and multi-state data,
lender-level benchmarking, React frontend, synthetic-document extraction demo.
