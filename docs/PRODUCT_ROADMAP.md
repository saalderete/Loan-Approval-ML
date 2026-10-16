# Product roadmap

| Phase | Status |
|---|---|
| 0. Clean repo, pinned deps, tooling, CI | Done |
| 1. HMDA dataset, validation, leakage-safe baseline, real metrics | Done |
| 2. Model comparison, calibration, explainability, fairness audit, model card | Done |
| 3. Premium Streamlit dashboard | Done |
| 4. FastAPI, Docker, automated tests | Done |
| 5. AI Model Analyst, drift monitoring, experiment history | Done (LLM path tested with a scripted provider only) |
| 6. Public demo hosting, screenshots/video walkthrough | Screenshots done; hosting not yet |

## Next ideas

- Host the dashboard publicly (Streamlit Community Cloud) and link it from the README.
- Add 2024 data to test temporal stability: train on 2024, test on 2025.
- Multi-state comparison and lender-level benchmarking (aggregated, not individual lenders by name).
- Matched-pair or regression-adjusted disparity analysis for the fairness page.
- Optional React/Next.js front end over the existing API.
- Synthetic paystub extraction demo (clearly watermarked, user-confirmed, never real documents).
