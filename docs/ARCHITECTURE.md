# Architecture

## Components

| Component | Code | Responsibility |
|---|---|---|
| Ingest | `data/ingest.py` | Download from the FFIEC Data Browser, record checksum and provenance, build the processed table |
| Validation | `data/validate.py` | Population filters, completeness rule, model domain, duplicates, quality report |
| Split | `data/split.py` | Deterministic stratified train / calibration / selection / test split and its hash |
| Features | `features/preprocess.py` | Stateless `HmdaFeatureBuilder` + per-algorithm `ColumnTransformer` encodings |
| Schema | `features/schema.py` | Pydantic input contract shared by API, dashboard and smoke tests |
| Training | `training/train.py` | The protocol: tune → fit → calibrate → select → test once → audit → save |
| Evaluation | `evaluation/*` | Metrics + bootstrap CIs, calibration, fairness audit, drift baseline, profile, model card |
| Registry | `registry/artifacts.py` | Versions, SHA-256 verification, lifecycle, promotion gates, audit log |
| Inference | `inference/*` | Cached verified model, prediction, group-Shapley explanations, sensitivity, comparison |
| Services | `services.py` | One JSON-returning layer used by both FastAPI and Streamlit |
| API | `api/*` | Routes, security middleware, rate limits, admin auth, background training jobs |
| Monitoring | `monitoring.py` | SQLite store for ops events, binned drift counts, AI usage, audit log, job state |
| AI | `ai/*` | Analyst loop, read-only evidence tools, guardrails, provider adapter |
| Dashboard | `app/` | Streamlit views and design system; `LocalBackend` or `HttpBackend` (via `API_URL`) |

## Key design decisions

1. **Complete applications only.** Missing fields carry outcome information (a missing loan term
   means 85% denial vs 18%), so imputing them would leak the decision. The model's scope is narrower
   as a result, and the dataset card says so.
2. **Protected attributes are audit-only.** They are joined back to held-out predictions after
   scoring. Geography and lender identity are excluded as proxies.
3. **Selection before testing.** Candidates and thresholds are chosen on a selection split. Every
   candidate is scored on the test split once, and all of them are reported.
4. **Calibration on a separate split.** Isotonic regression is fitted only on the calibration split.
   The calibrated estimate is what users see.
5. **Group Shapley for local explanations.** Exact Shapley values over 8 input groups (256 coalitions,
   about 80 ms) against a typical reference application. Correlated inputs are grouped so the
   explanation never evaluates impossible applications. Contributions sum to the difference in
   estimates.
6. **One service layer.** The API and the dashboard call the same functions, so their numbers match.
7. **Trust boundary at the registry.** Only hash-verified artifacts produced by the pipeline are
   unpickled.
8. **No training in web requests.** Admin training jobs run in a subprocess with a timeout and a
   single-job lock.

## Request flow: `POST /api/v1/predict`

1. Pydantic validates types, enums, ranges and cross-field rules (CLTV ≥ own LTV − 10, loan ≤ 2.5×
   property value). Unknown fields are rejected.
2. The active version is loaded once, verified by SHA-256 and cached by `(models dir, version,
   manifest hash)`.
3. The calibrated pipeline scores the scenario. Group Shapley and percentile context are computed.
4. Monitoring increments binned feature counts for drift and records latency and status. Scenario
   values themselves are not stored.
5. The response includes the version, outcome definition, calibration method, threshold context,
   contributions, warnings, a plain-English interpretation and the disclaimer.
