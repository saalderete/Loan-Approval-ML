# Data governance

| Topic | Policy |
|---|---|
| Source | Public HMDA modified LAR from the FFIEC Data Browser. Provenance and checksum are in `data/data_manifest.yaml`. |
| Storage | Raw and processed files stay local (`data/raw`, `data/processed`, gitignored). Only aggregates (profile, metrics, fairness tables) are committed with model versions. |
| Personal data | The public LAR is already de-identified by the agencies. The app never accepts names, SSNs, addresses or account numbers. The AI analyst refuses questions that contain identifiers. |
| Protected attributes | Used only in the fairness audit, joined after scoring. Never used as model inputs, never shown at row level. |
| Small groups | Groups with fewer than 300 test applications are suppressed in model-level tables. |
| Demo inputs | Ephemeral. Scenario values are not stored; monitoring keeps binned counts per feature for drift only. |
| Re-identification | Prohibited. Do not join HMDA rows with other data to identify applicants. |
| Retention | Monitoring data lives in `var/loan_approval.db`. Delete the file to reset it. |
| Redistribution | Check the current FFIEC terms before publishing raw files. The repo publishes only code, aggregates and model artifacts. |
