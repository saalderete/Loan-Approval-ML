# Runbook

## Local

```bash
pip install -r requirements.txt && pip install -e .
streamlit run app/streamlit_app.py
uvicorn loan_approval.api.app:app --port 8000
```

## Retrain on new HMDA data

```bash
loan-approval fetch --year 2025 --state TX --force   # checksum mismatch warnings mean HMDA republished the file
loan-approval prepare
loan-approval train
loan-approval list                                   # new version shows "review_required" if every gate passed
# Read models/<version>/model_card.md, check gates, fairness and limitations, then:
loan-approval promote <version> --actor "Your Name" --reason "Why this version"
git add models/ && git commit -m "Promote <version>"
```

Rollback: promote the previous version again. The registry audit log records both actions.

## Admin API

```bash
export ADMIN_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
curl -X POST localhost:8000/api/v1/train-jobs -H "X-Admin-Token: $ADMIN_TOKEN" \
     -H "Idempotency-Key: nightly-2026-10-16" -H 'Content-Type: application/json' \
     -d '{"dataset": "hmda", "algorithms": ["hist_gradient_boosting"]}'
curl localhost:8000/api/v1/train-jobs -H "X-Admin-Token: $ADMIN_TOKEN"
curl -X POST localhost:8000/api/v1/models/<version>/promote -H "X-Admin-Token: $ADMIN_TOKEN" \
     -H 'Content-Type: application/json' -d '{"reason": "Reviewed model card"}'
```

## Health and troubleshooting

| Symptom | Check |
|---|---|
| `/ready` returns 503 | No active version, or a checksum failed. Run `loan-approval list`; re-promote or retrain. Never hand-edit artifacts. |
| Dashboard shows "No approved model yet" | Same as above, or `LOAN_ML_MODELS_DIR` points to the wrong directory. |
| 429 responses | Rate limit (`RATE_LIMIT_PER_MINUTE`, default 60; AI 10/min). |
| AI analyst "disabled" | `ANTHROPIC_API_KEY` is not set on the server. |
| Drift alerts | Monitoring → input drift. PSI ≥ 0.25 means demo inputs differ a lot from training; it does not measure accuracy. |

## Deploy

- **Streamlit Community Cloud:** main file `app/streamlit_app.py`, Python 3.12. It uses `requirements.txt`
  and the committed model. Add `ANTHROPIC_API_KEY` as a secret only if you want the LLM analyst.
- **Containers:** `docker compose up --build`. The API runs as a non-root user with a healthcheck, and
  the dashboard calls it through `API_URL`. Put both behind HTTPS and keep `ADMIN_TOKEN` unset on
  public deployments unless you need remote training.
