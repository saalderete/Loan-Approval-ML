import time

import pytest
from fastapi.testclient import TestClient

from loan_approval.api.app import create_app
from loan_approval.api.security import limiter


@pytest.fixture
def client(trained_env, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    limiter.reset()
    return TestClient(create_app())


def test_health_and_ready(client, trained_env):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready").json()
    assert ready["ready"] and ready["model_version"] == trained_env["version"]


def test_public_read_endpoints(client):
    assert client.get("/api/v1/model").json()["input_schema"]["fields"]
    assert "test" in client.get("/api/v1/metrics").json()
    assert client.get("/api/v1/reports/fairness").status_code == 200
    assert client.get("/api/v1/reports/secrets").status_code == 404
    assert len(client.get("/api/v1/demo-scenarios").json()["scenarios"]) >= 5
    assert client.get("/api/v1/model-card").text.startswith("# Model card")
    assert client.get("/api/v1/experiments").json()["active_version"]


def test_predict_contract(client, demo_inputs):
    body = client.post("/api/v1/predict", json={"scenario": demo_inputs}).json()
    for key in (
        "prediction_id",
        "model_version",
        "estimated_probability",
        "calibrated",
        "top_contributors",
        "outcome_definition",
        "disclaimer",
        "interpretation",
        "warnings",
    ):
        assert key in body
    assert body["scenario_type"] == "synthetic"
    assert "approved" not in body["interpretation"].lower().split()


def test_invalid_input_returns_readable_422(client, demo_inputs):
    resp = client.post("/api/v1/predict", json={"scenario": {**demo_inputs, "loan_amount": -5}})
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["field"] == "loan_amount"
    resp = client.post("/api/v1/predict", json={"scenario": {**demo_inputs, "ssn": "123"}})
    assert resp.status_code == 422


def test_explain_and_compare(client, demo_inputs):
    exp = client.post("/api/v1/explain", json={"scenario": demo_inputs, "feature": "income"}).json()
    assert exp["points"] and "not advice" in exp["note"]
    assert (
        client.post("/api/v1/explain", json={"scenario": demo_inputs, "feature": "race"}).status_code == 422
    )
    cmp = client.post("/api/v1/compare", json={"scenario": demo_inputs}).json()
    assert len(cmp["results"]) >= 2


def test_admin_endpoints_disabled_without_token(client):
    assert client.post("/api/v1/train-jobs", json={}).status_code == 403
    assert client.post("/api/v1/models/x/promote", json={"reason": "abc"}).status_code == 403


def test_admin_requires_correct_token(trained_env, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret-token")
    c = TestClient(create_app())
    assert c.get("/api/v1/train-jobs").status_code == 401
    assert c.get("/api/v1/train-jobs", headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert c.get("/api/v1/train-jobs", headers={"X-Admin-Token": "s3cret-token"}).status_code == 200
    resp = c.post(
        f"/api/v1/models/{trained_env['version']}/promote",
        json={"reason": "re-promote in test"},
        headers={"X-Admin-Token": "s3cret-token"},
    )
    assert resp.status_code == 200
    assert (
        c.post(
            "/api/v1/models/unknown/promote",
            json={"reason": "nope"},
            headers={"X-Admin-Token": "s3cret-token"},
        ).status_code
        == 409
    )


@pytest.mark.slow
def test_training_job_runs_out_of_process_and_is_idempotent(trained_env, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "tok")
    c = TestClient(create_app())
    headers = {"X-Admin-Token": "tok", "Idempotency-Key": "job-1"}
    body = {"dataset": "synthetic", "rows": 2000, "fast": True, "algorithms": ["logistic_regression"]}
    job = c.post("/api/v1/train-jobs", json=body, headers=headers)
    assert job.status_code == 202
    again = c.post("/api/v1/train-jobs", json=body, headers=headers).json()
    assert again["id"] == job.json()["id"]
    deadline = time.time() + 240
    status = None
    while time.time() < deadline:
        status = c.get(f"/api/v1/train-jobs/{job.json()['id']}", headers=headers).json()
        if status["status"] in ("evaluated", "failed", "rejected"):
            break
        time.sleep(2)
    assert status["status"] == "evaluated", status
    assert status["version"].startswith("synthetic-")


def test_rate_limit(client, demo_inputs, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    codes = [client.post("/api/v1/predict", json={"scenario": demo_inputs}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200]
    assert codes[-1] == 429


def test_body_size_limit_and_security_headers(client):
    resp = client.post("/api/v1/predict", content=b"x" * 20000, headers={"content-type": "application/json"})
    assert resp.status_code == 413
    headers = client.get("/health").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"


def test_ai_endpoint_disabled_and_refusal(client):
    assert (
        client.post("/api/v1/ai/ask", json={"question": "Compare the models"}).json()["status"] == "disabled"
    )
    refused = client.post("/api/v1/ai/ask", json={"question": "Should I approve this applicant?"}).json()
    assert refused["status"] == "refused"


def test_monitoring_records_aggregates_only(client, demo_inputs, trained_env):
    client.post("/api/v1/predict", json={"scenario": demo_inputs})
    report = client.get("/api/v1/monitoring").json()
    assert report["operations"]["requests"] >= 1
    import sqlite3

    conn = sqlite3.connect(trained_env["state"] / "loan_approval.db")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(inference_events)")]
    conn.close()
    assert not {"loan_amount", "income", "inputs"} & set(cols)
