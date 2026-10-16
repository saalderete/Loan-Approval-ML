"""Operational monitoring and audit storage (SQLite).

Privacy: scenario inputs are never stored. For drift we only increment aggregate counters per
(model version, feature, bin), and inference events record timing and status, not content.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime

import numpy as np

from loan_approval.config import get_settings
from loan_approval.evaluation import drift

_LOCK = threading.RLock()
SCHEMA = """
CREATE TABLE IF NOT EXISTS inference_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    model_version TEXT,
    endpoint TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms REAL
);
CREATE TABLE IF NOT EXISTS feature_bins (
    model_version TEXT NOT NULL,
    feature TEXT NOT NULL,
    bin TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (model_version, feature, bin)
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS ai_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    day TEXT NOT NULL,
    status TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    tools TEXT
);
CREATE TABLE IF NOT EXISTS train_jobs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    actor TEXT NOT NULL,
    request TEXT NOT NULL,
    version TEXT,
    error TEXT,
    idempotency_key TEXT UNIQUE
);
"""


@contextmanager
def connect():
    path = get_settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        conn = sqlite3.connect(path, timeout=10)
        try:
            conn.executescript(SCHEMA)
            yield conn
            conn.commit()
        finally:
            conn.close()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def record_inference(
    endpoint: str,
    status: str,
    latency_ms: float | None,
    source: str,
    model_version: str | None = None,
    inputs: dict | None = None,
    baseline: dict | None = None,
) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO inference_events (ts, model_version, endpoint, source, status, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), model_version, endpoint, source, status, latency_ms),
        )
        if inputs and baseline and model_version:
            for feature, spec in baseline["features"].items():
                if feature not in inputs:
                    continue
                conn.execute(
                    "INSERT INTO feature_bins (model_version, feature, bin, count) VALUES (?, ?, ?, 1) "
                    "ON CONFLICT(model_version, feature, bin) DO UPDATE SET count = count + 1",
                    (model_version, feature, drift.bin_value(spec, inputs[feature])),
                )


def observed_bins(model_version: str) -> dict[str, dict[str, int]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT feature, bin, count FROM feature_bins WHERE model_version = ?", (model_version,)
        ).fetchall()
    out: dict[str, dict[str, int]] = {}
    for feature, b, c in rows:
        out.setdefault(feature, {})[b] = c
    return out


def drift_for(model_version: str, baseline: dict) -> dict:
    return drift.drift_report(baseline, observed_bins(model_version))


def operational_stats(window_hours: int = 24 * 7) -> dict:
    since = time.time() - window_hours * 3600
    with connect() as conn:
        rows = conn.execute(
            "SELECT ts, endpoint, source, status, latency_ms FROM inference_events WHERE ts >= ?",
            (since,),
        ).fetchall()
    total = len(rows)
    by_status: dict[str, int] = {}
    by_endpoint: dict[str, int] = {}
    latencies = []
    hourly: dict[str, int] = {}
    for ts, endpoint, _source, status, latency in rows:
        by_status[status] = by_status.get(status, 0) + 1
        by_endpoint[endpoint] = by_endpoint.get(endpoint, 0) + 1
        if latency is not None and status == "ok":
            latencies.append(latency)
        hour = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:00")
        hourly[hour] = hourly.get(hour, 0) + 1
    return {
        "window_hours": window_hours,
        "requests": total,
        "by_status": by_status,
        "by_endpoint": by_endpoint,
        "invalid_rate": (by_status.get("invalid", 0) / total) if total else None,
        "error_rate": (by_status.get("error", 0) / total) if total else None,
        "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
        "hourly": [{"hour": k, "requests": v} for k, v in sorted(hourly.items())],
    }


def audit(actor: str, action: str, detail: dict | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO audit_log (ts, actor, action, detail) VALUES (?, ?, ?, ?)",
            (now_iso(), actor, action, json.dumps(detail or {})),
        )


def audit_entries(limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT ts, actor, action, detail FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{"ts": t, "actor": a, "action": act, "detail": json.loads(d or "{}")} for t, a, act, d in rows]


def record_ai_usage(status: str, input_tokens: int = 0, output_tokens: int = 0, tools=()) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO ai_usage (ts, day, status, input_tokens, output_tokens, tools) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                now_iso(),
                datetime.now(UTC).date().isoformat(),
                status,
                input_tokens,
                output_tokens,
                json.dumps(list(tools)),
            ),
        )


def ai_requests_today() -> int:
    with connect() as conn:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM ai_usage WHERE day = ? AND status != 'refused_scope'",
            (datetime.now(UTC).date().isoformat(),),
        ).fetchone()
    return int(count)


def ai_usage_summary() -> dict:
    with connect() as conn:
        rows = conn.execute(
            "SELECT day, status, COUNT(*), SUM(input_tokens), SUM(output_tokens) FROM ai_usage "
            "GROUP BY day, status ORDER BY day DESC LIMIT 60"
        ).fetchall()
    return {
        "rows": [
            {"day": d, "status": s, "requests": n, "input_tokens": i or 0, "output_tokens": o or 0}
            for d, s, n, i, o in rows
        ]
    }


# --- training jobs ---------------------------------------------------------------------------


def create_job(job_id: str, actor: str, request: dict, idempotency_key: str | None) -> dict:
    with connect() as conn:
        if idempotency_key:
            row = conn.execute(
                "SELECT id FROM train_jobs WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if row:
                return get_job(row[0], conn) | {"duplicate": True}
        ts = now_iso()
        conn.execute(
            "INSERT INTO train_jobs (id, created_at, updated_at, status, actor, request, idempotency_key) "
            "VALUES (?, ?, ?, 'queued', ?, ?, ?)",
            (job_id, ts, ts, actor, json.dumps(request), idempotency_key),
        )
        return get_job(job_id, conn)


def update_job(job_id: str, status: str, version: str | None = None, error: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE train_jobs SET status = ?, updated_at = ?, version = COALESCE(?, version), "
            "error = ? WHERE id = ?",
            (status, now_iso(), version, error, job_id),
        )


def get_job(job_id: str, conn: sqlite3.Connection | None = None) -> dict:
    def _read(c):
        row = c.execute(
            "SELECT id, created_at, updated_at, status, actor, request, version, error "
            "FROM train_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise KeyError(job_id)
        keys = ["id", "created_at", "updated_at", "status", "actor", "request", "version", "error"]
        out = dict(zip(keys, row, strict=True))
        out["request"] = json.loads(out["request"])
        return out

    if conn is not None:
        return _read(conn)
    with connect() as c:
        return _read(c)


def list_jobs(limit: int = 20) -> list[dict]:
    with connect() as conn:
        ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM train_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        ]
        return [get_job(i, conn) for i in ids]
