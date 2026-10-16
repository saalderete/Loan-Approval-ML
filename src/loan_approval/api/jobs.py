"""Bounded training jobs, run out-of-process so the web worker never blocks or trains in-request."""

from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from loan_approval import monitoring
from loan_approval.config import ROOT, get_settings

JOB_TIMEOUT_SECONDS = 3600
_RUN_LOCK = threading.Lock()


class TrainJobRequest(BaseModel):
    dataset: Literal["hmda", "synthetic"] = "synthetic"
    rows: int = Field(20000, ge=2000, le=200000, description="Synthetic rows (ignored for hmda)")
    algorithms: list[Literal["logistic_regression", "random_forest", "hist_gradient_boosting"]] = Field(
        default_factory=lambda: ["logistic_regression", "random_forest", "hist_gradient_boosting"],
        min_length=1,
    )
    fast: bool = False
    reason: str = Field("", max_length=300)


def _command(req: TrainJobRequest) -> list[str]:
    cmd = [sys.executable, "-m", "loan_approval", "train", "--algorithms", ",".join(req.algorithms)]
    if req.dataset == "synthetic":
        cmd += ["--synthetic", "--rows", str(req.rows)]
        if req.fast:
            cmd.append("--fast")
    return cmd


def _run(job_id: str, req: TrainJobRequest) -> None:
    if not _RUN_LOCK.acquire(blocking=False):
        monitoring.update_job(job_id, "rejected", error="Another training job is already running.")
        return
    try:
        monitoring.update_job(job_id, "training")
        log_dir = get_settings().state_dir / "jobs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f"{job_id}.log").open("w") as log:
            proc = subprocess.run(
                _command(req),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=JOB_TIMEOUT_SECONDS,
                check=False,
            )
        lines = (log_dir / f"{job_id}.log").read_text().strip().splitlines()
        if proc.returncode == 0 and lines:
            monitoring.update_job(job_id, "evaluated", version=lines[-1].strip())
        else:
            monitoring.update_job(job_id, "failed", error=f"exit code {proc.returncode}")
    except subprocess.TimeoutExpired:
        monitoring.update_job(job_id, "failed", error=f"timed out after {JOB_TIMEOUT_SECONDS}s")
    except Exception as exc:
        monitoring.update_job(job_id, "failed", error=type(exc).__name__)
    finally:
        _RUN_LOCK.release()


def submit(req: TrainJobRequest, actor: str, idempotency_key: str | None, run_async: bool = True) -> dict:
    job = monitoring.create_job(str(uuid.uuid4()), actor, req.model_dump(), idempotency_key)
    if job.get("duplicate"):
        return job
    monitoring.audit(actor, "train_job_submitted", {"job_id": job["id"], "request": req.model_dump()})
    if run_async:
        threading.Thread(target=_run, args=(job["id"], req), daemon=True).start()
    else:
        _run(job["id"], req)
    return monitoring.get_job(job["id"])
