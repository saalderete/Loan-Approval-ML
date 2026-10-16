"""FastAPI service. Run with: ``uvicorn loan_approval.api.app:app``."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, ValidationError

from loan_approval import __version__, monitoring, services
from loan_approval.ai import analyst
from loan_approval.api import jobs
from loan_approval.api.security import (
    BodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
    rate_limit,
    require_admin,
)
from loan_approval.features.schema import ScenarioInput
from loan_approval.inference.predict import ModelUnavailable
from loan_approval.registry import artifacts

log = logging.getLogger("loan_approval.api")


class PredictRequest(BaseModel):
    scenario: ScenarioInput
    algorithm: str | None = Field(None, description="Defaults to the selected model")


class ExplainRequest(BaseModel):
    scenario: ScenarioInput
    feature: str
    algorithm: str | None = None


class CompareRequest(BaseModel):
    scenario: ScenarioInput


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    scenario_result: dict[str, Any] | None = Field(
        None, description="A prediction returned by /predict, used as evidence"
    )


class PromoteRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=300)


def _validation_detail(exc: ValidationError | RequestValidationError) -> list[dict]:
    return [
        {
            "field": ".".join(str(p) for p in e.get("loc", []) if p not in ("body", "scenario")),
            "message": e.get("msg", "invalid value"),
        }
        for e in exc.errors()
    ]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Loan Approval ML API",
        version=__version__,
        description="Educational research API over a model trained on public 2025 Texas HMDA data. "
        "Outputs are not lending decisions. Only fictional scenarios are accepted.",
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(BodySizeLimitMiddleware)

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        log.info(
            json.dumps(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round((time.perf_counter() - start) * 1000, 1),
                }
            )
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def _request_invalid(request: Request, exc: RequestValidationError):
        if request.url.path.endswith(("/predict", "/explain", "/compare")):
            monitoring.record_inference(request.url.path.rsplit("/", 1)[-1], "invalid", None, "api")
        return JSONResponse({"detail": _validation_detail(exc)}, status_code=422)

    @app.exception_handler(ValidationError)
    async def _invalid(request: Request, exc: ValidationError):
        return JSONResponse({"detail": _validation_detail(exc)}, status_code=422)

    @app.exception_handler(ModelUnavailable)
    async def _unavailable(request: Request, exc: ModelUnavailable):
        return JSONResponse({"detail": str(exc), "state": "setup_required"}, status_code=503)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.exception("Unhandled error on %s", request.url.path)
        return JSONResponse({"detail": "Internal error."}, status_code=500)

    # --- health -----------------------------------------------------------------------------
    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["health"])
    def ready():
        info = services.status()
        if not info["ready"]:
            return JSONResponse({"ready": False, "detail": info.get("error", "No active model")}, 503)
        return {"ready": True, "model_version": info["active_version"]}

    # --- public read-only -----------------------------------------------------------------------
    @app.get("/api/v1/model", tags=["model"])
    def model():
        return services.model_info()

    @app.get("/api/v1/metrics", tags=["model"])
    def metrics():
        return services.report("metrics")

    @app.get("/api/v1/reports/{name}", tags=["model"])
    def report(name: str):
        try:
            return services.report(name)
        except KeyError as exc:
            raise HTTPException(404, "Unknown report") from exc

    @app.get("/api/v1/model-card", tags=["model"], response_class=PlainTextResponse)
    def model_card():
        return services.model_card()

    @app.get("/api/v1/dataset-card", tags=["model"], response_class=PlainTextResponse)
    def dataset_card():
        return services.dataset_card()

    @app.get("/api/v1/schema", tags=["model"])
    def schema():
        return services.input_schema()

    @app.get("/api/v1/demo-scenarios", tags=["scenarios"])
    def demo_scenarios():
        return {"scenarios": services.demo_scenarios()}

    @app.get("/api/v1/experiments", tags=["research"])
    def experiments():
        return services.experiments()

    @app.get("/api/v1/monitoring", tags=["research"])
    def monitoring_report():
        return services.monitoring_report()

    # --- scenario scoring (rate limited) --------------------------------------------------------
    @app.post("/api/v1/predict", tags=["scenarios"], dependencies=[Depends(rate_limit("predict"))])
    def predict(body: PredictRequest):
        return services.predict(body.scenario.model_dump(), body.algorithm, source="api")

    @app.post("/api/v1/explain", tags=["scenarios"], dependencies=[Depends(rate_limit("explain"))])
    def explain(body: ExplainRequest):
        try:
            return services.explain(body.scenario.model_dump(), body.feature, body.algorithm, source="api")
        except KeyError as exc:
            raise HTTPException(422, "Unknown feature") from exc

    @app.post("/api/v1/compare", tags=["scenarios"], dependencies=[Depends(rate_limit("compare"))])
    def compare(body: CompareRequest):
        return services.compare(body.scenario.model_dump(), source="api")

    @app.post("/api/v1/ai/ask", tags=["ai"], dependencies=[Depends(rate_limit("ai", per_minute=10))])
    def ask(body: AskRequest):
        return analyst.ask(body.question, scenario_result=body.scenario_result)

    # --- admin ------------------------------------------------------------------------------
    @app.post("/api/v1/train-jobs", tags=["admin"], status_code=202)
    def create_train_job(
        body: jobs.TrainJobRequest,
        actor: str = Depends(require_admin),
        idempotency_key: str | None = Header(default=None, max_length=100),
    ):
        return jobs.submit(body, actor, idempotency_key)

    @app.get("/api/v1/train-jobs", tags=["admin"])
    def list_train_jobs(actor: str = Depends(require_admin)):
        return {"jobs": monitoring.list_jobs()}

    @app.get("/api/v1/train-jobs/{job_id}", tags=["admin"])
    def get_train_job(job_id: str, actor: str = Depends(require_admin)):
        try:
            return monitoring.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown job") from exc

    @app.post("/api/v1/models/{version}/promote", tags=["admin"])
    def promote(version: str, body: PromoteRequest, actor: str = Depends(require_admin)):
        try:
            result = artifacts.promote(version, actor=actor, reason=body.reason)
        except artifacts.ArtifactError as exc:
            raise HTTPException(409, str(exc)) from exc
        monitoring.audit(actor, "promote", {"version": version, "reason": body.reason})
        return result

    return app


app = create_app()
