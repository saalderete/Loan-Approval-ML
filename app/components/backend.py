"""Data access for the dashboard.

By default the dashboard calls the same service layer as the API, in-process (works on Streamlit
Community Cloud). Set ``API_URL`` to use a running FastAPI service instead (Docker Compose does this).
Both return identical JSON.
"""

from __future__ import annotations

import httpx
import streamlit as st

from loan_approval import services
from loan_approval.ai import analyst
from loan_approval.config import get_settings
from loan_approval.inference.predict import ModelUnavailable


class BackendError(RuntimeError):
    """Readable error for the UI (validation, missing model, backend down)."""

    def __init__(self, message: str, kind: str = "error", details: list | None = None):
        super().__init__(message)
        self.kind = kind
        self.details = details or []


class LocalBackend:
    mode = "in-process"

    def status(self) -> dict:
        return services.status()

    def model_info(self) -> dict:
        return self._guard(services.model_info)

    def report(self, name: str) -> dict:
        return self._guard(lambda: services.report(name))

    def model_card(self) -> str:
        return self._guard(services.model_card)

    def dataset_card(self) -> str:
        return services.dataset_card()

    def demo_scenarios(self) -> list[dict]:
        return services.demo_scenarios()

    def schema(self) -> dict:
        return services.input_schema()

    def experiments(self) -> dict:
        return services.experiments()

    def monitoring(self) -> dict:
        return services.monitoring_report()

    def predict(self, inputs: dict, algorithm: str | None = None) -> dict:
        return self._guard(lambda: services.predict(inputs, algorithm, source="dashboard"))

    def explain(self, inputs: dict, feature: str, algorithm: str | None = None) -> dict:
        return self._guard(lambda: services.explain(inputs, feature, algorithm, source="dashboard"))

    def compare(self, inputs: dict) -> dict:
        return self._guard(lambda: services.compare(inputs, source="dashboard"))

    def ask(self, question: str, scenario_result: dict | None = None) -> dict:
        return analyst.ask(question, scenario_result=scenario_result)

    def quick_answer(self, kind: str) -> dict:
        return self._guard(lambda: analyst.quick_answer(kind))

    @staticmethod
    def _guard(fn):
        try:
            return fn()
        except ModelUnavailable as exc:
            raise BackendError(str(exc), kind="setup") from exc
        except services.ValidationError as exc:
            details = [
                {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()
            ]
            raise BackendError("Some inputs are not valid.", kind="invalid", details=details) from exc


class HttpBackend:
    mode = "API"

    def __init__(self, base_url: str) -> None:
        self.base = base_url
        self.client = httpx.Client(base_url=base_url, timeout=30)

    def _call(self, method: str, path: str, **kwargs):
        try:
            resp = self.client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise BackendError(f"The API at {self.base} is not reachable.", kind="backend") from exc
        if resp.status_code == 503:
            raise BackendError(resp.json().get("detail", "No model available."), kind="setup")
        if resp.status_code == 422:
            raise BackendError(
                "Some inputs are not valid.", kind="invalid", details=resp.json().get("detail", [])
            )
        if resp.status_code == 429:
            raise BackendError("Too many requests. Please wait a moment.", kind="rate_limit")
        if resp.status_code >= 400:
            raise BackendError(f"The API returned an error ({resp.status_code}).", kind="backend")
        return resp.json() if "json" in resp.headers.get("content-type", "") else resp.text

    def status(self) -> dict:
        try:
            ready = self.client.get("/ready")
            if ready.status_code != 200:
                return {"ready": False, "active_version": None, "error": ready.json().get("detail")}
            model = self._call("GET", "/api/v1/model")
            return {
                "ready": True,
                "active_version": model["version"],
                "selected_algorithm": model["selected_algorithm"],
                "algorithm_name": model["algorithms"][model["selected_algorithm"]],
                "dataset": model["dataset"],
                "headline_metrics": model["headline_metrics"],
                "created_at": model["created_at"],
                "outcome_definition": model["outcome_definition"],
                "available_algorithms": list(model["artifacts"]),
            }
        except httpx.HTTPError:
            return {"ready": False, "active_version": None, "error": f"API at {self.base} is not reachable."}

    def model_info(self) -> dict:
        return self._call("GET", "/api/v1/model")

    def report(self, name: str) -> dict:
        return self._call("GET", f"/api/v1/reports/{name}")

    def model_card(self) -> str:
        return self._call("GET", "/api/v1/model-card")

    def dataset_card(self) -> str:
        return self._call("GET", "/api/v1/dataset-card")

    def demo_scenarios(self) -> list[dict]:
        return self._call("GET", "/api/v1/demo-scenarios")["scenarios"]

    def schema(self) -> dict:
        return self._call("GET", "/api/v1/schema")

    def experiments(self) -> dict:
        return self._call("GET", "/api/v1/experiments")

    def monitoring(self) -> dict:
        return self._call("GET", "/api/v1/monitoring")

    def predict(self, inputs: dict, algorithm: str | None = None) -> dict:
        return self._call("POST", "/api/v1/predict", json={"scenario": inputs, "algorithm": algorithm})

    def explain(self, inputs: dict, feature: str, algorithm: str | None = None) -> dict:
        return self._call(
            "POST", "/api/v1/explain", json={"scenario": inputs, "feature": feature, "algorithm": algorithm}
        )

    def compare(self, inputs: dict) -> dict:
        return self._call("POST", "/api/v1/compare", json={"scenario": inputs})

    def ask(self, question: str, scenario_result: dict | None = None) -> dict:
        return self._call(
            "POST", "/api/v1/ai/ask", json={"question": question, "scenario_result": scenario_result}
        )

    def quick_answer(self, kind: str) -> dict:
        # Quick answers are deterministic and only read saved reports, so compute them locally.
        return LocalBackend._guard(lambda: analyst.quick_answer(kind))


@st.cache_resource
def _backend(api_url: str):
    return HttpBackend(api_url) if api_url else LocalBackend()


def get_backend():
    return _backend(get_settings().api_url)


@st.cache_data(ttl=300, show_spinner=False)
def cached_report(version: str, name: str) -> dict:
    return get_backend().report(name)


@st.cache_data(ttl=300, show_spinner=False)
def cached_model_info(version: str) -> dict:
    return get_backend().model_info()
