"""Provider-independent LLM adapter. Keys stay server-side; nothing here runs in the browser."""

from __future__ import annotations

import os
from typing import Protocol

import httpx


class ProviderError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> dict:
        """Return {"content": [blocks], "stop_reason": str, "usage": {...}} (Anthropic block format)."""


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, timeout: float = 45.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = os.environ.get("LLM_API_BASE", "https://api.anthropic.com").rstrip("/")

    def complete(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> dict:
        try:
            resp = httpx.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": messages,
                    "tools": tools,
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"LLM request failed: {type(exc).__name__}") from exc
        if resp.status_code != 200:
            raise ProviderError(f"LLM provider returned HTTP {resp.status_code}")
        body = resp.json()
        return {
            "content": body.get("content", []),
            "stop_reason": body.get("stop_reason"),
            "usage": body.get("usage", {}),
        }


class ScriptedProvider:
    """Deterministic provider for tests: replays a list of prepared responses."""

    name = "scripted"

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> dict:
        self.calls.append({"system": system, "messages": messages, "tools": tools})
        if not self.responses:
            raise ProviderError("No scripted responses left")
        return self.responses.pop(0)
