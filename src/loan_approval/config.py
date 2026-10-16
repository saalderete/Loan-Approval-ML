"""Runtime settings, read from environment variables on every call so tests can override them."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("LOAN_ML_ROOT") or Path(__file__).resolve().parents[2])
CONFIGS_DIR = ROOT / "configs"

DISCLAIMER = (
    "Educational research output only. This is not a lending decision, a credit score, "
    "or advice about any real application."
)

OUTCOME_DEFINITION = (
    "Estimated probability that a Texas mortgage application with these characteristics "
    "was denied by the lender, as recorded in 2025 HMDA data (action_taken = 3), "
    "out of applications that were either approved or denied."
)


def _path(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value).expanduser().resolve() if value else default


def _int(env: str, default: int) -> int:
    try:
        return int(os.environ.get(env, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    models_dir: Path
    data_dir: Path
    state_dir: Path
    admin_token: str
    api_url: str
    anthropic_api_key: str
    llm_model: str
    ai_daily_request_limit: int
    rate_limit_per_minute: int

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def db_path(self) -> Path:
        return self.state_dir / "loan_approval.db"

    @property
    def ai_enabled(self) -> bool:
        return bool(self.anthropic_api_key)


def get_settings() -> Settings:
    return Settings(
        models_dir=_path("LOAN_ML_MODELS_DIR", ROOT / "models"),
        data_dir=_path("LOAN_ML_DATA_DIR", ROOT / "data"),
        state_dir=_path("LOAN_ML_STATE_DIR", ROOT / "var"),
        admin_token=os.environ.get("ADMIN_TOKEN", ""),
        api_url=os.environ.get("API_URL", "").rstrip("/"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        llm_model=os.environ.get("LLM_MODEL", "claude-opus-5-5"),
        ai_daily_request_limit=_int("AI_DAILY_REQUEST_LIMIT", 200),
        rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 60),
    )
