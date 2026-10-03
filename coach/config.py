"""Settings, read from Streamlit secrets first and environment variables second.

Every value has a safe default except the OpenAI key and the email allowlist,
which must be set for the deployed app (see README).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _secret(key: str, default=None):
    try:
        import streamlit as st

        if key in st.secrets:
            return st.secrets[key]
    except Exception:  # no secrets file, or not running under Streamlit
        pass
    return os.environ.get(key, default)


def _as_bool(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).strip().lower() for x in v if str(x).strip()]
    return [x.strip().lower() for x in str(v).split(",") if x.strip()]


@dataclass
class Settings:
    openai_api_key: str | None = None
    openai_model: str = "gpt-5-nano"
    reasoning_effort: str | None = None
    price_input_per_m: float = 0.05
    price_output_per_m: float = 0.40
    monthly_budget_usd: float = 2.0
    allowed_emails: list[str] = field(default_factory=list)
    dev_mode_no_auth: bool = False
    github_token: str | None = None
    github_repo: str | None = None  # "owner/name"
    github_branch: str = "main"
    github_state_path: str = "data/state.json"
    local_state_path: str = "data/state.json"
    timezone: str = "Asia/Kolkata"

    @property
    def use_github(self) -> bool:
        return bool(self.github_token and self.github_repo)


def load_settings() -> Settings:
    return Settings(
        openai_api_key=_secret("OPENAI_API_KEY"),
        openai_model=_secret("OPENAI_MODEL", "gpt-5-nano"),
        reasoning_effort=_secret("OPENAI_REASONING_EFFORT") or None,
        price_input_per_m=float(_secret("PRICE_INPUT_PER_M", 0.05)),
        price_output_per_m=float(_secret("PRICE_OUTPUT_PER_M", 0.40)),
        monthly_budget_usd=float(_secret("MONTHLY_BUDGET_USD", 2.0)),
        allowed_emails=_as_list(_secret("ALLOWED_EMAILS")),
        dev_mode_no_auth=_as_bool(_secret("DEV_MODE_NO_AUTH", False)),
        github_token=_secret("GITHUB_TOKEN"),
        github_repo=_secret("GITHUB_REPO"),
        github_branch=_secret("GITHUB_BRANCH", "main"),
        github_state_path=_secret("GITHUB_STATE_PATH", "data/state.json"),
        local_state_path=_secret("LOCAL_STATE_PATH", "data/state.json"),
        timezone=_secret("TIMEZONE", "Asia/Kolkata"),
    )
