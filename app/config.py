"""
Application configuration.

Responsibility: load .env, expose typed Settings, keep secrets as SecretStr.
This module must never perform HTTP calls or parse LLM payloads — that
belongs to app/core/llm_client.py.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App identity ---
    app_name: str = "llm-building"
    app_env: Literal["development", "production"] = "development"

    # --- Provider policy ---
    # Only "onprem" is wired up today. Cloud config is reserved for later
    # and must never be used automatically as a fallback.
    llm_provider: Literal["onprem", "cloud"] = "onprem"
    allow_cloud_llm: bool = False

    # --- On-prem LLM connection ---
    onprem_base_url: str = "https://onprem-llm.internal"
    onprem_chat_path: str = "/v1/chat/completions"
    onprem_model: str = "onprem-default"

    # Identification + auth headers. Treated as secrets even though some
    # (company/system-id/service-id) are identifiers rather than credentials,
    # per the security policy set in Phase 1.
    onprem_company: SecretStr = SecretStr("")
    onprem_system_id: SecretStr = SecretStr("")
    onprem_service_id: SecretStr = SecretStr("")
    onprem_api_key: SecretStr = SecretStr("")

    # --- Network / TLS ---
    # Direct connection is required: the corporate proxy redirects to a
    # block page for this endpoint. The HTTPX client MUST be constructed
    # with trust_env=False (enforced in llm_client.py, not configurable here).
    onprem_ca_bundle_path: Optional[str] = None  # set once company CA is obtained
    onprem_connect_timeout_s: float = 10.0
    onprem_read_timeout_s: float = 110.0

    # --- Local state (used by /readyz and, later, LangGraph checkpointer) ---
    sqlite_path: str = "./data/checkpoints.sqlite"

    @property
    def onprem_chat_url(self) -> str:
        return self.onprem_base_url.rstrip("/") + self.onprem_chat_path

    @property
    def onprem_config_complete(self) -> bool:
        return all(
            [
                self.onprem_base_url,
                self.onprem_model,
                self.onprem_company.get_secret_value(),
                self.onprem_system_id.get_secret_value(),
                self.onprem_service_id.get_secret_value(),
                self.onprem_api_key.get_secret_value(),
            ]
        )

    @property
    def sqlite_dir(self) -> Path:
        return Path(self.sqlite_path).resolve().parent


@lru_cache
def get_settings() -> Settings:
    return Settings()
