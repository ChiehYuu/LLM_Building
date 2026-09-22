"""
Internal LLM adapter for the on-prem, OpenAI-compatible chat API.

Responsibility: build headers/payload, perform the HTTPX call, and parse
the response into ChatResult. Nothing here talks to FastAPI or reads
request objects from the web layer, and nothing here reads .env directly
(it receives a Settings instance).

Hard rules (see Phase 2/3 notes):
  - trust_env is ALWAYS False. The corporate proxy redirects this endpoint
    to a block page; only a direct connection reaches the real API.
  - Never log header values or the API key.
  - Never return the raw upstream JSON to callers — always normalize to
    ChatResult first.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import httpx

from app.config import Settings
from app.core.errors import (
    LLMClientError,
    UpstreamConnectionError,
    UpstreamResponseError,
    UpstreamTimeoutError,
    map_http_error,
)
from app.domain.schemas import ChatMessage, ChatResult, Usage

logger = logging.getLogger(__name__)


class InternalLLMClient:
    def __init__(self, settings: Settings, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            # DIRECT CONNECTION REQUIRED: the environment's proxy redirects
            # this endpoint to /cgi-bin/blockpage.cgi. Do not remove.
            trust_env=False,
            verify=self._verify_param(),
            timeout=httpx.Timeout(
                connect=settings.onprem_connect_timeout_s,
                read=settings.onprem_read_timeout_s,
                write=10.0,
                pool=10.0,
            ),
            transport=transport,
        )

    def _verify_param(self):
        if self._settings.onprem_ca_bundle_path:
            return self._settings.onprem_ca_bundle_path
        # DEVELOPMENT ONLY: no company CA bundle configured yet (tracked
        # tech debt from Phase 2). Production must set
        # ONPREM_CA_BUNDLE_PATH and this fallback must not be relied on.
        if self._settings.app_env == "production":
            raise RuntimeError(
                "ONPREM_CA_BUNDLE_PATH is required when APP_ENV=production; "
                "refusing to start with verify=False."
            )
        logger.warning("onprem TLS verification disabled (verify=False) - development only")
        return False

    def _build_headers(self) -> dict[str, str]:
        return {
            "x-company": self._settings.onprem_company.get_secret_value(),
            "x-system-id": self._settings.onprem_system_id.get_secret_value(),
            "x-service-id": self._settings.onprem_service_id.get_secret_value(),
            "x-api-key": self._settings.onprem_api_key.get_secret_value(),
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        messages: Iterable[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        return {
            "model": self._settings.onprem_model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    async def chat(
        self,
        messages: Iterable[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ChatResult:
        headers = self._build_headers()
        payload = self._build_payload(messages, temperature, max_tokens)

        try:
            response = await self._client.post(
                self._settings.onprem_chat_url,
                headers=headers,
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError("Timed out calling on-prem LLM API") from exc
        except httpx.RequestError as exc:
            raise UpstreamConnectionError("Failed to reach on-prem LLM API") from exc

        if response.status_code != 200:
            raise map_http_error(response.status_code)

        try:
            data = response.json()
        except ValueError as exc:
            raise UpstreamResponseError("Upstream response was not valid JSON") from exc

        return self._parse_chat_completion(data)

    def _parse_chat_completion(self, data: dict) -> ChatResult:
        choices = data.get("choices") or []
        if not choices:
            raise UpstreamResponseError("Upstream response contained no choices")

        first_choice = choices[0]
        message = first_choice.get("message") or {}
        content = message.get("content")

        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            content = "".join(parts)

        if not isinstance(content, str):
            raise UpstreamResponseError("Upstream message.content had an unexpected type")

        usage_raw = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0) or 0,
            completion_tokens=usage_raw.get("completion_tokens", 0) or 0,
            total_tokens=usage_raw.get("total_tokens", 0) or 0,
        )

        return ChatResult(
            content=content,
            finish_reason=first_choice.get("finish_reason"),
            model=data.get("model") or self._settings.onprem_model,
            request_id=data.get("id"),
            usage=usage,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
