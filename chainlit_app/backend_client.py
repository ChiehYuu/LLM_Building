"""
Thin HTTP client from the Chainlit GUI to OUR OWN FastAPI backend.

Responsibility: call POST /api/conversations, POST
/api/conversations/{id}/messages, and translate a non-2xx response into
BackendClientError. Nothing more.

Hard rule (Phase 5): Chainlit talks to this backend only. It must never
read .env, build the four on-prem headers, set verify=False /
trust_env=False, touch SQLite directly, or call any internal tool/PM API
directly — all of that stays behind app/core/llm_client.py and the
FastAPI routes. This module's only secret-adjacent config is the
backend's own base URL, which is not a credential.
"""
from __future__ import annotations

import os
from typing import Optional

import httpx

BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "http://127.0.0.1:8000")


class BackendClientError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class BackendClient:
    def __init__(self, base_url: str = BACKEND_BASE_URL, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
            transport=transport,
        )

    async def create_conversation(self) -> str:
        try:
            response = await self._client.post("/api/conversations")
        except httpx.RequestError as exc:
            raise BackendClientError("Could not reach the backend service") from exc

        self._raise_for_status(response)
        return response.json()["conversation_id"]

    async def send_message(self, conversation_id: str, message: str) -> dict:
        try:
            response = await self._client.post(
                f"/api/conversations/{conversation_id}/messages",
                json={"message": message},
            )
        except httpx.RequestError as exc:
            raise BackendClientError("Could not reach the backend service") from exc

        self._raise_for_status(response)
        return response.json()

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            detail = "backend error"
            try:
                body = response.json()
                if isinstance(body, dict) and "detail" in body:
                    detail = str(body["detail"])
            except ValueError:
                pass
            raise BackendClientError(detail, status_code=response.status_code)

    async def aclose(self) -> None:
        await self._client.aclose()
