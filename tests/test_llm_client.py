from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.core.errors import (
    UpstreamAuthError,
    UpstreamRateLimitError,
    UpstreamResponseError,
    UpstreamServerError,
)
from app.core.llm_client import InternalLLMClient
from app.domain.schemas import ChatMessage


def make_settings(**overrides) -> Settings:
    defaults = dict(
        app_env="development",
        onprem_base_url="https://onprem-llm.test",
        onprem_chat_path="/v1/chat/completions",
        onprem_model="test-model",
        onprem_company="test-company",
        onprem_system_id="test-system",
        onprem_service_id="test-service",
        onprem_api_key="test-key",
    )
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def client_with_response(settings: Settings, handler) -> InternalLLMClient:
    transport = httpx.MockTransport(handler)
    return InternalLLMClient(settings, transport=transport)


STANDARD_OK_BODY = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "OK"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}


@pytest.mark.asyncio
async def test_headers_have_expected_names_and_values():
    settings = make_settings()
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(200, json=STANDARD_OK_BODY)

    client = client_with_response(settings, handler)
    try:
        await client.chat([ChatMessage(role="user", content="hi")])
    finally:
        await client.aclose()

    headers = captured["headers"]
    assert headers["x-company"] == "test-company"
    assert headers["x-system-id"] == "test-system"
    assert headers["x-service-id"] == "test-service"
    assert headers["x-api-key"] == "test-key"


@pytest.mark.asyncio
async def test_payload_has_required_fields():
    settings = make_settings()
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=STANDARD_OK_BODY)

    client = client_with_response(settings, handler)
    try:
        await client.chat(
            [ChatMessage(role="user", content="hi")],
            temperature=0.5,
            max_tokens=256,
        )
    finally:
        await client.aclose()

    body = captured["body"]
    assert body["model"] == "test-model"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["stream"] is False
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 256


@pytest.mark.asyncio
async def test_parses_standard_string_content():
    settings = make_settings()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=STANDARD_OK_BODY)

    client = client_with_response(settings, handler)
    try:
        result = await client.chat([ChatMessage(role="user", content="hi")])
    finally:
        await client.aclose()

    assert result.content == "OK"
    assert result.finish_reason == "stop"
    assert result.model == "test-model"
    assert result.request_id == "chatcmpl-123"
    assert result.usage.total_tokens == 6


@pytest.mark.asyncio
async def test_parses_list_of_content_parts():
    settings = make_settings()
    body = {
        "id": "chatcmpl-456",
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hello, "},
                        {"type": "text", "text": "world"},
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    client = client_with_response(settings, handler)
    try:
        result = await client.chat([ChatMessage(role="user", content="hi")])
    finally:
        await client.aclose()

    assert result.content == "Hello, world"


@pytest.mark.asyncio
async def test_empty_choices_raises_application_error():
    settings = make_settings()
    body = {"id": "chatcmpl-789", "model": "test-model", "choices": [], "usage": {}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    client = client_with_response(settings, handler)
    try:
        with pytest.raises(UpstreamResponseError):
            await client.chat([ChatMessage(role="user", content="hi")])
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code,expected_exc",
    [
        (401, UpstreamAuthError),
        (403, UpstreamAuthError),
        (429, UpstreamRateLimitError),
        (500, UpstreamServerError),
        (503, UpstreamServerError),
    ],
)
async def test_http_errors_are_mapped_safely(status_code, expected_exc):
    settings = make_settings()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": "upstream detail should not leak"})

    client = client_with_response(settings, handler)
    try:
        with pytest.raises(expected_exc) as exc_info:
            await client.chat([ChatMessage(role="user", content="hi")])
    finally:
        await client.aclose()

    # The application error message must not leak the raw upstream body.
    assert "upstream detail should not leak" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_trust_env_is_always_false():
    settings = make_settings()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=STANDARD_OK_BODY)

    client = client_with_response(settings, handler)
    try:
        assert client._client._transport is not None
        assert client._client.trust_env is False
    finally:
        await client.aclose()
