from __future__ import annotations

import httpx
import pytest

from chainlit_app.backend_client import BackendClient, BackendClientError


def client_with_response(handler) -> BackendClient:
    transport = httpx.MockTransport(handler)
    return BackendClient(base_url="http://backend.test", transport=transport)


@pytest.mark.asyncio
async def test_create_conversation_returns_id():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/conversations"
        return httpx.Response(201, json={"conversation_id": "abc-123"})

    client = client_with_response(handler)
    try:
        conversation_id = await client.create_conversation()
    finally:
        await client.aclose()

    assert conversation_id == "abc-123"


@pytest.mark.asyncio
async def test_send_message_posts_to_correct_path_and_returns_body():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "conversation_id": "abc-123",
                "content": "OK",
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "model": "test-model",
                "request_id": "req-1",
            },
        )

    client = client_with_response(handler)
    try:
        result = await client.send_message("abc-123", "hi")
    finally:
        await client.aclose()

    assert captured["path"] == "/api/conversations/abc-123/messages"
    assert b'"hi"' in captured["body"]
    assert result["content"] == "OK"


@pytest.mark.asyncio
async def test_backend_error_status_is_mapped_with_detail_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"detail": "Upstream server error (HTTP 500)"})

    client = client_with_response(handler)
    try:
        with pytest.raises(BackendClientError) as exc_info:
            await client.send_message("abc-123", "hi")
    finally:
        await client.aclose()

    assert exc_info.value.status_code == 502
    assert exc_info.value.message == "Upstream server error (HTTP 500)"


@pytest.mark.asyncio
async def test_backend_error_without_json_body_falls_back_to_generic_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"not json")

    client = client_with_response(handler)
    try:
        with pytest.raises(BackendClientError) as exc_info:
            await client.create_conversation()
    finally:
        await client.aclose()

    assert exc_info.value.status_code == 500
    assert exc_info.value.message == "backend error"


@pytest.mark.asyncio
async def test_connection_failure_is_wrapped_as_backend_client_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = client_with_response(handler)
    try:
        with pytest.raises(BackendClientError):
            await client.create_conversation()
    finally:
        await client.aclose()
