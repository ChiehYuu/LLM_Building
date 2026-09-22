from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.errors import UpstreamServerError
from app.domain.schemas import ChatResult, Usage
from app.main import app


class FakeLLMClient:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    async def chat(self, messages, temperature=0.2, max_tokens=1024):
        if self._error:
            raise self._error
        return self._result

    async def aclose(self):
        pass


def test_healthz_does_not_touch_llm_client():
    with TestClient(app) as client:
        # Swap in a client whose .chat() would raise; /healthz must never
        # reach it.
        client.app.state.llm_client = FakeLLMClient(error=RuntimeError("should not be called"))
        response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "app" in body and "environment" in body


def test_readyz_reports_incomplete_onprem_config():
    with TestClient(app) as client:
        client.app.state.llm_client = FakeLLMClient(error=RuntimeError("should not be called"))
        response = client.get("/readyz")

    # Default Settings() has empty SecretStr onprem credentials, so this
    # should be not_ready, and must never include secret values.
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["onprem_config_complete"] is False
    assert "onprem_api_key" not in str(body)


def test_chat_endpoint_returns_normalized_response():
    fake_result = ChatResult(
        content="OK",
        finish_reason="stop",
        model="test-model",
        request_id="req-1",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    with TestClient(app) as client:
        client.app.state.llm_client = FakeLLMClient(result=fake_result)
        response = client.post("/api/chat", json={"message": "Reply with exactly: OK"})

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "OK"
    assert body["finish_reason"] == "stop"
    assert body["model"] == "test-model"
    assert body["usage"]["total_tokens"] == 2
    # Never leak raw upstream-only fields.
    assert "prompt_text" not in body
    assert "kv_transfer_params" not in body


def test_chat_endpoint_maps_llm_client_error_to_safe_http_error():
    with TestClient(app) as client:
        client.app.state.llm_client = FakeLLMClient(
            error=UpstreamServerError("Upstream server error (HTTP 500)")
        )
        response = client.post("/api/chat", json={"message": "hi"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Upstream server error (HTTP 500)"


def test_chat_endpoint_rejects_empty_message():
    with TestClient(app) as client:
        client.app.state.llm_client = FakeLLMClient(error=RuntimeError("should not be called"))
        response = client.post("/api/chat", json={"message": ""})

    assert response.status_code == 422
