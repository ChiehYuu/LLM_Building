from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.core.errors import UpstreamServerError
from app.domain.schemas import ChatResult, ConversationHistoryItem, Usage
from app.main import app


class FakeConversationService:
    def __init__(self, result=None, error=None, history=None):
        self._result = result
        self._error = error
        self._history = history or []
        self.last_call = None

    async def send_message(self, conversation_id, message, temperature=0.2, max_tokens=1024):
        self.last_call = (conversation_id, message, temperature, max_tokens)
        if self._error:
            raise self._error
        return self._result

    async def get_history(self, conversation_id):
        return self._history


def test_create_conversation_returns_a_valid_uuid():
    with TestClient(app) as client:
        response = client.post("/api/conversations")

    assert response.status_code == 201
    conversation_id = response.json()["conversation_id"]
    # Must round-trip through uuid.UUID without raising.
    uuid.UUID(conversation_id)


def test_post_message_returns_normalized_response_with_conversation_id():
    conv_id = uuid.uuid4()
    fake_result = ChatResult(
        content="OK",
        finish_reason="stop",
        model="test-model",
        request_id="req-1",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    with TestClient(app) as client:
        client.app.state.conversation_service = FakeConversationService(result=fake_result)
        response = client.post(f"/api/conversations/{conv_id}/messages", json={"message": "hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == str(conv_id)
    assert body["content"] == "OK"
    assert body["usage"]["total_tokens"] == 2


def test_post_message_rejects_non_uuid_conversation_id():
    with TestClient(app) as client:
        client.app.state.conversation_service = FakeConversationService(
            error=RuntimeError("should not be called")
        )
        response = client.post("/api/conversations/not-a-uuid/messages", json={"message": "hi"})

    assert response.status_code == 422


def test_post_message_maps_llm_client_error_to_safe_http_error():
    conv_id = uuid.uuid4()
    with TestClient(app) as client:
        client.app.state.conversation_service = FakeConversationService(
            error=UpstreamServerError("Upstream server error (HTTP 500)")
        )
        response = client.post(f"/api/conversations/{conv_id}/messages", json={"message": "hi"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Upstream server error (HTTP 500)"


def test_get_history_returns_404_when_conversation_unknown():
    conv_id = uuid.uuid4()
    with TestClient(app) as client:
        client.app.state.conversation_service = FakeConversationService(history=[])
        response = client.get(f"/api/conversations/{conv_id}")

    assert response.status_code == 404


def test_get_history_returns_messages_in_order():
    conv_id = uuid.uuid4()
    history = [
        ConversationHistoryItem(role="user", content="hi"),
        ConversationHistoryItem(role="assistant", content="hello"),
    ]
    with TestClient(app) as client:
        client.app.state.conversation_service = FakeConversationService(history=history)
        response = client.get(f"/api/conversations/{conv_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == str(conv_id)
    assert body["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
