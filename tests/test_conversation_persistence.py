"""
Proves the Phase 4 acceptance criterion: SQLite checkpoint state survives
a restart. We open a fresh AsyncSqliteSaver against the same file to
simulate the process being restarted between requests.
"""
from __future__ import annotations

import uuid

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.core.conversation_graph import build_conversation_graph
from app.core.conversation_service import ConversationService
from app.domain.schemas import ChatResult, Usage


class FakeLLMClient:
    """Duck-typed stand-in for InternalLLMClient: echoes how many prior
    messages it saw, so we can assert on ordering/accumulation."""

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, temperature=0.2, max_tokens=1024):
        self.calls += 1
        seen = len(list(messages))
        return ChatResult(
            content=f"reply-{self.calls} (saw {seen} messages)",
            finish_reason="stop",
            model="fake-model",
            request_id=f"req-{self.calls}",
            usage=Usage(prompt_tokens=seen, completion_tokens=1, total_tokens=seen + 1),
        )


@pytest.mark.asyncio
async def test_history_survives_reopening_the_sqlite_file(tmp_path):
    db_path = str(tmp_path / "checkpoints.sqlite")
    conversation_id = str(uuid.uuid4())
    llm_client = FakeLLMClient()

    # "Process 1": send the first turn, then close the connection.
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        await checkpointer.setup()
        graph = build_conversation_graph(llm_client, checkpointer)
        service = ConversationService(graph)
        result = await service.send_message(conversation_id, "hello")
        assert result.content == "reply-1 (saw 1 messages)"

    # "Process 2": reopen the same file (fresh connection + fresh graph)
    # and confirm the earlier turn is still there.
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        await checkpointer.setup()
        graph = build_conversation_graph(llm_client, checkpointer)
        service = ConversationService(graph)

        history = await service.get_history(conversation_id)
        assert [m.role for m in history] == ["user", "assistant"]
        assert history[0].content == "hello"
        assert history[1].content == "reply-1 (saw 1 messages)"

        # Second turn should see the full prior history (2 messages) plus
        # the new human message.
        result = await service.send_message(conversation_id, "again")
        assert result.content == "reply-2 (saw 3 messages)"

        history = await service.get_history(conversation_id)
        assert [m.role for m in history] == ["user", "assistant", "user", "assistant"]
        assert history[2].content == "again"


@pytest.mark.asyncio
async def test_different_conversation_ids_are_isolated(tmp_path):
    db_path = str(tmp_path / "checkpoints.sqlite")
    llm_client = FakeLLMClient()

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        await checkpointer.setup()
        graph = build_conversation_graph(llm_client, checkpointer)
        service = ConversationService(graph)

        id_a, id_b = str(uuid.uuid4()), str(uuid.uuid4())
        await service.send_message(id_a, "from A")
        await service.send_message(id_b, "from B")

        history_a = await service.get_history(id_a)
        history_b = await service.get_history(id_b)

        assert [m.content for m in history_a if m.role == "user"] == ["from A"]
        assert [m.content for m in history_b if m.role == "user"] == ["from B"]


@pytest.mark.asyncio
async def test_unknown_conversation_id_has_empty_history(tmp_path):
    db_path = str(tmp_path / "checkpoints.sqlite")
    llm_client = FakeLLMClient()

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        await checkpointer.setup()
        graph = build_conversation_graph(llm_client, checkpointer)
        service = ConversationService(graph)

        history = await service.get_history(str(uuid.uuid4()))
        assert history == []
