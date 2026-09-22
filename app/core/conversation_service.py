"""
Conversation service: the thin layer between FastAPI routes and the
compiled LangGraph graph.

Responsibility: translate (conversation_id, message text) into a graph
invocation keyed by thread_id, and translate the resulting LangChain
messages back into our own ChatResult / history models. No FastAPI
objects, no HTTP, no .env reading here.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.core.conversation_graph import ROLE_BY_MESSAGE_TYPE
from app.domain.schemas import ChatResult, ConversationHistoryItem, Usage


class ConversationService:
    def __init__(self, graph) -> None:
        self._graph = graph

    def _config(self, conversation_id: str, temperature: float | None = None, max_tokens: int | None = None) -> dict:
        configurable: dict = {"thread_id": conversation_id}
        if temperature is not None:
            configurable["temperature"] = temperature
        if max_tokens is not None:
            configurable["max_tokens"] = max_tokens
        return {"configurable": configurable}

    async def send_message(
        self,
        conversation_id: str,
        message: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ChatResult:
        config = self._config(conversation_id, temperature, max_tokens)
        state = await self._graph.ainvoke({"messages": [HumanMessage(content=message)]}, config=config)

        last_message = state["messages"][-1]
        metadata = last_message.response_metadata if isinstance(last_message, AIMessage) else {}
        usage_raw = metadata.get("usage") or {}

        return ChatResult(
            content=last_message.content,
            finish_reason=metadata.get("finish_reason"),
            model=metadata.get("model", "unknown"),
            request_id=metadata.get("request_id"),
            usage=Usage(**usage_raw) if usage_raw else Usage(),
        )

    async def get_history(self, conversation_id: str) -> list[ConversationHistoryItem]:
        config = self._config(conversation_id)
        snapshot = await self._graph.aget_state(config)
        raw_messages = (snapshot.values or {}).get("messages", [])
        return [
            ConversationHistoryItem(
                role=ROLE_BY_MESSAGE_TYPE.get(m.type, "user"),
                content=m.content,
            )
            for m in raw_messages
        ]
