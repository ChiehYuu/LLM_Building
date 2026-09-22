"""
LangGraph state graph for multi-turn conversations.

Responsibility: define the conversation State, the single node that calls
InternalLLMClient, and build/compile the graph against a checkpointer.
No FastAPI, no HTTP, no .env reading here — the compiled graph is driven
by core/conversation_service.py.

Design (Phase 4, minimal loop):
  START -> call_llm -> END
The checkpointer persists the running message list per thread_id (i.e.
conversation_id). Only one node exists today; routing/tools are later
phases (see Phase 6).
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.core.llm_client import InternalLLMClient
from app.domain.schemas import ChatMessage


class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


ROLE_BY_MESSAGE_TYPE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
}


def to_chat_message(message: BaseMessage) -> ChatMessage:
    role = ROLE_BY_MESSAGE_TYPE.get(message.type, "user")
    return ChatMessage(role=role, content=message.content)


def build_conversation_graph(llm_client: InternalLLMClient, checkpointer: BaseCheckpointSaver):
    async def call_llm(state: ConversationState, config: RunnableConfig) -> dict:
        configurable = config.get("configurable", {})
        temperature = configurable.get("temperature", 0.2)
        max_tokens = configurable.get("max_tokens", 1024)

        chat_messages = [to_chat_message(m) for m in state["messages"]]
        result = await llm_client.chat(chat_messages, temperature=temperature, max_tokens=max_tokens)

        ai_message = AIMessage(
            content=result.content,
            response_metadata={
                "finish_reason": result.finish_reason,
                "model": result.model,
                "request_id": result.request_id,
                "usage": result.usage.model_dump(),
            },
        )
        return {"messages": [ai_message]}

    builder = StateGraph(ConversationState)
    builder.add_node("call_llm", call_llm)
    builder.add_edge(START, "call_llm")
    builder.add_edge("call_llm", END)

    return builder.compile(checkpointer=checkpointer)
