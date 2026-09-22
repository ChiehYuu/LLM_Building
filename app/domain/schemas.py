"""
API and internal data models.

Responsibility: define request/response shapes for the FastAPI layer and
the normalized result the LLM client hands back. This module must never
call the API or read .env — that belongs to core/llm_client.py and
config.py respectively.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Single-turn, stateless chat request (POST /api/chat).

    conversation_id is accepted but ignored here — it exists only so the
    same request shape can be echoed back. Multi-turn state lives behind
    POST /api/conversations/{conversation_id}/messages (Phase 4).
    """

    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResult(BaseModel):
    """Normalized result produced by InternalLLMClient.

    Deliberately excludes anything from the raw upstream payload beyond
    what a caller of our API is meant to see (no prompt_token_ids,
    kv_transfer_params, ec_transfer_params, or full metrics).
    """

    content: str
    finish_reason: Optional[str] = None
    model: str
    request_id: Optional[str] = None
    usage: Usage = Field(default_factory=Usage)


class ChatResponse(BaseModel):
    conversation_id: Optional[str] = None
    content: str
    finish_reason: Optional[str] = None
    usage: Usage
    model: str
    request_id: Optional[str] = None


class ConversationCreateResponse(BaseModel):
    """Backend-generated conversation identity. The frontend must not
    invent its own conversation_id — see Phase 4 notes on why an
    unguessable, backend-issued id matters once authorization lands."""

    conversation_id: str


class ConversationMessageRequest(BaseModel):
    """A single turn posted into an existing conversation. conversation_id
    comes from the URL path, not the body, so there is exactly one source
    of truth for which thread this turn belongs to."""

    message: str = Field(..., min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)


class ConversationHistoryItem(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    messages: list[ConversationHistoryItem]
