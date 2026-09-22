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
    """Single-turn chat request for the Phase 3 minimal loop.

    conversation_id is accepted but unused until Phase 4 wires up
    LangGraph + the SQLite checkpointer; it must never be trusted for
    authorization once that lands.
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
