"""
Chat HTTP route.

Responsibility: accept a validated ChatRequest, call InternalLLMClient,
translate LLMClientError into a safe HTTPException, and shape the
response. No header building, no payload assembly, no response parsing
belongs here — that's core/llm_client.py's job.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.core.errors import LLMClientError
from app.core.llm_client import InternalLLMClient
from app.domain.schemas import ChatMessage, ChatRequest, ChatResponse

router = APIRouter()


def get_llm_client(request: Request) -> InternalLLMClient:
    return request.app.state.llm_client


@router.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    client = get_llm_client(request)
    messages = [ChatMessage(role="user", content=payload.message)]

    try:
        result = await client.chat(
            messages,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
    except LLMClientError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.message) from exc

    return ChatResponse(
        conversation_id=payload.conversation_id,
        content=result.content,
        finish_reason=result.finish_reason,
        usage=result.usage,
        model=result.model,
        request_id=result.request_id,
    )
