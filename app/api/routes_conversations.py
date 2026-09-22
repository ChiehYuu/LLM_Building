"""
Conversation HTTP routes (Phase 4).

Responsibility: accept validated requests, drive ConversationService, and
translate LLMClientError into a safe HTTPException. No graph wiring, no
LangChain message handling, no checkpoint access belongs here.
"""
from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request

from app.core.conversation_service import ConversationService
from app.core.errors import LLMClientError
from app.domain.schemas import (
    ChatResponse,
    ConversationCreateResponse,
    ConversationHistoryResponse,
    ConversationMessageRequest,
)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def get_conversation_service(request: Request) -> ConversationService:
    return request.app.state.conversation_service


@router.post("", response_model=ConversationCreateResponse, status_code=201)
async def create_conversation() -> ConversationCreateResponse:
    """Issue a backend-generated conversation id. The frontend must use
    this id rather than inventing its own — see Phase 4/6 notes on why
    conversation_id cannot be client-trusted once authorization lands."""
    return ConversationCreateResponse(conversation_id=str(uuid4()))


@router.post("/{conversation_id}/messages", response_model=ChatResponse)
async def post_message(
    conversation_id: UUID, payload: ConversationMessageRequest, request: Request
) -> ChatResponse:
    service = get_conversation_service(request)
    try:
        result = await service.send_message(
            str(conversation_id),
            payload.message,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
    except LLMClientError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.message) from exc

    return ChatResponse(
        conversation_id=str(conversation_id),
        content=result.content,
        finish_reason=result.finish_reason,
        usage=result.usage,
        model=result.model,
        request_id=result.request_id,
    )


@router.get("/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation(conversation_id: UUID, request: Request) -> ConversationHistoryResponse:
    service = get_conversation_service(request)
    messages = await service.get_history(str(conversation_id))
    if not messages:
        raise HTTPException(status_code=404, detail="conversation not found")
    return ConversationHistoryResponse(conversation_id=str(conversation_id), messages=messages)
