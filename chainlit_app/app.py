"""
Chainlit GUI (Phase 5).

Responsibility, per the Phase 5 scope:
  - show the chat UI
  - create/hold a server-side conversation_id (from our own backend)
  - forward the user's message to the backend
  - show the final response, or a safe error message

Explicitly out of scope here (stays in app/):
  - reading .env / on-prem secrets
  - building the four on-prem headers
  - setting verify=False / trust_env=False
  - writing to SQLite
  - calling internal tool/project-management APIs directly
"""
from __future__ import annotations

import chainlit as cl

# Chainlit loads this file by path and only puts this file's own directory
# on sys.path (see chainlit.config.load_module), not the repo root, so this
# must be a bare import rather than `chainlit_app.backend_client`.
from backend_client import BackendClient, BackendClientError


@cl.on_chat_start
async def on_chat_start() -> None:
    client = BackendClient()
    cl.user_session.set("backend_client", client)

    try:
        conversation_id = await client.create_conversation()
    except BackendClientError as exc:
        await cl.ErrorMessage(content=f"Could not start a conversation: {exc.message}").send()
        return

    cl.user_session.set("conversation_id", conversation_id)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    client: BackendClient | None = cl.user_session.get("backend_client")
    conversation_id: str | None = cl.user_session.get("conversation_id")

    if client is None or conversation_id is None:
        await cl.ErrorMessage(content="Conversation not initialized. Please refresh the page.").send()
        return

    try:
        result = await client.send_message(conversation_id, message.content)
    except BackendClientError as exc:
        await cl.ErrorMessage(content=f"The assistant could not reply: {exc.message}").send()
        return

    await cl.Message(content=result["content"]).send()


@cl.on_chat_end
async def on_chat_end() -> None:
    client: BackendClient | None = cl.user_session.get("backend_client")
    if client is not None:
        await client.aclose()
