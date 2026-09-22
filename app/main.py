"""
FastAPI application entrypoint.

Responsibility: build the app, wire up lifespan-managed dependencies
(InternalLLMClient), register routes, and expose /healthz and /readyz.
No LLM business logic belongs here.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.routes_chat import router as chat_router
from app.config import get_settings
from app.core.llm_client import InternalLLMClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.llm_client = InternalLLMClient(settings)
    try:
        yield
    finally:
        await app.state.llm_client.aclose()


app = FastAPI(title="LLM Building", lifespan=lifespan)
app.include_router(chat_router)


@app.get("/healthz")
async def healthz():
    """Liveness only. Must never call the LLM API."""
    settings = get_settings()
    return {"app": settings.app_name, "environment": settings.app_env, "status": "ok"}


@app.get("/readyz")
async def readyz():
    """Readiness: config completeness + local storage writability.
    Must never call the LLM API and must never return secret values."""
    settings = get_settings()

    config_ok = settings.onprem_config_complete

    sqlite_dir = settings.sqlite_dir
    try:
        sqlite_dir.mkdir(parents=True, exist_ok=True)
        storage_ok = sqlite_dir.is_dir() and os.access(sqlite_dir, os.W_OK)
    except OSError:
        storage_ok = False

    ready = config_ok and storage_ok
    body = {
        "status": "ready" if ready else "not_ready",
        "checks": {
            "onprem_config_complete": config_ok,
            "sqlite_dir_writable": storage_ok,
        },
    }
    if not ready:
        return JSONResponse(status_code=503, content=body)
    return body
