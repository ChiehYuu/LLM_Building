"""
Phase 2 connectivity probe for the on-prem, OpenAI-compatible chat API.

Standalone diagnostic CLI, not part of the app's import graph: it exists
to answer "can we reach the on-prem API at all, and does the response
match the expected contract" without going through FastAPI/LangGraph.
core/llm_client.py is the production adapter that resulted from what
this probe verified; keep this script decoupled from it so probing
never depends on the adapter working first.

Usage:
    python scripts/probe_onprem_llm.py

Exit codes: 0 success, 2 wrong provider, 10/11 connect/read timeout,
12 other request error, 20 non-2xx HTTP response.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Allow running as `python scripts/probe_onprem_llm.py` from anywhere,
# without installing this project as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings, get_settings  # noqa: E402

PROBE_MESSAGE = "Reply with exactly: OK"


def build_headers(settings: Settings) -> dict[str, str]:
    return {
        "x-company": settings.onprem_company.get_secret_value(),
        "x-system-id": settings.onprem_system_id.get_secret_value(),
        "x-service-id": settings.onprem_service_id.get_secret_value(),
        "x-api-key": settings.onprem_api_key.get_secret_value(),
        "accept": "application/json",
        "content-type": "application/json",
    }


def build_payload(settings: Settings) -> dict[str, Any]:
    return {
        "model": settings.onprem_model,
        "messages": [
            {
                "role": "user",
                "content": PROBE_MESSAGE,
            }
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": 8,
    }


def safe_preview(text: str, limit: int = 1000) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}... [truncated]"


def summarize_json(value: Any) -> str:
    if isinstance(value, dict):
        keys = ", ".join(sorted(map(str, value.keys())))
        return f"JSON object keys: [{keys}]"

    if isinstance(value, list):
        return f"JSON array length: {len(value)}"

    return f"JSON scalar type: {type(value).__name__}"


def extract_text_content(content: Any) -> str | None:
    """Extract text from a string or OpenAI-style content parts."""
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return None

    text_parts: list[str] = []

    for part in content:
        if not isinstance(part, dict):
            continue

        if part.get("type") == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        elif isinstance(part.get("text"), str):
            text_parts.append(part["text"])

    return "".join(text_parts) if text_parts else None


def parse_chat_completion(payload: Any) -> dict[str, Any]:
    """Parse the first response choice without assuming optional fields exist."""
    if not isinstance(payload, dict):
        raise ValueError("Response JSON must be an object.")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Response JSON has no non-empty 'choices' list.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("The first item in 'choices' must be an object.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("choices[0].message must be an object.")

    tool_calls = message.get("tool_calls")

    return {
        "content": extract_text_content(message.get("content")),
        "reasoning_content": extract_text_content(message.get("reasoning_content")),
        "finish_reason": first_choice.get("finish_reason"),
        "has_tool_calls": isinstance(tool_calls, list) and bool(tool_calls),
        "choice_index": first_choice.get("index"),
        "usage": payload.get("usage"),
    }


async def probe() -> int:
    settings = get_settings()

    if settings.llm_provider != "onprem":
        print("Probe blocked: LLM_PROVIDER must be 'onprem'.")
        return 2

    url = settings.onprem_chat_url
    headers = build_headers(settings)
    payload = build_payload(settings)

    timeout = httpx.Timeout(
        connect=settings.onprem_connect_timeout_s,
        read=settings.onprem_read_timeout_s,
        write=30.0,
        pool=10.0,
    )

    print("=== On-prem LLM API Probe ===")
    print("Provider: onprem")
    print(f"URL: {url}")
    print("Method: POST")
    print("Proxy routing: disabled for this probe (direct connection)")
    print("TLS verification: disabled for this development-only probe")
    print(f"Request stream mode: {payload['stream']}")
    print(f"Request message length: {len(PROBE_MESSAGE)} characters")
    print()

    started = time.perf_counter()

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            verify=False,
            trust_env=False,
        ) as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
            )
    except httpx.ConnectTimeout:
        elapsed = time.perf_counter() - started
        print(f"Result: connect timeout after {elapsed:.2f}s")
        return 10
    except httpx.ReadTimeout:
        elapsed = time.perf_counter() - started
        print(f"Result: read timeout after {elapsed:.2f}s")
        return 11
    except httpx.RequestError as exc:
        elapsed = time.perf_counter() - started
        print(f"Result: request error after {elapsed:.2f}s")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {safe_preview(str(exc), 200)}")
        return 12

    elapsed = time.perf_counter() - started
    content_type = response.headers.get("content-type", "not provided")
    request_id = (
        response.headers.get("x-request-id")
        or response.headers.get("request-id")
        or response.headers.get("trace-id")
        or "not provided"
    )

    print(f"HTTP status: {response.status_code}")
    print(f"Latency: {elapsed:.2f}s")
    print(f"Content-Type: {content_type}")
    print(f"Response request/trace ID: {request_id}")
    print(f"Response body length: {len(response.content)} bytes")

    try:
        response_json = response.json()
        print(summarize_json(response_json))

        try:
            parsed = parse_chat_completion(response_json)
        except ValueError as exc:
            print(f"Chat completion parser: {exc}")
        else:
            print(f"Choice index: {parsed['choice_index']}")
            print(f"Finish reason: {parsed['finish_reason']}")
            print(f"Tool calls present: {parsed['has_tool_calls']}")

            if parsed["content"] is not None:
                print(f"Assistant response: {safe_preview(parsed['content'])}")
            else:
                print("Assistant response: no final text content.")

            if parsed["reasoning_content"] is not None:
                print(
                    "Reasoning content present: yes "
                    f"(length={len(parsed['reasoning_content'])})"
                )
            else:
                print("Reasoning content present: no")

            usage = parsed["usage"]
            if isinstance(usage, dict):
                print(
                    "Usage: "
                    f"prompt={usage.get('prompt_tokens')}, "
                    f"completion={usage.get('completion_tokens')}, "
                    f"total={usage.get('total_tokens')}"
                )

    except json.JSONDecodeError:
        print("Response is not JSON.")
        print(f"Safe body preview: {safe_preview(response.text, limit=300)}")

    if 200 <= response.status_code < 300:
        print("Result: success")
        return 0

    print("Result: non-success HTTP response")
    print(f"Safe error body preview: {safe_preview(response.text, limit=300)}")
    return 20


if __name__ == "__main__":
    raise SystemExit(asyncio.run(probe()))
