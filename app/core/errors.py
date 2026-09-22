"""
Application-level errors for the LLM adapter.

Responsibility: translate upstream (HTTP transport / on-prem API) failures
into a small set of application errors that routes can map to safe HTTP
responses. Never include header values, API keys, or raw upstream bodies
in these messages — they may end up in logs.
"""
from __future__ import annotations


class LLMClientError(Exception):
    """Base class for all InternalLLMClient failures."""

    http_status: int = 502

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UpstreamAuthError(LLMClientError):
    """Upstream rejected our credentials (401/403). This is a server-side
    misconfiguration from the caller's point of view, not theirs."""

    http_status = 502


class UpstreamRateLimitError(LLMClientError):
    """Upstream returned 429."""

    http_status = 429


class UpstreamServerError(LLMClientError):
    """Upstream returned 5xx."""

    http_status = 502


class UpstreamTimeoutError(LLMClientError):
    """Connect or read timeout talking to the upstream API."""

    http_status = 504


class UpstreamConnectionError(LLMClientError):
    """Transport-level failure (DNS, TCP, TLS, proxy block page, ...)."""

    http_status = 502


class UpstreamResponseError(LLMClientError):
    """Upstream returned 200 but the body didn't match the expected
    OpenAI-compatible chat.completion contract."""

    http_status = 502


def map_http_error(status_code: int) -> LLMClientError:
    if status_code in (401, 403):
        return UpstreamAuthError(f"Upstream authentication failed (HTTP {status_code})")
    if status_code == 429:
        return UpstreamRateLimitError("Upstream rate limit exceeded")
    if status_code >= 500:
        return UpstreamServerError(f"Upstream server error (HTTP {status_code})")
    return UpstreamResponseError(f"Unexpected upstream status (HTTP {status_code})")
