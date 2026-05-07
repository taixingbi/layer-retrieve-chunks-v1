"""Shared correlation HTTP headers for upstream API calls (embed / rerank / chat)."""
from __future__ import annotations


def correlation_headers(
    request_id: str,
    session_id: str,
    *,
    trace_id: str | None = None,
) -> dict[str, str]:
    """
    Return ``{X-Request-Id, X-Session-Id, [X-Trace-Id]}`` for forwarding to upstream services.

    ``trace_id`` is included only when truthy (after strip); ``request_id`` and ``session_id``
    must both be non-empty.
    """
    if not request_id or not request_id.strip():
        raise ValueError("request_id is required and must be non-empty.")
    if not session_id or not session_id.strip():
        raise ValueError("session_id is required and must be non-empty.")
    h: dict[str, str] = {
        "X-Request-Id": request_id,
        "X-Session-Id": session_id,
    }
    t = (trace_id or "").strip()
    if t:
        h["X-Trace-Id"] = t
    return h
