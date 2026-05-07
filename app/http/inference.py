"""OpenAI-compatible /v1/chat/completions HTTP client (async)."""

from __future__ import annotations

import httpx

from app.http._correlation import correlation_headers
from app.logging_config import logger


async def chat_complete(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    timeout: float = 60.0,
) -> str:
    """Return assistant content from one chat completion call. Correlation forwarded as
    ``X-Request-Id`` / ``X-Session-Id`` / ``X-Trace-Id`` (last only when set)."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        **correlation_headers(request_id, session_id, trace_id=trace_id),
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, TypeError, IndexError) as e:
        raise RuntimeError(f"Unexpected chat response shape: {data!r}") from e

    reply = content if isinstance(content, str) else str(content)
    logger.info(
        "chat_complete ok url=%s model=%s max_tokens=%s reply_chars=%s",
        url,
        model,
        max_tokens,
        len(reply),
    )
    return reply
