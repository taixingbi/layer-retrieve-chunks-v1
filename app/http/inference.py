"""OpenAI-compatible /v1/chat/completions HTTP client (async)."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

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


async def chat_complete_stream(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    timeout: float = 60.0,
) -> AsyncIterator[str]:
    """Yield assistant ``content`` deltas as they arrive from a streaming chat-completions
    call (``stream: true``). Forwards correlation as ``X-Request-Id`` / ``X-Session-Id`` /
    ``X-Trace-Id`` (last only when set). On exit, emits one structured log line with TTFT
    (time-to-first-token, ms) and total generation time (ms) for SLO dashboards."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        **correlation_headers(request_id, session_id, trace_id=trace_id),
    }
    t0 = time.perf_counter()
    ttft_ms: int | None = None
    char_count = 0
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    body = line[6:]
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or [{}]
                    delta = (choices[0] or {}).get("delta", {}).get("content") or ""
                    if not delta:
                        continue
                    if ttft_ms is None:
                        ttft_ms = int(round((time.perf_counter() - t0) * 1000))
                    char_count += len(delta)
                    yield delta
    except asyncio.CancelledError:
        # Client disconnect / Pause: closing the `async with` blocks above tears down
        # the TCP connection to the upstream so vLLM cancels its in-flight generation
        # (frees the GPU slot). Log structured before re-raising so ops can distinguish
        # cancel from completion.
        cancel_ms = int(round((time.perf_counter() - t0) * 1000))
        cancel_ttft = ttft_ms if ttft_ms is not None else 0
        logger.warning(
            "chat_complete_stream cancelled url=%s model=%s max_tokens=%s "
            "reply_chars=%s ttft_ms=%s gen_ms=%s",
            url,
            model,
            max_tokens,
            char_count,
            cancel_ttft,
            cancel_ms,
            extra={
                "ttft_ms": cancel_ttft,
                "gen_ms": cancel_ms,
                "reason": "client_cancelled",
            },
        )
        raise
    gen_ms = int(round((time.perf_counter() - t0) * 1000))
    final_ttft = ttft_ms if ttft_ms is not None else 0
    logger.info(
        "chat_complete_stream ok url=%s model=%s max_tokens=%s reply_chars=%s ttft_ms=%s gen_ms=%s",
        url,
        model,
        max_tokens,
        char_count,
        final_ttft,
        gen_ms,
        extra={"ttft_ms": final_ttft, "gen_ms": gen_ms},
    )
