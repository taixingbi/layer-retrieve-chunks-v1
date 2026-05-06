"""OpenAI-compatible /v1/chat/completions HTTP client (async)."""

from __future__ import annotations

import httpx

from app.logging_config import logger


async def chat_complete(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: float = 60.0,
) -> str:
    """Return assistant content from one chat completion call."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
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
