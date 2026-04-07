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
    timeout: float = 120.0,
) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            json={"model": model, "messages": messages, "max_tokens": max_tokens},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        try:
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected chat response shape: {data!r}") from e
        logger.info(
            "chat_complete ok url=%s model=%s reply_chars=%s",
            url,
            model,
            len(text),
        )
        return text
