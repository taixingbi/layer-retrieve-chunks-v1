"""OpenAI-compatible /v1/rerank HTTP client (async)."""

from __future__ import annotations

import httpx

from app.logging_config import logger


async def rerank_texts(
    *,
    base_url: str,
    model: str,
    query: str,
    documents: list[str],
    top_n: int,
    timeout: float = 30.0,
) -> list[dict]:
    """
    Return ranked rows with ``index`` and ``score``.

    Expected API shape:
    {
      "results": [{"index": 0, "relevance_score": 0.92}, ...]
    }
    """
    if not documents or top_n < 1:
        return []
    url = f"{base_url.rstrip('/')}/v1/rerank"
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
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
        rows = data["results"]
        out = [
            {
                "index": int(x["index"]),
                "score": float(x.get("relevance_score", x.get("score", 0.0))),
            }
            for x in rows
        ]
    except (KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"Unexpected rerank response shape: {data!r}") from e
    logger.info(
        "rerank_texts ok url=%s model=%s docs=%s returned=%s",
        url,
        model,
        len(documents),
        len(out),
    )
    return out
