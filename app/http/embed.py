"""OpenAI-compatible /v1/embeddings HTTP client (async)."""

from __future__ import annotations

import httpx

from app.config import get_embedding_model, get_embedding_url, VECTOR_SIZE
from app.logging_config import logger
from app.request_context import bind_request_context


def _embed_headers(
    request_id: str,
    session_id: str,
    *,
    trace_id: str | None = None,
) -> dict[str, str]:
    if not request_id or not request_id.strip():
        raise ValueError("request_id is required and must be non-empty.")
    if not session_id or not session_id.strip():
        raise ValueError("session_id is required and must be non-empty.")
    h: dict[str, str] = {
        "Content-Type": "application/json",
        "X-Request-Id": request_id,
        "X-Session-Id": session_id,
    }
    t = (trace_id or "").strip()
    if t:
        h["X-Trace-Id"] = t
    return h


async def embed_texts(
    texts: list[str],
    *,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
) -> list[list[float]]:
    """Embed a batch of strings using configured embeddings API."""
    if not texts:
        return []
    m = model or get_embedding_model()
    base = (base_url or get_embedding_url()).rstrip("/")
    url = f"{base}/v1/embeddings"
    payload = {"model": m, "input": texts}
    headers = _embed_headers(request_id, session_id, trace_id=trace_id)
    with bind_request_context(request_id, session_id, trace_id=trace_id):
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
    try:
        rows = data["data"]
        out = [row["embedding"] for row in rows]
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"Unexpected embedding response shape: {data!r}") from e

    for vec in out:
        if not isinstance(vec, list):
            raise RuntimeError("Embedding API returned non-list embedding.")
        if len(vec) != VECTOR_SIZE:
            raise RuntimeError(
                f"Embedding length mismatch: got {len(vec)} expected {VECTOR_SIZE}."
            )
    logger.info("embed_texts ok url=%s model=%s count=%s", url, m, len(out))
    return out


async def embed_text(
    text: str,
    *,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
) -> list[float]:
    """Embed one string and return a single vector."""
    rows = await embed_texts(
        [text],
        request_id=request_id,
        session_id=session_id,
        trace_id=trace_id,
        model=model,
        base_url=base_url,
        timeout=timeout,
    )
    if not rows:
        raise RuntimeError("Embedding API returned empty data.")
    return rows[0]
