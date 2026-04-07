"""Generate embeddings via remote /v1/embeddings HTTP API (async)."""
import httpx

from app.config import (
    VECTOR_SIZE,
    get_embedding_internal_key,
    get_embedding_model,
    get_embedding_url,
)
from app.logging_config import logger
from app.request_context import bind_request_context


def _require_trace_ids(request_id: str, session_id: str) -> tuple[str, str]:
    rid = (request_id or "").strip()
    sid = (session_id or "").strip()
    if not rid:
        raise ValueError("request_id is required (embedding header X-Request-Id).")
    if not sid:
        raise ValueError("session_id is required (embedding header X-Session-Id).")
    return rid, sid


def _request_headers(*, request_id: str, session_id: str) -> dict[str, str]:
    """Embedding gateway: X-Internal-Key (if set), X-Request-Id, X-Session-Id."""
    rid, sid = _require_trace_ids(request_id, session_id)
    h: dict[str, str] = {
        "X-Request-Id": rid,
        "X-Session-Id": sid,
    }
    key = get_embedding_internal_key()
    if key:
        h["X-Internal-Key"] = key
    return h


async def embed_text(
    text: str,
    *,
    request_id: str,
    session_id: str,
) -> list[float]:
    """Embed a single text."""
    vecs = await embed_texts([text], request_id=request_id, session_id=session_id)
    return vecs[0]


async def embed_texts(
    texts: list[str],
    *,
    request_id: str,
    session_id: str,
) -> list[list[float]]:
    """Embed texts via /v1/embeddings API (async)."""
    if not texts:
        return []

    with bind_request_context(request_id, session_id):
        url = f"{get_embedding_url().rstrip('/')}/v1/embeddings"
        model = get_embedding_model()
        logger.info("embed_texts start count=%s url=%s", len(texts), url)

        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {"model": model, "input": texts if len(texts) > 1 else texts[0]}
            resp = await client.post(
                url,
                json=payload,
                headers=_request_headers(request_id=request_id, session_id=session_id),
            )
            if resp.status_code == 200:
                data = resp.json()
                embeddings = [
                    e["embedding"] for e in sorted(data["data"], key=lambda x: x["index"])
                ]
            else:
                logger.info(
                    "embed_texts batch status=%s falling back to per-text requests",
                    resp.status_code,
                )
                embeddings = []
                for text in texts:
                    r = await client.post(
                        url,
                        json={"model": model, "input": text},
                        headers=_request_headers(request_id=request_id, session_id=session_id),
                    )
                    r.raise_for_status()
                    embeddings.append(r.json()["data"][0]["embedding"])

        for emb in embeddings:
            if len(emb) != VECTOR_SIZE:
                raise ValueError(
                    f"Embedding dim {len(emb)} != VECTOR_SIZE {VECTOR_SIZE}. "
                    "Set VECTOR_SIZE in .env to match your embedding model."
                )
        logger.info("embed_texts ok count=%s", len(embeddings))
        return embeddings
