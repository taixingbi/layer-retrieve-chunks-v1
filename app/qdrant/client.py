"""Async Qdrant client construction and connection parameters."""
from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from app.config import get_qdrant_api_key, get_qdrant_url


def resolve_connection_params(
    qdrant_url: str | None,
    qdrant_api_key: str | None,
) -> tuple[str, str | None]:
    """URL and API key from overrides or ``.env`` (``QDRANT_URL`` / ``QDRANT_API_KEY``)."""
    u = qdrant_url if qdrant_url is not None else get_qdrant_url()
    k = qdrant_api_key if qdrant_api_key is not None else get_qdrant_api_key()
    return u, k or None


def create_async_client(
    url: str,
    api_key: str | None = None,
    *,
    check_compatibility: bool = False,
) -> AsyncQdrantClient:
    """New ``AsyncQdrantClient``; caller must ``await client.close()`` when done (if not using a shared instance)."""
    return AsyncQdrantClient(
        url=url,
        api_key=api_key,
        check_compatibility=check_compatibility,
    )
