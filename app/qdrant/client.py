"""Async Qdrant client construction and connection resolution."""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from app.config import get_qdrant_api_key, get_qdrant_url


def resolve_connection_params(
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
) -> tuple[str, str]:
    """
    Resolve Qdrant connection settings.

    Explicit args override environment settings from ``app.config``.
    """
    url = (qdrant_url or get_qdrant_url()).strip()
    if not url:
        raise ValueError("Qdrant URL is required.")
    key = qdrant_api_key if qdrant_api_key is not None else get_qdrant_api_key()
    return url, (key or "")


def create_async_client(url: str, api_key: str | None) -> AsyncQdrantClient:
    """Create an ``AsyncQdrantClient`` with optional API key."""
    k = (api_key or "").strip()
    return AsyncQdrantClient(url=url, api_key=k or None)
