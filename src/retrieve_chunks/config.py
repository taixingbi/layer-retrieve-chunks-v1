"""
Configuration for query (Qdrant URL, API key, collection, embeddings).
Loads .env if present; values can be overridden by environment variables or configure().
"""
import os

from dotenv import load_dotenv

load_dotenv()

_overrides: dict = {}

DEFAULT_QDRANT_URL = "http://192.168.86.173:6333"


def get_qdrant_url() -> str:
    """Qdrant URL: override > env > default."""
    return _overrides.get("qdrant_url") or os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)


def get_qdrant_api_key() -> str:
    """Qdrant API key: override > env > default."""
    return _overrides.get("qdrant_api_key", os.getenv("QDRANT_API_KEY", ""))


def configure(
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    embedding_url: str | None = None,
    embedding_model: str | None = None,
    collection_name: str | None = None,
    env: str | None = None,
    **kwargs,
) -> None:
    """
    Set configuration overrides (used before first query).
    Resets cached Qdrant client so new config takes effect.
    """
    global _overrides
    if qdrant_url is not None:
        _overrides["qdrant_url"] = qdrant_url
    if qdrant_api_key is not None:
        _overrides["qdrant_api_key"] = qdrant_api_key
    if embedding_url is not None:
        _overrides["embedding_url"] = embedding_url
    if embedding_model is not None:
        _overrides["embedding_model"] = embedding_model
    if collection_name is not None:
        _overrides["collection_name"] = collection_name
    if env is not None:
        _overrides["env"] = env
    for k, v in kwargs.items():
        if v is not None:
            _overrides[k] = v
    # Reset cached client so new config is used
    import retrieve_chunks.query as _query

    _query._reset_client()


def get_embedding_url() -> str:
    """Embedding API URL: override > env > default."""
    return _overrides.get("embedding_url") or os.getenv(
        "EMBEDDING_URL", "http://192.168.86.173:8001"
    )


def get_collection_name() -> str:
    """Collection name: override > env > default."""
    return _overrides.get("collection_name") or os.getenv("COLLECTION_NAME", "rag_dev")


def get_embedding_model() -> str:
    """Embedding model: override > env > default."""
    return _overrides.get("embedding_model") or os.getenv(
        "EMBEDDING_MODEL", "BAAI/bge-m3"
    )


# Embedding (local v1/embeddings API) - legacy
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://192.168.86.173:8001")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Query defaults
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_dev")
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))

# Vector size (BAAI/bge-m3 outputs 1024)
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))
