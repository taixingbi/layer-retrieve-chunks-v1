"""
Load `.env` from the project root into ``os.environ``, then read configuration.

The project root ``.env`` must exist and define every name in ``REQUIRED_ENV_VARS`` (values may
be empty where allowed). Pass ``request_id`` and ``session_id`` into ``embed_text`` / ``query_chunks``
for embedding ``X-Request-Id`` and ``X-Session-Id``.

Optional: ``INFERENCE_URL``, ``INFERENCE_MODEL``, ``INFERENCE_MAX_TOKENS``,
``RERANK_URL``, ``RERANK_MODEL``, ``RERANK_TOP_N``, ``RERANK_RETURN_TOP_K``,
``RETRIEVE_FALLBACK_N``, ``FINAL_CONTEXT_TOP_K`` (see getters).
"""
from pathlib import Path

import os

from dotenv import dotenv_values, load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

REQUIRED_ENV_VARS: tuple[str, ...] = (
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "EMBEDDING_URL",
    "EMBEDDING_MODEL",
    "VECTOR_SIZE",
    "ENV",
    "TOP_K_DENSE",
    "RRF_K",
)


def _ensure_dotenv() -> None:
    """Load ``.env`` from project root, or accept variables already set (Docker / k8s)."""
    if _ENV_PATH.is_file():
        file_vals = dotenv_values(_ENV_PATH)
        missing_file = [k for k in REQUIRED_ENV_VARS if k not in file_vals]
        if missing_file:
            raise ValueError(
                "`.env` is missing required keys. "
                f"Add: {', '.join(sorted(missing_file))}. See `.env.example`."
            )
        load_dotenv(_ENV_PATH)
        missing_env = [k for k in REQUIRED_ENV_VARS if k not in os.environ]
        if missing_env:
            raise ValueError(
                "Required variables were not loaded into the environment after reading `.env`. "
                f"Check for syntax errors in `{_ENV_PATH}`. Missing: {', '.join(sorted(missing_env))}"
            )
        return

    # No on-disk `.env` (e.g. Docker Compose `env_file`, k3s Secrets → env): require full set in os.environ.
    missing_env = [k for k in REQUIRED_ENV_VARS if k not in os.environ]
    if missing_env:
        raise ValueError(
            f"No `.env` file at {_ENV_PATH} and the following required variables are unset in the "
            f"environment: {', '.join(sorted(missing_env))}. Copy `.env.example` to `.env`, or inject "
            "these keys via your orchestrator."
        )


_ensure_dotenv()

# After project `.env` is in os.environ: tb-loki's package import runs load_dotenv_cwd();
# importing logging_config before _ensure_dotenv can miss GRAFANA_* when CWD has no `.env`.
from app.logging_config import setup_logging

setup_logging()


def get_qdrant_url() -> str:
    return os.environ["QDRANT_URL"]


def get_qdrant_api_key() -> str:
    return os.environ["QDRANT_API_KEY"]


def get_embedding_url() -> str:
    return os.environ["EMBEDDING_URL"]


def get_embedding_model() -> str:
    return os.environ["EMBEDDING_MODEL"]


def get_env() -> str:
    """Deploy suffix ``dev`` / ``qa`` / ``prod``; empty string means no ``_suffix`` on collection base."""
    return os.environ["ENV"]


TOP_K_DENSE = int(os.environ["TOP_K_DENSE"])
RRF_K = int(os.environ["RRF_K"])
VECTOR_SIZE = int(os.environ["VECTOR_SIZE"])

_DEFAULT_INFERENCE_URL = "http://192.168.86.179:30080"
_DEFAULT_INFERENCE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_DEFAULT_INFERENCE_MAX_TOKENS = 512
_DEFAULT_RERANK_URL = "http://localhost:30080"
_DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
_DEFAULT_RERANK_TOP_N = 50
_DEFAULT_RERANK_RETURN_TOP_K = 25
_DEFAULT_RETRIEVE_FALLBACK_N = 3
_DEFAULT_FINAL_CONTEXT_TOP_K = 12


def get_inference_url() -> str:
    """OpenAI-compatible chat base URL (no trailing slash). Optional in ``.env``."""
    return os.environ.get("INFERENCE_URL", _DEFAULT_INFERENCE_URL).rstrip("/")


def get_inference_model() -> str:
    """Chat model name for ``/v1/chat/completions``. Optional in ``.env``."""
    return os.environ.get("INFERENCE_MODEL", _DEFAULT_INFERENCE_MODEL)


def get_inference_max_tokens() -> int:
    """Default max output tokens for chat. Optional ``INFERENCE_MAX_TOKENS`` in ``.env``."""
    return int(os.environ.get("INFERENCE_MAX_TOKENS", str(_DEFAULT_INFERENCE_MAX_TOKENS)))


def get_rerank_url() -> str:
    """OpenAI-compatible rerank base URL (no trailing slash). Optional in ``.env``."""
    return os.environ.get("RERANK_URL", _DEFAULT_RERANK_URL).rstrip("/")


def get_rerank_model() -> str:
    """Reranker model name for ``/v1/rerank``. Optional in ``.env``."""
    return os.environ.get("RERANK_MODEL", _DEFAULT_RERANK_MODEL)


def get_rerank_top_n() -> int:
    """Fused-retrieval candidates sent into the reranker (document count). Optional in ``.env``."""
    return int(os.environ.get("RERANK_TOP_N", str(_DEFAULT_RERANK_TOP_N)))


def get_rerank_return_top_k() -> int:
    """``top_n`` for ``/v1/rerank``: how many ranked passages to keep (ordering / recall before prompt cap)."""
    return int(os.environ.get("RERANK_RETURN_TOP_K", str(_DEFAULT_RERANK_RETURN_TOP_K)))


def get_retrieve_fallback_n() -> int:
    """After rerank, append up to this many extra chunks from raw RRF order (ids not already in rerank list)."""
    return int(os.environ.get("RETRIEVE_FALLBACK_N", str(_DEFAULT_RETRIEVE_FALLBACK_N)))


def get_final_context_top_k() -> int:
    """Max passages in one chat context (initial ``k`` widen cap). Optional in ``.env``."""
    return int(os.environ.get("FINAL_CONTEXT_TOP_K", str(_DEFAULT_FINAL_CONTEXT_TOP_K)))
