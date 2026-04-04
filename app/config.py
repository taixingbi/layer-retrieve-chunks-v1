"""
Load `.env` from the project root into ``os.environ``, then read configuration.

The project root ``.env`` must exist and define every name in ``REQUIRED_ENV_VARS`` (values may
be empty where allowed). Pass ``request_id`` and ``session_id`` into ``embed_text`` / ``query_chunks``
for embedding ``X-Request-Id`` and ``X-Session-Id``.

Optional: ``INFERENCE_URL``, ``INFERENCE_MODEL``, ``INFERENCE_MAX_TOKENS`` (see ``get_inference_*``).
"""
from pathlib import Path

import os

from dotenv import dotenv_values, load_dotenv

from app.logging_config import setup_logging

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

REQUIRED_ENV_VARS: tuple[str, ...] = (
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "EMBEDDING_URL",
    "EMBEDDING_INTERNAL_KEY",
    "EMBEDDING_MODEL",
    "VECTOR_SIZE",
    "ENV",
    "TOP_K_DENSE",
    "RRF_K",
)


def _ensure_dotenv() -> None:
    if not _ENV_PATH.is_file():
        raise FileNotFoundError(
            f"Missing `.env` at {_ENV_PATH}. Copy `.env.example` to `.env` and set each variable."
        )
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


_ensure_dotenv()
setup_logging()


def get_qdrant_url() -> str:
    return os.environ["QDRANT_URL"]


def get_qdrant_api_key() -> str:
    return os.environ["QDRANT_API_KEY"]


def get_embedding_url() -> str:
    return os.environ["EMBEDDING_URL"]


def get_embedding_model() -> str:
    return os.environ["EMBEDDING_MODEL"]


def get_embedding_internal_key() -> str:
    return os.environ["EMBEDDING_INTERNAL_KEY"]


def get_env() -> str:
    """Deploy suffix ``dev`` / ``qa`` / ``prod``; empty string means no ``_suffix`` on collection base."""
    return os.environ["ENV"]


TOP_K_DENSE = int(os.environ["TOP_K_DENSE"])
RRF_K = int(os.environ["RRF_K"])
VECTOR_SIZE = int(os.environ["VECTOR_SIZE"])

_DEFAULT_INFERENCE_URL = "http://192.168.86.179:30080"
_DEFAULT_INFERENCE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_DEFAULT_INFERENCE_MAX_TOKENS = 512


def get_inference_url() -> str:
    """OpenAI-compatible chat base URL (no trailing slash). Optional in ``.env``."""
    return os.environ.get("INFERENCE_URL", _DEFAULT_INFERENCE_URL).rstrip("/")


def get_inference_model() -> str:
    """Chat model name for ``/v1/chat/completions``. Optional in ``.env``."""
    return os.environ.get("INFERENCE_MODEL", _DEFAULT_INFERENCE_MODEL)


def get_inference_max_tokens() -> int:
    """Default max output tokens for chat. Optional ``INFERENCE_MAX_TOKENS`` in ``.env``."""
    return int(os.environ.get("INFERENCE_MAX_TOKENS", str(_DEFAULT_INFERENCE_MAX_TOKENS)))
