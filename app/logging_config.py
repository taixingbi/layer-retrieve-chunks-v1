"""Stderr JSON logging for ``layer_rag.query``."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from .request_context import (
    get_http_method,
    get_http_path,
    get_http_status,
    get_request_id,
    get_session_id,
)

logger = logging.getLogger("layer_rag.query")

_LOG_TZ = ZoneInfo("America/New_York")
# Merged onto JSON when present on the LogRecord (from logger.*(..., extra={...})).
_EXTRA_JSON_FIELDS = (
    "duration_ms",
    "latency_total_ms",
    "latency_embed_ms",
    "latency_retrieve_ms",
    "latency_chunk_rerank_ms",
    "latency_chat_ms",
    "latency_follow_up_chat_ms",
    "latency_follow_up_rerank_ms",
    "backend",
    "gpu",
    "reason",
    "upstream_status",
    "error_type",
    "error_message",
    "missing",
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"


class _RequestContextFilter(logging.Filter):
    """Attach request/session IDs and HTTP method/path/status from context onto each ``LogRecord``."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Populate correlation and HTTP fields; always return True so the record is emitted."""
        rid = get_request_id()
        sid = get_session_id()
        # Outside request context, ids default to "-".
        # When request_id is missing, keep session_id as "-" in logs (no partial correlation).
        record.request_id = "-" if rid == "-" else rid
        record.session_id = "-" if rid == "-" else sid
        record.method = get_http_method()
        record.path = get_http_path()
        # ASGI response.start runs after the route returns; logs inside the route
        # must pass status via logger.info(..., extra={"status": ...}) or it stays "-".
        ctx_status = get_http_status()
        if ctx_status != "-":
            record.status = ctx_status
        elif not hasattr(record, "status"):
            record.status = "-"
        return True


class _JsonFormatter(logging.Formatter):
    """One JSON object per line for stderr."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize ``record`` to a single JSON line; include ``error`` only when ``exc_info`` is set."""
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=_LOG_TZ).isoformat(),
            "level": record.levelname,
            "request_id": getattr(record, "request_id", "-"),
            "session_id": getattr(record, "session_id", "-"),
            "method": getattr(record, "method", "-"),
            "path": getattr(record, "path", "-"),
            "status": getattr(record, "status", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        for key in _EXTRA_JSON_FIELDS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


_JSON_FORMATTER = _JsonFormatter()


def setup_logging() -> None:
    """Configure ``layer_rag.query`` logger: stderr JSON."""
    load_dotenv(_ENV_PATH)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.filters.clear()
    logger.propagate = False
    logger.addFilter(_RequestContextFilter())

    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.INFO)
    stderr.setFormatter(_JSON_FORMATTER)
    logger.addHandler(stderr)

    logger.info("logging configured (stderr JSON)")


def shutdown_logging() -> None:
    """No-op; retained for callers that may stop logging explicitly."""
