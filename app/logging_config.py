"""Stderr logging plus optional Grafana Loki via tb-loki-central-logger."""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import queue
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tb_loki_central_logger import LokiClient, basic_auth_from_env, load_dotenv

from . import __version__
from .request_context import (
    get_http_method,
    get_http_path,
    get_http_status,
    get_request_id,
    get_session_id,
)

logger = logging.getLogger("layer_rag.query")

_LOKI_LEVEL = {
    "DEBUG": "debug",
    "INFO": "info",
    "WARNING": "warn",
    "ERROR": "error",
    "CRITICAL": "critical",
}

_LOG_TZ = ZoneInfo("America/New_York")
# Merged onto JSON when present on the LogRecord (from logger.*(..., extra={...})).
_EXTRA_JSON_FIELDS = (
    "duration_ms",
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


class _SyncLokiHandler(logging.Handler):
    """
    Ship each record with LokiClient in emit().

    Used only from a ``QueueListener`` worker thread so ``push()`` never blocks the asyncio
    event loop (see ``setup_logging``).
    """

    def __init__(
        self,
        *,
        labels: dict[str, str],
        basic_auth: tuple[str, str],
        timeout: int = 15,
    ) -> None:
        super().__init__()
        self._client = LokiClient(
            labels=labels,
            timeout=timeout,
            basic_auth=basic_auth,
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _LOKI_LEVEL.get(record.levelname, "info")
            message = self.format(record)
            self._client.push(message, level=level, labels={"logger": record.name})
        except Exception:
            self.handleError(record)


class _RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
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
    """One JSON object per line for stderr and Loki."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=_LOG_TZ).isoformat(),
            "request_id": getattr(record, "request_id", "-"),
            "session_id": getattr(record, "session_id", "-"),
            "method": getattr(record, "method", "-"),
            "path": getattr(record, "path", "-"),
            "status": getattr(record, "status", "-"),
            "message": record.getMessage(),
            "error": self.formatException(record.exc_info)
            if record.exc_info
            else None,
        }
        for key in _EXTRA_JSON_FIELDS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


_JSON_FORMATTER = _JsonFormatter()

_loki_listener: logging.handlers.QueueListener | None = None
_loki_queue_handler: logging.handlers.QueueHandler | None = None
_loki_worker_handler: _SyncLokiHandler | None = None


def setup_logging() -> None:
    global _loki_listener, _loki_queue_handler, _loki_worker_handler
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

    auth = basic_auth_from_env()
    if auth is not None:
        log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)
        _loki_queue_handler = logging.handlers.QueueHandler(log_queue)
        _loki_queue_handler.setLevel(logging.INFO)
        logger.addHandler(_loki_queue_handler)

        _loki_worker_handler = _SyncLokiHandler(
            labels={
                "service": "layer-rag-query",
                "component": "query",
                "env": os.getenv("ENV", "dev"),
                "version": __version__,
            },
            basic_auth=auth,
        )
        _loki_worker_handler.setLevel(logging.INFO)
        _loki_worker_handler.setFormatter(_JSON_FORMATTER)
        _loki_listener = logging.handlers.QueueListener(
            log_queue,
            _loki_worker_handler,
            respect_handler_level=True,
        )
        _loki_listener.start()
        logger.info("centralized Loki logging enabled")
    else:
        logger.info(
            "Loki disabled (set GRAFANA_CLOUD_WRITE_API_KEY to ship logs to Grafana)"
        )


def shutdown_logging() -> None:
    global _loki_listener, _loki_queue_handler, _loki_worker_handler
    if _loki_listener is not None:
        _loki_listener.stop()
        _loki_listener = None
    if _loki_queue_handler is not None:
        logger.removeHandler(_loki_queue_handler)
        _loki_queue_handler.close()
        _loki_queue_handler = None
    if _loki_worker_handler is not None:
        _loki_worker_handler.close()
        _loki_worker_handler = None
