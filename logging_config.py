"""
Structured logging configuration for Loki/Grafana.
Uses structlog with JSON output (LOG_JSON=true) or pretty console (LOG_JSON=false).
When LOG_FILE is set, logs are also written to that file for Promtail to scrape.
"""
import logging
import sys

import structlog

from config import LOG_FILE, LOG_JSON, LOG_LEVEL


class _TeeWriter:
    """Writes to multiple streams."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def configure_logging() -> None:
    """Configure structlog for the application."""
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if LOG_JSON:
        processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        processors = shared_processors + [structlog.dev.ConsoleRenderer()]

    out = sys.stdout
    if LOG_FILE:
        try:
            log_file = open(LOG_FILE, "a", encoding="utf-8")
            out = _TeeWriter(sys.stdout, log_file)
        except OSError:
            pass  # Fall back to stdout only

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=out),
        cache_logger_on_first_use=True,
    )
