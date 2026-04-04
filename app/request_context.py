"""Request-scoped ids and HTTP metadata for log correlation (contextvars)."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
_session_id_ctx: ContextVar[str] = ContextVar("session_id", default="-")
_http_method_ctx: ContextVar[str] = ContextVar("http_method", default="-")
_http_path_ctx: ContextVar[str] = ContextVar("http_path", default="-")
_http_status_ctx: ContextVar[str] = ContextVar("http_status", default="-")


def get_request_id() -> str:
    return _request_id_ctx.get()


def get_session_id() -> str:
    return _session_id_ctx.get()


def get_http_method() -> str:
    return _http_method_ctx.get()


def get_http_path() -> str:
    return _http_path_ctx.get()


def get_http_status() -> str:
    return _http_status_ctx.get()


@contextmanager
def bind_request_context(request_id: str, session_id: str):
    """Bind trace ids for the current call (embedding / retrieval / RAG)."""
    rid = (request_id or "").strip() or "-"
    sid = (session_id or "").strip() or "-"
    t_rid = _request_id_ctx.set(rid)
    t_sid = _session_id_ctx.set(sid)
    try:
        yield
    finally:
        _request_id_ctx.reset(t_rid)
        _session_id_ctx.reset(t_sid)


@contextmanager
def bind_http_context(
    method: str,
    path: str,
    *,
    status: str = "-",
):
    """Bind HTTP method/path/status for ASGI or MCP HTTP handlers (optional)."""
    t_m = _http_method_ctx.set((method or "").strip() or "-")
    t_p = _http_path_ctx.set((path or "").strip() or "-")
    t_s = _http_status_ctx.set((status or "").strip() or "-")
    try:
        yield
    finally:
        _http_method_ctx.reset(t_m)
        _http_path_ctx.reset(t_p)
        _http_status_ctx.reset(t_s)
