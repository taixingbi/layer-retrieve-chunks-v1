"""Helpers for calling async code from sync MCP / CLI entry points."""
from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[object, object, T]) -> T:
    """
    Run ``coro`` when no event loop is running (e.g. stdio MCP tools).

    From an ``async def`` route or task, use ``await`` on the coroutine instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "run_async() cannot be used with a running event loop; await the coroutine instead"
    )
