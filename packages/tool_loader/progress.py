"""Lightweight per-tool progress reporting for synchronous tool execution."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Callable, Dict, Optional


ProgressCallback = Callable[[Dict[str, Any]], None]

_TOOL_PROGRESS_CALLBACK: ContextVar[Optional[ProgressCallback]] = ContextVar(
    "tool_progress_callback",
    default=None,
)


def set_tool_progress_callback(callback: Optional[ProgressCallback]) -> Token[Optional[ProgressCallback]]:
    return _TOOL_PROGRESS_CALLBACK.set(callback)


def reset_tool_progress_callback(token: Token[Optional[ProgressCallback]]) -> None:
    _TOOL_PROGRESS_CALLBACK.reset(token)


def report_tool_progress(**payload: Any) -> None:
    callback = _TOOL_PROGRESS_CALLBACK.get()
    if callback is None:
        return
    callback({key: value for key, value in payload.items() if value is not None})
