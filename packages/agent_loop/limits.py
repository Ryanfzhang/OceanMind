"""Small, testable guards for the single-agent loop."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any


def budget_violation(
    state: Mapping[str, Any], max_rounds: int, *, now: float | None = None
) -> dict[str, str] | None:
    """Return an incomplete status before another model or tool call."""
    deadline = state.get("deadline")
    if deadline is not None and (time.monotonic() if now is None else now) >= deadline:
        return {"status": "incomplete", "termination_reason": "deadline_exceeded"}
    if state["rounds"] >= max_rounds:
        return {"status": "incomplete", "termination_reason": "max_rounds_exceeded"}
    return None


def _canonical_arguments(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return raw


def repeated_failure_count(messages: Sequence[Mapping[str, Any]]) -> int:
    """Count the trailing identical failures, matched to their tool call IDs."""
    pending: dict[str, tuple[str, str]] = {}
    previous: tuple[str, str, str] | None = None
    count = 0
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call["function"]
                pending[call["id"]] = (
                    function["name"],
                    _canonical_arguments(function["arguments"]),
                )
        elif message.get("role") == "tool":
            action = pending.pop(message.get("tool_call_id"), None)
            if action is None:
                continue
            try:
                payload = json.loads(message["content"])
            except (TypeError, ValueError):
                payload = None
            error = payload.get("error") if isinstance(payload, dict) else None
            if error is None and isinstance(payload, dict) and payload.get("status") in {
                "failed", "timed_out",
            }:
                error = {"status": payload["status"], "exit_code": payload.get("exit_code")}
            if error is None:
                previous, count = None, 0
                continue
            fingerprint = (*action, json.dumps(error, sort_keys=True, ensure_ascii=False))
            count = count + 1 if fingerprint == previous else 1
            previous = fingerprint
    return count
