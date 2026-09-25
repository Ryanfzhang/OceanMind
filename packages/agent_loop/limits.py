"""Small, testable guards for the single-agent loop."""

from __future__ import annotations

import time
from collections.abc import Mapping
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
