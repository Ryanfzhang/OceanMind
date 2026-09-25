"""Small native tool registry for the first agent loop."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import Any


CALCULATOR_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Add or multiply two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "multiply"]},
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
            "additionalProperties": False,
        },
    },
}

TOOL_SCHEMAS = [CALCULATOR_SCHEMA]


def calculator(*, operation: str, a: float, b: float) -> dict[str, float]:
    """Evaluate one simple calculation and return a JSON-compatible result."""
    if operation not in ("add", "multiply"):
        raise ValueError("operation must be 'add' or 'multiply'")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (a, b)):
        raise ValueError("a and b must be numbers")
    if not all(math.isfinite(value) for value in (a, b)):
        raise ValueError("a and b must be finite")
    result = a + b if operation == "add" else a * b
    if not math.isfinite(result):
        raise ValueError("result must be finite")
    return {"result": result}


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {"calculator": calculator}


def execute_tool_calls(
    assistant_message: Mapping[str, Any],
    registry: Mapping[str, Callable[..., Any]] | None = None,
) -> list[dict[str, str]]:
    """Return one tool observation per call, preserving order and call IDs."""
    available = TOOL_REGISTRY if registry is None else registry
    observations = []
    for call in assistant_message.get("tool_calls") or []:
        name = call["function"]["name"]
        try:
            if name not in available:
                payload = {"error": {"type": "unknown_tool", "message": f"Unknown tool: {name}"}}
            else:
                arguments = json.loads(call["function"]["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be a JSON object")
                payload = available[name](**arguments)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            payload = {"error": {"type": "invalid_arguments", "message": str(exc)}}
        except Exception as exc:
            payload = {"error": {"type": "tool_error", "message": str(exc)}}
        try:
            content = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            content = json.dumps({"error": {"type": "tool_error", "message": str(exc)}})
        observations.append({"role": "tool", "tool_call_id": call["id"], "content": content})
    return observations
