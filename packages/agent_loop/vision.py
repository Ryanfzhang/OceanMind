"""Keep image references in graph state and materialize pixels only for the model."""

from __future__ import annotations

import base64
import json
from typing import Any, Callable

from packages.agent_loop.analysis import AnalysisSession


VIEW_IMAGE_SCHEMA = {
    "type": "function", "function": {
        "name": "view_image",
        "description": "Inspect a completed PNG analysis artifact with its date, axes and color context.",
        "parameters": {"type": "object", "properties": {
            "artifact_id": {"type": "string"},
        }, "required": ["artifact_id"], "additionalProperties": False},
    },
}

_MARKER = "oceanmind_view_image"


def make_view_image(session: AnalysisSession) -> Callable[[str], dict[str, Any]]:
    """Validate a run-owned image; the tool observation contains no image bytes."""
    def view_image(artifact_id: str) -> dict[str, Any]:
        # The session method performs run/attempt ownership checks. Read the
        # stored summary afterward because its generic text view may truncate
        # rich figure context to fit a short tool observation.
        session.read_artifact(artifact_id)
        metadata = session.artifacts.read_artifact(artifact_id)
        if (metadata.get("status") != "completed" or
                metadata.get("kind") != "image_png"):
            raise ValueError("Image artifact is missing or incomplete")
        # Confirm the exact saved payload is still present before telling the
        # agent it can be observed. The second read before model call guards
        # against replacement between tool execution and hydration.
        session.artifacts.read_image(artifact_id)
        return {_MARKER: artifact_id, "status": "ready",
                "image": metadata["summary"]}

    return view_image


def hydrate_vision_messages(
    messages: list[dict[str, Any]], session: AnalysisSession | None,
) -> list[dict[str, Any]]:
    """Attach saved PNG bytes in a transient user message after a view tool call.

    DeepSeek Chat Completions accepts image_url blocks in user messages, not
    system, assistant or tool messages. Only the most recent assistant tool
    batch is hydrated, so a long loop does not resend historical images on
    every model call. The returned list is never persisted as graph state.
    """
    if session is None:
        return messages
    last_assistant = max((i for i, item in enumerate(messages)
                          if item.get("role") == "assistant"), default=-1)
    if last_assistant < 0:
        return messages
    view_ids = {
        call.get("id") for call in (messages[last_assistant].get("tool_calls") or [])
        if call.get("function", {}).get("name") == "view_image"
    }
    if not view_ids:
        return messages
    parts: list[dict[str, Any]] = []
    for item in messages[last_assistant + 1:]:
        if item.get("role") != "tool" or item.get("tool_call_id") not in view_ids:
            continue
        try:
            observation = json.loads(item.get("content", ""))
        except (TypeError, ValueError):
            continue
        if not isinstance(observation, dict) or observation.get("status") != "ready":
            continue
        artifact_id = observation.get(_MARKER)
        if not isinstance(artifact_id, str):
            continue
        verified = make_view_image(session)(artifact_id)
        image_bytes = session.artifacts.read_image(artifact_id)
        label = json.dumps({"artifact_id": artifact_id,
                            "metadata": verified["image"]}, ensure_ascii=False)
        parts.extend((
            {"type": "text", "text": "Analysis image and plotting context: " + label},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
                "detail": "original",
            }},
        ))
    if not parts:
        return messages
    return [*messages, {"role": "user", "content": parts}]
