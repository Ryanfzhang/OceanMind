"""Classify the outcome of one model and tool loop."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from packages.agent_loop.state import AgentState
from packages.analysis_runtime.artifacts import _stored_payload


DELIVERY_REF = re.compile(r"\b(?:code_[0-9a-f]{32}|(?:run_[a-z0-9_]+_)?artifact_[0-9a-f]{32})\b")


REQUEST_CLARIFICATION_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "request_clarification",
        "description": "Ask the user one specific question needed to continue.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}


def clarification_request(assistant_message: Mapping[str, Any]) -> tuple[str, str] | None:
    """Read a native clarification call; reject malformed or mixed calls."""
    calls = assistant_message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ValueError("tool_calls must be a list")
    matches = [
        call for call in calls
        if isinstance(call, Mapping)
        and isinstance(call.get("function"), Mapping)
        and call["function"].get("name") == "request_clarification"
    ]
    if not matches:
        return None
    if len(calls) != 1:
        raise ValueError("request_clarification must be the only tool call")
    call = matches[0]
    try:
        arguments = json.loads(call["function"]["arguments"])
        question = arguments["question"]
        call_id = call["id"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("request_clarification requires a question and call ID") from exc
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("request_clarification requires a nonempty call ID")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("request_clarification requires a nonempty question")
    return call_id, question.strip()


def finalize_state(state: AgentState) -> AgentState:
    """Always return a readable delivery unless the agent requested user input."""
    if state["status"] in {"completed", "needs_input", "failed"}:
        return state.copy()
    final = state["messages"][-1] if state["messages"] else {}
    if state["status"] == "running" and isinstance(final, Mapping):
        try:
            question = clarification_request(final)
        except ValueError:
            question = None
        if question:
            return {**state, "status": "needs_input",
                    "termination_reason": "clarification_requested"}
        content = final.get("content")
        if (final.get("role") == "assistant" and not final.get("tool_calls")
                and isinstance(content, str) and content.strip()):
            return {**state, "status": "completed", "termination_reason": None}

    draft = state.get("draft", "").strip()
    reason = state.get("termination_reason") or "the answer was not finished"
    if draft:
        content = f"{draft}\n\nFurther analysis was interrupted ({reason})."
    else:
        content = f"I could not complete the analysis ({reason}). Any saved results remain available."
    return {**state, "messages": [*state["messages"],
                                 {"role": "assistant", "content": content}],
            "status": "completed"}


def finalize_delivery(state: AgentState, session: Any) -> AgentState:
    """Check cited references and attach verified results from all attempts."""
    classified = finalize_state(state)
    if classified["status"] != "completed":
        return classified
    content = classified["messages"][-1]["content"]
    invalid: list[str] = []
    for ref in set(DELIVERY_REF.findall(content)):
        try:
            if ref.startswith("code_"):
                session.codes.get_path(ref)
            else:
                metadata = session.artifacts.read_artifact(ref)
                if (metadata["status"] != "completed"
                        or metadata.get("run_id") != session.run_id):
                    raise ValueError("Artifact is not a completed result in this run")
                attempt = session.records.read("attempt", metadata["attempt_id"])
                if attempt["run_id"] != session.run_id:
                    raise ValueError("Artifact attempt belongs to another run")
                if metadata.get("call_id"):
                    call = session.records.read("call", metadata["call_id"])
                    if call["status"] != "completed" or ref not in call["artifact_ids"]:
                        raise ValueError("Artifact has no completed call record")
                if metadata.get("kind") == "image_png":
                    session.artifacts.read_image(ref)
                elif metadata.get("kind") in {"json", "dataarray_netcdf"}:
                    _stored_payload(session.root, metadata["payload"])
                elif metadata.get("kind") == "source_netcdf":
                    session.artifacts._source(metadata["source_path"])
                else:
                    raise ValueError("Unknown artifact format")
        except (KeyError, OSError, TypeError, ValueError):
            invalid.append(ref)
    if invalid:
        # The backend supplies verified attachments below. A mistyped long ID in
        # prose must not trigger another execution of an already successful run.
        cleaned = content
        for ref in invalid:
            cleaned = cleaned.replace(ref, "[unverified reference omitted]")
        messages = [*classified["messages"]]
        messages[-1] = {**messages[-1], "content": cleaned}
        classified = {**classified, "messages": messages}

    attempts = []
    directory = session.root / "records" / "attempt"
    if directory.exists():
        for path in directory.glob(f"{session.run_id}_attempt_*.json"):
            attempt = session.records.read("attempt", path.stem)
            if attempt.get("run_id") == session.run_id:
                attempts.append(attempt)
    attachments: list[dict[str, str]] = []
    if attempts:
        attempts.sort(key=lambda item: item.get("created_at", ""))
        attempt = next((item for item in reversed(attempts)
                        if item.get("status") == "completed"), attempts[-1])
        code_id = attempt.get("code_version")
        if isinstance(code_id, str):
            session.codes.get_path(code_id)
            attachments.append({"kind": "code", "ref": code_id})
        result_dir = session.root / "artifacts" / "index"
        if result_dir.exists():
            artifacts = []
            for saved_attempt in attempts:
                for path in result_dir.glob(f"{saved_attempt['attempt_id']}_artifact_*.json"):
                    metadata = session.artifacts.read_artifact(path.stem)
                    if (metadata.get("status") == "completed"
                            and metadata.get("run_id") == session.run_id
                            and metadata.get("attempt_id") == saved_attempt["attempt_id"]):
                        try:
                            if metadata.get("kind") == "image_png":
                                session.artifacts.read_image(metadata["artifact_id"])
                            elif metadata.get("kind") in {"json", "dataarray_netcdf"}:
                                _stored_payload(session.root, metadata["payload"])
                            elif metadata.get("kind") == "source_netcdf":
                                session.artifacts._source(metadata["source_path"])
                            else:
                                continue
                        except (KeyError, OSError, TypeError, ValueError):
                            continue
                        artifacts.append(metadata)
            artifacts.sort(key=lambda item: item.get("created_at", ""))
            images = [item for item in artifacts if item.get("kind") == "image_png"]
            ordinary = [item for item in artifacts if item.get("kind") != "image_png"]
            for item in ordinary[-3:] + images[-2:]:
                attachments.append({"kind": item["kind"], "ref": item["artifact_id"]})
    return {**classified, "attachments": attachments}
