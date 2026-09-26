"""Minimal state shared by the single-agent LangGraph loop."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from packages.agent_loop.language import Language, preferred_language

Message = dict[str, Any]
RunStatus = Literal["running", "completed", "needs_input", "incomplete", "failed"]


class AgentState(TypedDict):
    """Keep message history in OpenAI-compatible order and shape."""

    messages: list[Message]
    rounds: int
    status: RunStatus
    termination_reason: str | None
    deadline: float | None
    run_id: str | None
    run_root: str | None
    attachments: list[dict[str, str]]
    draft: str
    language: Language
    answer_active: bool
    failure_explained: bool
    current_query: str
    turn_start: int


def initial_state(
    user_query: str, *, deadline: float | None = None,
    run_id: str | None = None, run_root: str | None = None,
    language: Language | None = None,
) -> AgentState:
    """Start a task with its original user message."""

    return {
        "messages": [{"role": "user", "content": user_query}],
        "rounds": 0,
        "status": "running",
        "termination_reason": None,
        "deadline": deadline,
        "run_id": run_id,
        "run_root": run_root,
        "attachments": [],
        "draft": "",
        "language": language or preferred_language(user_query),
        "answer_active": False,
        "failure_explained": False,
        "current_query": user_query,
        "turn_start": 0,
    }


def append_message(state: AgentState, message: Message) -> AgentState:
    """Return a new state without altering earlier message records."""

    return {**state, "messages": [*state["messages"], message.copy()]}
