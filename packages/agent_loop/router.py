"""Route a user turn by its information source before running the agent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

RouteMode = Literal["workspace_analysis", "web_information", "conversation", "clarification"]


class RouteDecision(TypedDict):
    mode: RouteMode
    search_query: str | None
    question: str | None


ROUTER_PROMPT = (
    "Route the latest user request. Return JSON only: "
    '{"mode":"workspace_analysis|web_information|conversation|clarification",'
    '"search_query":null,"question":null}. '
    "workspace_analysis = compute/plot/inspect the ocean workspace. "
    "web_information = requests needing fresh or verifiable outside sources, such as "
    "live conditions, weather, news, requested citations or recent literature. "
    "conversation = greetings, creative work or stable knowledge answerable without search. "
    "clarification = essential detail missing; briefly explain why it matters "
    "and ask one actionable question, suggesting an unconfirmed option if useful "
    "(weather needs a location; workspace map is not user location). Do not assume "
    "workspace geometry is absent merely because the query omits coordinates. "
    "If answering pending_question, "
    "combine pending_request and latest_request. For web_information set search_query "
    "to the complete subject/time/place, resolving relative dates using current_utc_time "
    "and the location; for clarification set question. Otherwise "
    "leave both null. Ocean data lacking a variable does not prevent web search."
)


def route_query(
    model: Any, query: str, *, pending_request: str | None = None,
    pending_question: str | None = None, timeout: float | None = None,
) -> RouteDecision | None:
    """Use a dedicated model decision; malformed output falls back to normal agent routing."""
    payload = {"latest_request": query,
               "current_utc_time": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    if pending_request and pending_question:
        payload.update({"pending_request": pending_request,
                        "pending_question": pending_question})
    messages = [
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    if callable(getattr(model, "complete_route", None)):
        reply = model.complete_route(messages, timeout=timeout)
    else:
        reply = model.complete(messages, tools=[], timeout=timeout)
    content = reply.get("content")
    if not isinstance(content, str):
        return None
    try:
        start = content.index("{")
        result, _ = json.JSONDecoder().raw_decode(content[start:])
    except (TypeError, ValueError):
        return None
    if not isinstance(result, dict):
        return None
    mode = result.get("mode")
    if mode not in {"workspace_analysis", "web_information", "conversation", "clarification"}:
        return None
    search_query = result.get("search_query")
    question = result.get("question")
    if mode == "web_information" and not (isinstance(search_query, str) and search_query.strip()):
        return None
    if mode == "clarification" and not (isinstance(question, str) and question.strip()):
        return None
    return {"mode": mode,
            "search_query": search_query.strip() if isinstance(search_query, str) else None,
            "question": question.strip() if isinstance(question, str) else None}
