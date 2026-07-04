"""Client wiring helpers for the skill planner facade."""

from __future__ import annotations

from typing import Any, Dict, List


def get_client(planner: Any) -> Any:
    planner.client = planner._adapter.get_client()
    return planner.client


def create_message(
    planner: Any,
    client: Any,
    max_tokens: int,
    temperature: float,
    system: str,
    messages: List[Dict[str, Any]],
    request_name: str,
    json_response: bool = False,
) -> Any:
    return planner._adapter.create_message(
        client=client,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
        request_name=request_name,
        json_response=json_response,
    )
