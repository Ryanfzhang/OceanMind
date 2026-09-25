"""The first single-agent LangGraph loop."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from packages.agent_loop.analysis import (
    LIST_RESULTS_SCHEMA,
    READ_ARTIFACT_SCHEMA,
    RUN_ANALYSIS_SCHEMA,
    WRITE_ANALYSIS_SCHEMA,
    AnalysisSession,
)
from packages.agent_loop.finalize import (
    REQUEST_CLARIFICATION_SCHEMA,
    clarification_request,
    finalize_delivery,
    finalize_state,
)
from packages.agent_loop.limits import budget_violation, repeated_failure_count
from packages.agent_loop.ocean_tools import (
    FIND_TOOLS_SCHEMA,
    INSPECT_DATA_SCHEMA,
    find_tools,
    make_inspect_data,
)
from packages.agent_loop.search import WEB_SEARCH_SCHEMA, make_web_search_tool
from packages.agent_loop.skills import READ_SKILL_SCHEMA, list_skills, read_skill
from packages.agent_loop.state import AgentState, append_message
from packages.agent_loop.tools import TOOL_REGISTRY, TOOL_SCHEMAS, execute_tool_calls
from packages.agent_loop.vision import (
    VIEW_IMAGE_SCHEMA,
    hydrate_vision_messages,
    make_view_image,
)
from packages.runtime.dataset_config import get_active_dataset_config


SYSTEM_PROMPT = (
    "You are OceanMind. Answer directly when no external information is needed. "
    "Use web_search for current facts and cite the returned source URLs. "
    "For ocean analysis, skills are optional method guidance: read one when useful, "
    "and use find_tools to inspect actual Python tool signatures. Skills may contain "
    "legacy DSL examples; do not copy their .data, $ref, or workflow wrappers as Python. "
    "A task without a matching skill can still use tools or ordinary Python. "
    "Workspace geometry is optional context: explicit coordinates in the current "
    "query take precedence. Selected polygon and transect vertices are [lon, lat]; "
    "selected_point has named lon and lat fields; a selected box supplies "
    "lon_range and lat_range. Preserve every transect vertex, "
    "and apply a polygon mask when analyzing the drawn polygon rather than "
    "treating only its bounding box as the selected area. "
    "If essential information is missing, call request_clarification with one question."
)


def _configured_data_prompt(data_roots: tuple[Path, ...]) -> str:
    config = get_active_dataset_config()
    if Path(config.data_path).resolve() not in data_roots:
        return ""
    return (
        f"\nConfigured dataset: {config.name} ({config.backend}); "
        f"variables: {', '.join(config.variables)}; "
        f"time: {config.temporal_extent}; region: {config.spatial_extent}; "
        f"depth levels: {len(config.depth_levels)} from "
        f"{config.depth_levels[0] if config.depth_levels else 'unknown'} to "
        f"{config.depth_levels[-1] if config.depth_levels else 'unknown'} m. "
        "tools.load_dataset uses this configured source by default. "
        "Use these known facts directly; inspect_data is for an unfamiliar source or "
        "variable, not repeated discovery of this configured dataset."
    )


def build_graph(
    model: Any,
    *,
    max_rounds: int = 8,
    repeated_failure_limit: int = 2,
    tool_registry: Mapping[str, Callable[..., Any]] | None = None,
    web_search: Callable[..., Any] | None = None,
    skills_root: Path | str | None = None,
    analysis_session: AnalysisSession | None = None,
):
    """Compile one model/tool loop; injected model and tools support offline tests."""
    if max_rounds < 1 or repeated_failure_limit < 1:
        raise ValueError("loop limits must be positive")
    search = web_search or make_web_search_tool()
    skill_index = list_skills(skills_root)
    skill_prompt = "\nAvailable skill guidance (ID: summary):\n" + "\n".join(
        f"- {item['skill_id']}: {item['description']}" for item in skill_index
    )
    registry = {
        **TOOL_REGISTRY,
        **(tool_registry or {}),
        "web_search": search,
        "read_skill": lambda skill_id: {
            "skill_id": skill_id,
            "content": read_skill(skill_id, root=skills_root),
        },
        "find_tools": find_tools,
    }
    schemas = [
        *TOOL_SCHEMAS, WEB_SEARCH_SCHEMA, READ_SKILL_SCHEMA, FIND_TOOLS_SCHEMA,
        REQUEST_CLARIFICATION_SCHEMA,
    ]
    analysis_prompt = ""
    if analysis_session is not None:
        registry.update({
            "write_analysis": analysis_session.write_analysis,
            "read_artifact": analysis_session.read_artifact,
            "run_analysis": analysis_session.run_analysis,
            "list_results": analysis_session.list_results,
            "view_image": make_view_image(analysis_session),
        })
        schemas.extend([
            WRITE_ANALYSIS_SCHEMA, READ_ARTIFACT_SCHEMA,
            RUN_ANALYSIS_SCHEMA, LIST_RESULTS_SCHEMA, VIEW_IMAGE_SCHEMA,
        ])
        if analysis_session.data_roots:
            registry["inspect_data"] = make_inspect_data(analysis_session.data_roots)
            schemas.append(INSPECT_DATA_SCHEMA)
        analysis_prompt = (
            "\nFor data work, use the configured dataset directly when it covers the query. "
            "Use find_tools for an unfamiliar function signature; do not write and run "
            "probe scripts to inspect runtime internals, artifact storage, or installed "
            "plotting libraries. Write the requested analysis first. "
            "Save complete ordinary Python with write_analysis, then run its code_id. "
            "Inside the script, use `from oceanmind_runtime import tools, stage, publish, "
            "load_result`; wrap related work in stage blocks and keep stages outside loops. "
            "Use `publish(name, value, inputs=[])` for custom computed results; "
            "if the value derives from a tool result, pass its saved ID in inputs using "
            "`tools.ref(result)`. Tool calls already save their results; do not publish "
            "the same result again. Do not pass a `kind` argument. "
            "After execution, check saved results "
            "and quality metrics; a zero exit code alone does not validate a calculation. "
            "Use list_results for earlier results and read_artifact for bounded details."
            " Saved tool results automatically become interactive frontend views when "
            "their data shape is supported; do not build standalone HTML for them. "
            "For a T-S plot, pass the objects returned by tools.load_dataset directly "
            "as temp and salt to tools.compute_ts_diagram; never pass their `.data` "
            "arrays, which discard coordinates. Its saved result renders as an "
            "interactive T-S chart. "
            " To draw an interpretable eddy figure, import render_eddy_figure from "
            "packages.analysis_runtime.figures and publish its PngFigure. "
            "Call view_image with the published image artifact ID when visual patterns matter. "
            "In the final answer use exact saved code and artifact IDs for any attachments; "
            "verify numerical findings before making claims from an image."
            "\nAuthorized data roots: "
            + (", ".join(str(path) for path in analysis_session.data_roots)
               if analysis_session.data_roots else "none")
            + _configured_data_prompt(analysis_session.data_roots)
        )

    def agent(state: AgentState) -> AgentState:
        violation = budget_violation(state, max_rounds)
        if violation:
            return {**state, **violation}
        deadline = state["deadline"]
        timeout = max(0.001, deadline - time.monotonic()) if deadline else None
        try:
            model_messages = hydrate_vision_messages(
                [{"role": "system", "content": SYSTEM_PROMPT + skill_prompt + analysis_prompt},
                 *state["messages"]], analysis_session,
            )
            answer = model.complete(
                model_messages,
                tools=schemas,
                timeout=timeout,
            )
        except Exception:
            return {**state, "status": "failed", "termination_reason": "model_error"}
        updated = append_message(state, answer)
        return {**updated, "rounds": state["rounds"] + 1}

    def after_agent(state: AgentState) -> str:
        if state["status"] != "running":
            return "finalize"
        calls = state["messages"][-1].get("tool_calls") or []
        return "tools" if calls else "finalize"

    def run_tools(state: AgentState) -> AgentState:
        updated = state
        assistant = state["messages"][-1]
        calls = assistant["tool_calls"]
        def close_unexecuted(current: AgentState, remaining: list[dict], reason: str) -> AgentState:
            for pending in remaining:
                current = append_message(current, {
                    "role": "tool", "tool_call_id": pending["id"],
                    "content": json.dumps({"status": "not_executed", "reason": reason}),
                })
            return {**current, "status": "incomplete", "termination_reason": reason}

        if state["deadline"] is not None and time.monotonic() >= state["deadline"]:
            return close_unexecuted(updated, calls, "deadline_exceeded")
        try:
            clarification = clarification_request(assistant)
        except ValueError:
            return {**state, "status": "failed", "termination_reason": "invalid_clarification_request"}
        if clarification:
            call_id, question = clarification
            observation = {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps({"status": "needs_input", "question": question}),
            }
            return {
                **append_message(state, observation),
                "status": "needs_input",
                "termination_reason": "clarification_requested",
            }
        for index, call in enumerate(calls):
            violation = budget_violation(updated, max_rounds)
            if violation:
                return close_unexecuted(updated, calls[index:], violation["termination_reason"])
            available = registry
            name = call["function"]["name"]
            if name == "web_search" and state["deadline"]:
                remaining = max(0.001, state["deadline"] - time.monotonic())
                available = {**registry, "web_search": lambda **args: search(timeout=remaining, **args)}
            if name == "run_analysis" and analysis_session is not None and state["deadline"]:
                remaining = max(0.001, state["deadline"] - time.monotonic())
                available = {**registry, "run_analysis": lambda code_id, timeout_seconds=60: (
                    analysis_session.run_analysis(code_id, timeout_seconds=min(timeout_seconds, remaining))
                )}
            observation = execute_tool_calls({"tool_calls": [call]}, available)[0]
            updated = append_message(updated, observation)
            if repeated_failure_count(updated["messages"]) >= repeated_failure_limit:
                return close_unexecuted(updated, calls[index + 1:], "repeated_tool_failure")
        return updated

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("finalize", (lambda state: finalize_delivery(state, analysis_session))
                   if analysis_session is not None else finalize_state)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", "finalize": "finalize"})
    graph.add_conditional_edges(
        "tools",
        lambda state: "agent" if state["status"] == "running" else "finalize",
        {"agent": "agent", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "finalize",
        lambda state: "agent" if state["status"] == "running" else "end",
        {"agent": "agent", "end": END},
    )
    return graph.compile()
