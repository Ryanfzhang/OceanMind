"""The first single-agent LangGraph loop."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.errors import NodeError
from langgraph.types import Command, RetryPolicy

from packages.agent_loop.analysis import (
    LIST_RESULTS_SCHEMA,
    READ_ARTIFACT_SCHEMA,
    RUN_ANALYSIS_SCHEMA,
    WRITE_ANALYSIS_SCHEMA,
    AnalysisSession,
)
from packages.agent_loop.answer import ANSWER_PROMPT, REQUEST_VERIFICATION_SCHEMA
from packages.agent_loop.finalize import (
    REQUEST_CLARIFICATION_SCHEMA,
    clarification_request,
    finalize_delivery,
    finalize_state,
)
from packages.agent_loop.language import language_instruction, preferred_language
from packages.agent_loop.limits import budget_violation
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
    "You are OceanMind. For questions that do not need dataset analysis, use "
    "web_search when current information, source verification, or requested citations "
    "are needed. Stable knowledge can be answered directly. Use relevant retrieved "
    "URLs as citations when useful, but "
    "citations are optional; never invent them. If search fails, answer stable "
    "historical facts from knowledge without pretending sources were checked; "
    "do not infer current conditions. Answer conversational or creative "
    "requests directly. "
    "For ocean analysis, skills are optional method guidance: read one when useful, "
    "and use find_tools to inspect actual Python tool signatures. Treat skill examples "
    "as method guidance and call the actual functions through tools; do not copy "
    "legacy $ref or workflow wrappers as Python. "
    "A task without a matching skill can still use tools or ordinary Python. "
    "Workspace geometry is optional context: explicit coordinates in the current "
    "query take precedence. Selected polygon and transect vertices are [lon, lat]; "
    "selected_point has named lon and lat fields; a selected box supplies "
    "lon_range and lat_range. Preserve every transect vertex, "
    "and apply a polygon mask when analyzing the drawn polygon rather than "
    "treating only its bounding box as the selected area. "
    "If essential information is missing, call request_clarification with one question."
    " Weather and other current external facts require web_search; the configured "
    "ocean dataset is not evidence that live information is unavailable. For weather, "
    "ask for a city or location when none is specified. If a search fails, say the "
    "live information could not be verified; do not invent current conditions."
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
    answer_model: Any | None = None,
    max_rounds: int = 60,
    tool_registry: Mapping[str, Callable[..., Any]] | None = None,
    web_search: Callable[..., Any] | None = None,
    skills_root: Path | str | None = None,
    analysis_session: AnalysisSession | None = None,
):
    """Compile one model/tool loop; injected model and tools support offline tests."""
    if max_rounds < 1:
        raise ValueError("model decision limit must be positive")
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
            "Save ordinary Python cells with write_analysis, then run each code_id. "
            "The Python namespace persists across run_analysis calls within this query: "
            "variables from an earlier cell remain available. Each new cell should "
            "contain only the next analysis or a repair, never replay completed "
            "loading, calculation, or publishing stages. previous_version records "
            "code history but does not replay that version. If the environment is "
            "lost after a process crash, recover prior saved values with load_result "
            "using their artifact IDs. For cross-cell result provenance, pass those "
            "artifact IDs as input_refs or publish inputs. "
            "Inside the script, use `from oceanmind_runtime import tools, stage, publish, "
            "load_result`; wrap related work in stage blocks and keep stages outside loops. "
            "Keep each publish call inside the stage that produced its value; do not "
            "create a generic Run analysis stage for publishing. "
            "Use short, descriptive English stage titles for the English workspace UI; "
            "name actual analysis actions, not diagnostic probes. "
            "Use `publish(name, value, inputs=[])` for custom computed results; "
            "if the value derives from a tool result, pass its saved ID in inputs using "
            "`tools.ref(result)`. Tool calls already save their results; do not publish "
            "the same result again. Do not pass a `kind` argument. "
            "For a custom result derived from a selected transect or polygon without "
            "a geometry-aware tool input reference, pass "
            "`geometry={'type': 'transect' or 'polygon', 'points': vertices}` "
            "to publish using the vertices actually used in the calculation. "
            "For a raw field that you load and publish yourself, set "
            "`presentation=\"summary\"`; published calculated spatial fields use "
            "the default interactive map when they have lat/lon coordinates. "
            "When a map is requested, save the computed spatial field in its calculation "
            "stage; loading the source dataset is not the map result. "
            "After execution, check saved results "
            "and quality metrics; a zero exit code alone does not validate a calculation. "
            "Use list_results for earlier results and read_artifact for bounded details. "
            "When the requested result is saved and its quality is established, answer "
            "without rereading large arrays or running an unrelated extra analysis."
            " If a tool or cell fails, use its error observation to repair only the "
            "failed and dependent work; an error alone does not end the task. "
            "Once you have a defensible "
            "partial answer, keep a short current answer in the text content of later "
            "tool-calling messages and update it when evidence changes. If work must "
            "stop, this text can be delivered with the verified saved results."
            " Saved tool results automatically become interactive frontend views when "
            "their data shape is supported; do not build standalone HTML for them. "
            "For a T-S plot, pass the objects returned by tools.load_dataset directly "
            "as temp and salt to tools.compute_ts_diagram; never pass their `.data` "
            "arrays, which discard coordinates. Its saved result renders as an "
            "interactive T-S chart. "
            " To draw an interpretable eddy figure, import render_eddy_figure from "
            "packages.analysis_runtime.figures and publish its PngFigure. "
            "Call view_image with the published image artifact ID when visual patterns matter. "
            "Provide a concise evidence-based draft for the answer agent. "
            "Do not transcribe code or artifact IDs: verified saved files are attached "
            "automatically. Never rerun successful analysis just to repair answer text "
            "or attachment references. Verify numerical findings before making claims "
            "from an image."
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
        remaining = max_rounds - state["rounds"]
        delivery_hint = (
            f"\nYou have {remaining} model decisions left. Finish from verified results "
            "and state any missing work; do not start a new analysis."
            if remaining <= 3 else ""
        )
        model_messages = hydrate_vision_messages(
            [{"role": "system", "content": SYSTEM_PROMPT + skill_prompt + analysis_prompt
              + language_instruction(state["language"]) + delivery_hint},
             *state["messages"]], analysis_session,
        )
        answer = model.complete(
            model_messages,
            tools=[] if remaining <= (2 if answer_model is not None else 1) else schemas,
            timeout=timeout,
        )
        updated = append_message(state, answer)
        content = answer.get("content")
        return {**updated, "rounds": state["rounds"] + 1,
                "draft": content.strip() if isinstance(content, str) and content.strip()
                else state.get("draft", "")}

    def model_error(state: AgentState, error: NodeError) -> Command:
        return Command(
            update={"status": "incomplete",
                    "termination_reason": f"model_error: {type(error.error).__name__}"},
            goto="finalize",
        )

    def answer_error(state: AgentState, error: NodeError) -> Command:
        return Command(
            update={"status": "failed",
                    "termination_reason": f"answer_model_error: {type(error.error).__name__}"},
            goto="finalize",
        )

    def after_agent(state: AgentState) -> str:
        if state["status"] != "running":
            return "finalize"
        calls = state["messages"][-1].get("tool_calls") or []
        return "tools" if calls else "answer_agent" if answer_model is not None else "finalize"

    answer_registry = {"web_search": search}
    answer_schemas = [WEB_SEARCH_SCHEMA, REQUEST_VERIFICATION_SCHEMA]
    if analysis_session is not None:
        answer_registry.update({
            "list_results": analysis_session.list_results,
            "read_artifact": analysis_session.read_artifact,
            "view_image": make_view_image(analysis_session),
        })
        answer_schemas.extend([LIST_RESULTS_SCHEMA, READ_ARTIFACT_SCHEMA, VIEW_IMAGE_SCHEMA])

    def answer_agent(state: AgentState) -> AgentState:
        violation = budget_violation(state, max_rounds)
        if violation:
            return {**state, **violation}
        if not state["answer_active"] and analysis_session and analysis_session.on_event:
            analysis_session.on_event({"type": "synthesis_started"})
        remaining = max_rounds - state["rounds"]
        deadline = state["deadline"]
        prompt = ANSWER_PROMPT + language_instruction(state["language"])
        if remaining == 1:
            prompt += "\nThis is the final model decision. Deliver from verified evidence now."
        messages = hydrate_vision_messages(
            [{"role": "system", "content": prompt}, *state["messages"]],
            analysis_session,
        )
        reply = answer_model.complete(
            messages,
            tools=[] if remaining == 1 else answer_schemas,
            timeout=max(0.001, deadline - time.monotonic()) if deadline else None,
        )
        updated = append_message(state, reply)
        content = reply.get("content")
        return {**updated, "rounds": state["rounds"] + 1, "answer_active": True,
                "draft": content.strip() if isinstance(content, str) and content.strip()
                else state["draft"]}

    def after_answer(state: AgentState) -> str:
        if state["status"] != "running":
            return "finalize"
        return "answer_tools" if state["messages"][-1].get("tool_calls") else "finalize"

    def run_answer_tools(state: AgentState) -> Command:
        calls = state["messages"][-1]["tool_calls"]
        if len(calls) == 1 and calls[0]["function"]["name"] == "request_verification":
            observation = execute_tool_calls({"tool_calls": calls}, {
                "request_verification": lambda issue: {"verification_needed": issue},
            })[0]
            return Command(update={**append_message(state, observation),
                                   "answer_active": False}, goto="agent")
        updated = state
        for call in calls:
            available = answer_registry
            if call["function"]["name"] == "web_search" and state["deadline"]:
                remaining = max(0.001, state["deadline"] - time.monotonic())
                available = {**answer_registry,
                             "web_search": lambda **args: search(timeout=remaining, **args)}
            updated = append_message(updated, execute_tool_calls(
                {"tool_calls": [call]}, available,
            )[0])
        return Command(update=updated, goto="answer_agent")

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
        for call in calls:
            available = registry
            name = call["function"]["name"]
            if name == "web_search" and state["deadline"]:
                remaining = max(0.001, state["deadline"] - time.monotonic())
                available = {**registry, "web_search": lambda **args: search(timeout=remaining, **args)}
            if name == "run_analysis" and analysis_session is not None and state["deadline"]:
                remaining = max(0.001, state["deadline"] - time.monotonic())
                available = {**registry, "run_analysis": lambda code_id, timeout_seconds=1800: (
                    analysis_session.run_analysis(code_id, timeout_seconds=min(timeout_seconds, remaining))
                )}
            observation = execute_tool_calls({"tool_calls": [call]}, available)[0]
            updated = append_message(updated, observation)
        return updated

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent, retry_policy=RetryPolicy(max_attempts=3),
                   error_handler=model_error)
    graph.add_node("tools", run_tools)
    def finalize(state: AgentState) -> AgentState:
        classified = finalize_state(state)
        if classified["status"] == "completed":
            messages = classified["messages"]
            content = messages[-1].get("content") if messages else None
            if isinstance(content, str) and content.strip() and (
                preferred_language(content) != classified["language"]
            ):
                name = "Chinese" if classified["language"] == "zh" else "English"
                corrected = (answer_model or model).complete([
                    {"role": "system", "content": (
                        f"Translate the following answer into {name}. Return only the "
                        "translated answer. Preserve every number, unit, URL, artifact "
                        "ID, citation, and Markdown structure exactly; do not add facts."
                    )},
                    {"role": "user", "content": content},
                ], tools=[], timeout=None)
                translation = corrected.get("content")
                if (isinstance(translation, str) and translation.strip()
                        and preferred_language(translation) == classified["language"]):
                    messages = [*messages]
                    messages[-1] = {**messages[-1], "content": translation.strip()}
                    classified = {**classified, "messages": messages}
        return (finalize_delivery(classified, analysis_session)
                if analysis_session is not None else classified)

    graph.add_node("finalize", finalize)
    if answer_model is not None:
        graph.add_node("answer_agent", answer_agent,
                       retry_policy=RetryPolicy(max_attempts=3), error_handler=answer_error)
        graph.add_node("answer_tools", run_answer_tools)
        graph.add_conditional_edges("answer_agent", after_answer,
                                    {"answer_tools": "answer_tools", "finalize": "finalize"})
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, {
        "tools": "tools", "finalize": "finalize",
        **({"answer_agent": "answer_agent"} if answer_model is not None else {}),
    })
    graph.add_conditional_edges(
        "tools",
        lambda state: "agent" if state["status"] == "running" else "finalize",
        {"agent": "agent", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()
