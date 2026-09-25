"""Query endpoints for one resumable LangGraph agent loop."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from packages.agent_loop.conversations import ConversationRecord, ConversationStore
from packages.agent_loop.graph import build_graph
from packages.agent_loop.language import preferred_language
from packages.agent_loop.model import OpenAIChatModel
from packages.agent_loop.state import AgentState, initial_state
from packages.runtime.dataset_config import (
    get_active_dataset_config,
    get_active_dataset_public_config,
)
from packages.runtime.llm_config import load_agent_model_config


router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = None
    continue_pending: bool = False
    extracted_params: dict[str, Any] = Field(default_factory=dict)
    additional_context: dict[str, Any] = Field(default_factory=dict)
    synthesize: bool = True
    trust_env: bool = False


def _workspace_context(request: QueryRequest) -> dict[str, Any]:
    """Keep the selected geometry once while retaining other UI context."""
    params = dict(request.extracted_params)
    additional = dict(request.additional_context)
    geometry_aliases = {
        "mask_polygon": ("mask_polygon", "drawn_polygon_points"),
        "transect_points": ("transect_points", "drawn_transect_points"),
        "selected_point": ("selected_point",),
    }
    def covered(selection: Any) -> bool:
        if not isinstance(selection, dict):
            return False
        region = selection.get("selected_region")
        if isinstance(region, dict):
            if region.get("type") == "polygon" and "mask_polygon" not in params:
                return False
            if region.get("type") == "box" and "region" not in params:
                return False
            if region.get("type") not in {"polygon", "box"}:
                return False
        return (not selection.get("selected_transect") or "transect_points" in params) and (
            not selection.get("selected_point") or "selected_point" in params
        )

    if covered(params.get("workspace_selection")):
        params.pop("workspace_selection", None)
    for canonical, aliases in geometry_aliases.items():
        if canonical in params:
            for alias in aliases[1:]:
                params.pop(alias, None)
    workspace = additional.get("workspace_context")
    if isinstance(workspace, dict):
        workspace = dict(workspace)
        if covered(workspace.get("workspace_selection")):
            workspace.pop("workspace_selection", None)
        for canonical, aliases in geometry_aliases.items():
            if canonical in params:
                for alias in aliases:
                    workspace.pop(alias, None)
        additional["workspace_context"] = workspace
    return {key: value for key, value in {
        "extracted_params": params,
        "additional_context": additional,
    }.items() if value}


def _default_model() -> OpenAIChatModel:
    config = load_agent_model_config()
    return OpenAIChatModel(
        api_key=config.api_key, base_url=config.base_url, model=config.model,
    )


def _default_data_roots() -> tuple[Path, ...]:
    configured = get_active_dataset_config().data_path
    candidates = [configured]
    extra = os.getenv("OCEANMIND_DATA_ROOTS", "")
    candidates.extend(item for item in extra.split(os.pathsep) if item)
    roots: list[Path] = []
    for item in candidates:
        path = Path(item).expanduser()
        if path.exists():
            resolved = path.resolve(strict=True)
            if resolved not in roots:
                roots.append(resolved)
    return tuple(roots)


def _question(state: AgentState) -> str | None:
    if state["status"] != "needs_input":
        return None
    for message in reversed(state["messages"]):
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content") or "")
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("question"), str):
            return payload["question"]
    return None


def _summary(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            if message["content"].strip():
                return message["content"].strip()
    if state["status"] == "needs_input":
        return _question(state) or "More information is needed."
    return "The analysis ended without a complete answer."


def _skills_used(messages: list[dict[str, Any]]) -> list[str]:
    used: list[str] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != "read_skill":
                continue
            try:
                skill_id = json.loads(function.get("arguments") or "{}").get("skill_id")
            except (TypeError, ValueError):
                continue
            if isinstance(skill_id, str) and skill_id not in used:
                used.append(skill_id)
    return used


def _sources(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content") or "")
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            continue
        for item in payload["results"]:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not isinstance(url, str) or not url.startswith(("https://", "http://")) or url in seen:
                continue
            seen.add(url)
            cards.append({
                "title": str(item.get("title") or url),
                "source": str(item.get("source") or "Web"),
                "url": url,
                "short_snippet": str(item.get("snippet") or "")[:600],
                "why_it_matters": "Source consulted by the agent.",
                "provider": payload.get("provider"),
                "search_query": payload.get("query"),
                "rank": len(cards) + 1,
            })
    return cards


def _response(
    state: AgentState, record: ConversationRecord, query: str, turn_start: int,
) -> dict[str, Any]:
    success = state["status"] == "completed"
    needs_input = state["status"] == "needs_input"
    summary = _summary(state)
    turn_messages = state["messages"][turn_start:]
    analysis = any(
        call.get("function", {}).get("name") in {"write_analysis", "run_analysis"}
        for message in turn_messages for call in message.get("tool_calls") or []
    )
    return {
        "status": "completed" if success else "clarification_needed" if needs_input else "failed",
        "query": query,
        "language": state["language"],
        "conversation_id": record.conversation_id,
        "routing_mode": "dataset_analysis" if analysis else "general_answer",
        "router_confidence": None,
        "router_reason": None,
        "skill_id": None,
        "skills_used": _skills_used(turn_messages),
        "clarification_question": _question(state),
        "missing_fields": [],
        "analysis_proposal": None,
        "dataset_info": get_active_dataset_public_config(),
        "plan_summary": None,
        "plan_steps": [],
        "step_cards": [],
        "result_cards": [],
        "result_summaries": {},
        "synthesis": {"summary": summary} if success else None,
        "summary_status": "completed" if success else "failed",
        "source_cards": _sources(turn_messages),
        "active_result_id": None,
        "active_map_step_id": None,
        "workspace_data": {},
        "workspace_data_by_result": {},
        "attachments": state.get("attachments", []),
        "error": None if success or needs_input else state["termination_reason"] or summary,
        "failure_kind": None if success or needs_input else "execution",
        "recoverable": state["status"] == "incomplete",
    }


ProgressFactory = Callable[[Any, Callable[[dict[str, Any]], None]], Any]


class QueryService:
    def __init__(
        self,
        workspace: str | Path,
        *,
        model_factory: Callable[[], Any] = _default_model,
        data_roots: Callable[[], tuple[str | Path, ...]] = _default_data_roots,
        progress_factory: ProgressFactory | None = None,
        max_rounds: int = 60,
        timeout_seconds: float | None = None,
    ) -> None:
        self.store = ConversationStore(workspace)
        self.model_factory = model_factory
        self.data_roots = data_roots
        self.progress_factory = progress_factory
        self.max_rounds = max_rounds
        self.timeout_seconds = timeout_seconds

    def _record(self, request: QueryRequest, roots: tuple[str | Path, ...]):
        conversation_id = request.conversation_id
        if conversation_id and conversation_id.startswith("conv_"):
            try:
                record = self.store.resolve(conversation_id)
            except FileNotFoundError as exc:
                raise ValueError("Unknown conversation ID") from exc
            return record, self.store.open_analysis(record, data_roots=roots)
        if request.continue_pending:
            raise ValueError("Cannot continue an unknown conversation")
        return self.store.create(data_roots=roots)

    def execute(
        self, request: QueryRequest, *,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        emit = emit or (lambda _event: None)
        roots = self.data_roots()
        record, session = self._record(request, roots)
        with self.store.locked(record.conversation_id):
            # Re-read under the lock so simultaneous turns cannot lose history.
            record = self.store.resolve(record.conversation_id)
            session = self.store.open_analysis(record, data_roots=roots)
            adapter = self.progress_factory(session, emit) if self.progress_factory else None
            if adapter is not None:
                session.on_event = adapter.on_event
            else:
                session.on_event = lambda event: emit({"event": "execution_event", "payload": event})
            user_content = request.query.strip()
            if not user_content:
                raise ValueError("Query is empty")
            context = _workspace_context(request)
            if context:
                encoded = json.dumps(context, ensure_ascii=False, default=str)
                if len(encoded) > 20_000:
                    raise ValueError("Workspace context is too large")
                user_content += "\n\nWorkspace context supplied by the user:\n" + encoded
            state = initial_state(
                user_content,
                deadline=time.monotonic() + self.timeout_seconds if self.timeout_seconds else None,
                run_id=session.run_id,
                run_root=str(session.root),
                language=preferred_language(request.query),
            )
            turn_start = len(record.messages)
            state["messages"] = [*record.messages, *state["messages"]]
            emit({"event": "execution_event", "payload": {"type": "planning_started"}})
            graph = build_graph(
                self.model_factory(), max_rounds=self.max_rounds,
                analysis_session=session,
            )
            state = graph.invoke(
                state, config={"recursion_limit": 2 * self.max_rounds + 8},
            )
            record = replace(record, messages=state["messages"], status=state["status"])
            self.store.save(record)
            response = _response(state, record, request.query, turn_start)
            if adapter is not None:
                response.update(adapter.finalize(state))
            return response

    def stream(self, request: QueryRequest) -> Iterator[str]:
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def worker() -> None:
            try:
                response = self.execute(request, emit=events.put)
                events.put({"event": "final", "payload": response})
            except Exception as exc:
                events.put({"event": "error", "payload": {
                    "detail": str(exc), "failure_kind": "execution", "recoverable": True,
                }})
                events.put({"event": "final", "payload": {
                    "status": "failed", "query": request.query,
                    "conversation_id": None, "skills_used": [],
                    "missing_fields": [], "plan_steps": [], "step_cards": [],
                    "result_cards": [], "result_summaries": {}, "source_cards": [],
                    "error": str(exc), "failure_kind": "execution", "recoverable": True,
                }})
            finally:
                events.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            try:
                event = events.get(timeout=15)
            except queue.Empty:
                yield json.dumps({"event": "execution_event", "payload": {"type": "heartbeat"}}) + "\n"
                continue
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"


@lru_cache(maxsize=1)
def get_query_service() -> QueryService:
    from apps.api.langgraph_progress import ProgressAdapter

    workspace = Path(
        os.getenv("OCEANMIND_RUNS_DIR") or PROJECT_ROOT / "outputs" / "agent_runs"
    ).expanduser()
    return QueryService(workspace,
                        progress_factory=ProgressAdapter)


@router.post("/query")
def query(request: QueryRequest, service: QueryService = Depends(get_query_service)) -> dict[str, Any]:
    try:
        return service.execute(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query/stream")
def query_stream(
    request: QueryRequest, service: QueryService = Depends(get_query_service),
) -> StreamingResponse:
    return StreamingResponse(
        service.stream(request), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
