"""Small direct entry point for the B-stage graph, before API integration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from packages.agent_loop.analysis import AnalysisSession
from packages.agent_loop.graph import build_graph
from packages.agent_loop.model import OpenAIChatModel
from packages.agent_loop.state import AgentState, initial_state
from packages.runtime.llm_config import load_agent_model_config


def run_query(
    user_query: str,
    *,
    model: Any = None,
    max_rounds: int = 60,
    timeout_seconds: float | None = None,
    analysis_workspace: str | Path | None = None,
    data_roots: tuple[str | Path, ...] = (),
) -> AgentState:
    """Run one query; a supplied model enables offline verification."""
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if model is None:
        config = load_agent_model_config()
        model = OpenAIChatModel(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
        )
    session = (AnalysisSession(analysis_workspace, data_roots=data_roots)
               if analysis_workspace is not None else None)
    graph = build_graph(model, max_rounds=max_rounds, analysis_session=session)
    state = initial_state(
        user_query, deadline=time.monotonic() + timeout_seconds if timeout_seconds else None,
        run_id=session.run_id if session else None,
        run_root=str(session.root) if session else None,
    )
    return graph.invoke(state, config={"recursion_limit": 2 * max_rounds + 8})
