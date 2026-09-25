"""Run one real model acceptance trace without logging credentials or large data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.langgraph_progress import ProgressAdapter
from apps.api.langgraph_query import QueryRequest, QueryService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--data-root", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--conversation")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1] / "outputs" / "agent_runs"
    events: list[dict] = []
    service = QueryService(
        workspace, data_roots=lambda: tuple(args.data_root),
        progress_factory=ProgressAdapter,
        timeout_seconds=args.timeout, max_rounds=args.rounds,
    )
    response = service.execute(QueryRequest(query=args.query,
                                           conversation_id=args.conversation,
                                           continue_pending=bool(args.conversation)), emit=events.append)
    record = service.store.resolve(response["conversation_id"])
    calls = [call.get("function", {}).get("name") for message in record.messages
             for call in message.get("tool_calls") or []]
    print(json.dumps({
        "status": response["status"],
        "conversation_id": response["conversation_id"],
        "run_id": record.run_id,
        "run_root": str(record.run_root),
        "tool_calls": calls,
        "events": [event.get("payload", {}).get("type") for event in events],
        "stages": [{"step_id": card["step_id"], "status": card["status"],
                    "progress": card.get("progress")} for card in response["step_cards"]],
        "attachments": response.get("attachments", []),
        "answer": (response.get("synthesis") or {}).get("summary", "")[:800],
        "error": response.get("error"),
    }, ensure_ascii=False, indent=2))
    return 0 if response["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
