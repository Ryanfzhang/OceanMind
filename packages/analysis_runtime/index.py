"""Bounded access to every saved result of one analysis attempt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .records import RunRecords


def list_results(
    root: str | Path, run_id: str, attempt_id: str, *, offset: int = 0,
    limit: int = 10,
) -> dict[str, Any]:
    """Page result references without loading their stored array or JSON payloads."""
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("offset must be nonnegative and limit must be 1..20")
    records = RunRecords(root, run_id=run_id)
    attempt = records.read("attempt", attempt_id)
    if attempt["run_id"] != run_id:
        raise ValueError("Attempt belongs to another run")
    directory = records.root / "records" / "stage"
    stages = sorted(
        (records.read("stage", path.stem)
         for path in directory.glob(f"{attempt_id}_stage_*.json")),
        key=lambda item: item["created_at"],
    ) if directory.exists() else []
    entries = [
        {"stage_id": stage["stage_id"], "stage": stage["name"],
         "call_id": entry["call_id"], "artifact_id": entry.get("artifact_id"),
         "status": entry["status"], "time": entry.get("time"),
         "depth": entry.get("depth"), "summary": entry.get("summary", "")[:300]}
        for stage in stages for entry in stage.get("result_index", [])
    ]
    return {
        "run_id": run_id, "attempt_id": attempt_id,
        "total": len(entries), "offset": offset, "limit": limit,
        "results": entries[offset:offset + limit],
        "next_offset": offset + limit if offset + limit < len(entries) else None,
    }
