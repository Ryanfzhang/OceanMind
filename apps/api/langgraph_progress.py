"""Project analysis runtime stages onto the existing workspace event protocol."""

from __future__ import annotations

import math
import time
from copy import deepcopy
from collections import OrderedDict
from typing import Any, Callable

from packages.agent_loop.analysis import AnalysisSession
from packages.analysis_runtime.artifacts import _stored_payload


PREVIEW_RESULTS = 2
PREVIEW_JSON_BYTES = 4 * 1024 * 1024


def _current_unit(value: Any) -> str | None:
    if isinstance(value, dict):
        return ", ".join(f"{key}: {item}" for key, item in value.items())[:120]
    return str(value)[:120] if value is not None else None


def _overlay(event: dict, artifact_id: str, number: int, when: Any, depth: Any) -> dict | None:
    center = event.get("center")
    if not isinstance(center, dict):
        return None
    lon, lat, radius = center.get("lon"), center.get("lat"), event.get("radius_km")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lon, lat, radius)):
        return None
    if not -180 <= lon <= 360 or not -90 <= lat <= 90 or radius <= 0:
        return None
    kind = str(event.get("type", "eddy"))
    details = [f"Radius: {radius:.1f} km"]
    if when is not None:
        details.append(f"Time: {when}")
    if depth is not None:
        details.append(f"Depth: {depth}")
    return {"id": f"{artifact_id}_{number}", "eventType": "eddy",
            "title": f"{kind.title()} eddy {number}", "center": {"lat": lat, "lon": lon},
            "shape": "circle", "radiusKm": radius, "details": details,
            "timestamp": str(when) if when is not None else None}


class ProgressAdapter:
    """Maintain bounded UI previews while the full runtime index stays on disk."""

    def __init__(self, session: AnalysisSession,
                 emit: Callable[[dict[str, Any]], None] | None = None,
                 *, interval_seconds: float = 0.25) -> None:
        self.session = session
        self.emit = emit or (lambda envelope: None)
        self.interval_seconds = interval_seconds
        self.cards: OrderedDict[str, dict] = OrderedDict()
        self.result_cards: list[dict] = []
        self.workspace_by_result: dict[str, dict] = {}
        self.counts: dict[str, dict[str, int]] = {}
        self.last_progress: dict[str, float] = {}
        self.active_result_id: str | None = None

    def _send(self, event_type: str, payload: dict[str, Any]) -> None:
        self.emit({"event": "execution_event", "payload": deepcopy({"type": event_type, **payload})})

    def _card(self, stage_id: str, title: str | None = None) -> dict:
        if stage_id not in self.cards:
            self.cards[stage_id] = {
                "step_id": stage_id, "human_label": title or "运行分析",
                "technical_label": "analysis stage", "status": "running",
                "results_hidden_by_default": False, "results": [], "actions": [],
                "is_map_bound": False, "is_expanded": False,
            }
            self.counts[stage_id] = {"completed": 0, "failed": 0}
        return self.cards[stage_id]

    def _attach(self, stage_id: str, entry: dict) -> None:
        card = self._card(stage_id)
        status = entry.get("status")
        if status in ("completed", "failed"):
            self.counts[stage_id][status] += 1
        artifact_id = entry.get("artifact_id")
        if status != "completed" or not isinstance(artifact_id, str):
            return
        try:
            metadata = self.session.artifacts.read_artifact(artifact_id)
        except (OSError, ValueError):
            return
        if (metadata.get("status") != "completed" or metadata.get("run_id") != self.session.run_id
                or metadata.get("stage_id") != stage_id):
            return
        is_eddy = metadata.get("name") == "detect_eddies"
        if len(card["results"]) >= PREVIEW_RESULTS:
            if not is_eddy:
                return
            evict = next((index for index, item in enumerate(card["results"])
                          if item.get("type") != "eddy_detection"), None)
            if evict is None:
                return
            old = card["results"].pop(evict)
            self.result_cards = [item for item in self.result_cards if item["id"] != old["id"]]
            self.workspace_by_result.pop(old["id"], None)
        when, depth = entry.get("time"), entry.get("depth")
        scope = ", ".join(f"{label}: {value}" for label, value in (("time", when), ("depth", depth))
                          if value is not None)
        name = str(metadata.get("name", "Result"))[:120]
        result = {"id": artifact_id, "title": name.replace("_", " ").title(),
                  "type": "eddy_detection" if is_eddy else str(metadata.get("kind", "result")),
                  "headline": scope or name,
                  "description": str(entry.get("summary", ""))[:300],
                  "renderer": "summary", "metrics": [], "surface": "inline",
                  "ownerStepId": stage_id}
        if is_eddy and metadata.get("kind") == "json":
            try:
                path = _stored_payload(self.session.root, metadata["payload"])
                if path.stat().st_size <= PREVIEW_JSON_BYTES:
                    value = self.session.artifacts.load_result(artifact_id)
                    events = value.get("events", []) if isinstance(value, dict) else []
                    overlays = [overlay for i, item in enumerate(events, 1)
                                if isinstance(item, dict)
                                if (overlay := _overlay(item, artifact_id, i, when, depth))]
                    result.update(renderer="event", surface="map",
                                  metrics=[{"label": "Accepted eddies", "value": str(len(overlays))}],
                                  workspaceData={"eventOverlays": overlays},
                                  actions=[{"id": "focus_map", "label": "Show on map"}])
                    self.workspace_by_result[artifact_id] = {"eventOverlays": overlays}
                    card["is_map_bound"] = True
                    self.active_result_id = artifact_id
            except (OSError, KeyError, TypeError, ValueError):
                pass
        card["results"].append(result)
        self.result_cards.append(result)
        self._send("step_result_attached", {"step_id": stage_id, "step_card": card.copy()})

    def on_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        stage_id = event.get("stage_id") or event.get("step_id")
        if not isinstance(stage_id, str):
            return
        if kind == "stage_result_indexed":
            if isinstance(event.get("entry"), dict):
                self._attach(stage_id, event["entry"])
            return
        if kind not in {"step_started", "step_progress", "step_completed", "step_failed"}:
            return
        card = self._card(stage_id, event.get("title"))
        if event.get("title"):
            card["human_label"] = str(event["title"])[:120]
        if kind == "step_progress":
            now = time.monotonic()
            if now - self.last_progress.get(stage_id, float("-inf")) < self.interval_seconds:
                return
            self.last_progress[stage_id] = now
        count = self.counts[stage_id]
        completed = event.get("completed_units")
        total = event.get("total_units")
        progress: dict[str, Any] = {"phase": "complete" if kind == "step_completed" else "computing",
                                    "message": f"{count['completed']} results saved"
                                    + (f", {count['failed']} failed" if count["failed"] else "")}
        if isinstance(completed, int):
            progress["completed_units"] = completed
        if isinstance(total, int):
            progress["total_units"] = total
        if isinstance(event.get("unit"), str):
            progress["unit_label"] = event["unit"]
        current = _current_unit(event.get("current_unit"))
        if current:
            progress["current_unit"] = current
        if isinstance(event.get("percent"), (int, float)):
            progress["percent"] = max(0.0, min(1.0, event["percent"] / 100))
        card["progress"] = progress
        if kind in ("step_completed", "step_failed"):
            card["status"] = "completed" if kind == "step_completed" else "failed"
            if kind == "step_failed":
                card["error"] = str(event.get("error", "Analysis stage failed"))[:500]
        self._send(kind, {"step_id": stage_id, "step_card": card.copy(),
                          "error": card.get("error"), "recoverable": False})

    def finalize(self, state: Any = None) -> dict[str, Any]:
        cards = list(self.cards.values())
        active = self.active_result_id
        return {
            "step_cards": cards, "result_cards": list(self.result_cards),
            "workspace_data_by_result": dict(self.workspace_by_result),
            "workspace_data": self.workspace_by_result.get(active, {}) if active else {},
            "active_result_id": active,
            "active_map_step_id": next((card["step_id"] for card in cards
                                        if any(result["id"] == active for result in card["results"])), None),
            "plan_steps": [{"step_id": card["step_id"], "tool": "run_analysis",
                            "human_label": card["human_label"],
                            "technical_label": card["technical_label"],
                            "status": card["status"]} for card in cards],
            "result_summaries": {card["step_id"]: {"completed": self.counts[card["step_id"]]["completed"],
                                                 "failed": self.counts[card["step_id"]]["failed"],
                                                 "preview_count": len(card["results"])} for card in cards},
        }
