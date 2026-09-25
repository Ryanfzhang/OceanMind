"""Script-level stages: one UI step may contain many tool calls."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime
from math import isfinite
from typing import Any, Callable, Iterator

from packages.analysis_runtime.records import new_id


class StageCountError(ValueError):
    """Declared work units do not match completed units."""


class StageLimitError(RuntimeError):
    """The script created too many UI stages."""


_CURRENT_STAGE: ContextVar[Stage | None] = ContextVar("analysis_stage", default=None)
_CURRENT_MANAGER: ContextVar[StageManager | None] = ContextVar("analysis_stage_manager", default=None)


def current_stage() -> Stage | None:
    """Return the explicit stage active in this context, if any."""
    return _CURRENT_STAGE.get()


def stage(title: str, *, total: int | None = None, unit: str | None = None):
    """Script-facing stage context; the runner must activate its manager first."""
    manager = _CURRENT_MANAGER.get()
    if manager is None:
        raise RuntimeError("stage(...) requires an active analysis script")
    return manager.stage(title, total=total, unit=unit)


def _scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("Stage coordinates must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return _scalar(value.item())
    raise TypeError("Stage coordinates must be scalar values, not arrays or objects")


@dataclass
class Stage:
    manager: StageManager
    id: str
    title: str
    total: int | None = None
    unit: str | None = None
    completed: int = 0
    current: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    results: list[dict[str, Any]] = field(default_factory=list)
    visible: bool = True

    def set_current(self, **coordinates: Any) -> None:
        self._require_running()
        self.current = {key: _scalar(value) for key, value in coordinates.items()}
        self.manager._emit_progress(self)

    def advance(self, units: int = 1) -> None:
        self._require_running()
        if isinstance(units, bool) or not isinstance(units, int) or units <= 0:
            raise StageCountError("advance() requires a positive integer")
        if self.total is not None and self.completed + units > self.total:
            raise StageCountError(f"Stage '{self.title}' would exceed total={self.total}")
        self.completed += units
        self.manager._emit_progress(self)

    def _require_running(self) -> None:
        if self.status != "running":
            raise RuntimeError(f"Stage '{self.title}' is already {self.status}")


class StageManager:
    """Owns stages for one script attempt and emits lightweight stage events."""

    def __init__(
        self,
        run_id: str,
        attempt_id: str,
        *,
        records: Any = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
        max_stages: int = 20,
        retain_events: bool = True,
    ) -> None:
        if isinstance(max_stages, bool) or not isinstance(max_stages, int) or max_stages < 1:
            raise ValueError("max_stages must be a positive integer")
        self.run_id, self.attempt_id = run_id, attempt_id
        self.records, self.callback, self.max_stages = records, callback, max_stages
        self.retain_events = retain_events
        self.stages: list[Stage] = []
        self.events: list[dict[str, Any]] = []
        self._default: Stage | None = None

    @contextmanager
    def activate(self) -> Iterator[StageManager]:
        """Bind this attempt to script-facing stage(...), closing its default stage."""
        token = _CURRENT_MANAGER.set(self)
        try:
            yield self
        except BaseException as exc:
            self.close(exc)
            raise
        else:
            self.close()
        finally:
            _CURRENT_MANAGER.reset(token)

    def _event(self, event: dict[str, Any]) -> None:
        event = {**event, "attempt_id": self.attempt_id}
        if self.retain_events:
            self.events.append(event)
        if self.callback:
            self.callback(event)

    def _new_stage(self, title: str, total: int | None, unit: str | None,
                   *, visible: bool = True) -> Stage:
        if not isinstance(title, str) or not title.strip() or len(title) > 120:
            raise ValueError("Stage title must contain 1..120 characters")
        if unit is not None and (not isinstance(unit, str) or len(unit) > 40):
            raise ValueError("Stage unit must contain at most 40 characters")
        if len(self.stages) >= self.max_stages:
            raise StageLimitError(
                f"At most {self.max_stages} stages per script; move stage(...) outside the loop"
            )
        if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < 0):
            raise ValueError("total must be a nonnegative integer")
        stage_id = (
            self.records.new_stage(self.attempt_id, title) if self.records
            else new_id("stage", run_id=self.run_id, attempt_id=self.attempt_id)
        )
        item = Stage(self, stage_id, title, total, unit, visible=visible)
        self.stages.append(item)
        if visible:
            self._event(self._snapshot("step_started", item))
        return item

    def _snapshot(self, kind: str, item: Stage) -> dict[str, Any]:
        event: dict[str, Any] = {
            "type": kind, "step_id": item.id, "stage_id": item.id,
            "title": item.title, "completed_units": item.completed,
            "visible_step": item.visible,
        }
        if item.total is not None:
            event["total_units"] = item.total
            if item.total > 0:
                event["percent"] = 100 * item.completed / item.total
            else:
                event["empty_input"] = True
        if item.unit:
            event["unit"] = item.unit
        if item.current:
            event["current_unit"] = dict(item.current)
        return event

    def _emit_progress(self, item: Stage) -> None:
        self._event(self._snapshot("step_progress", item))

    def _finish(self, item: Stage, status: str, error: str | None = None) -> None:
        if item.status != "running":
            return
        item.status = status
        if self.records:
            self.records.update("stage", item.id, status=status, completed_units=item.completed)
        event = self._snapshot("step_completed" if status == "completed" else "step_failed", item)
        if error:
            event["error"] = error
        if item.visible:
            self._event(event)

    @contextmanager
    def stage(self, title: str, *, total: int | None = None, unit: str | None = None) -> Iterator[Stage]:
        item = self._new_stage(title, total, unit)
        token = _CURRENT_STAGE.set(item)
        try:
            yield item
            if item.total is not None and item.completed != item.total:
                raise StageCountError(
                    f"Stage '{title}' completed {item.completed}/{item.total} units"
                )
        except BaseException as exc:
            self._finish(item, "failed", str(exc))
            raise
        else:
            self._finish(item, "completed")
        finally:
            _CURRENT_STAGE.reset(token)

    def ensure_stage(self, *, visible: bool = True) -> Stage:
        """Return the active stage, lazily creating one default script stage."""
        item = current_stage()
        if item is not None:
            if item.manager is not self:
                raise RuntimeError("Current stage belongs to another script attempt")
            return item
        if self._default is None:
            self._default = self._new_stage("Run analysis", None, None, visible=visible)
        elif visible and not self._default.visible:
            self._default.visible = True
            self._event(self._snapshot("step_started", self._default))
        return self._default

    def close(self, error: BaseException | None = None) -> None:
        """Close a lazily created default stage when the script finishes."""
        if self._default is not None:
            self._finish(self._default, "failed" if error else "completed", str(error) if error else None)

    def register_call_result(
        self,
        call_id: str,
        *,
        status: str,
        artifact_id: str | None = None,
        summary: str = "",
        inputs: list[str] | None = None,
    ) -> dict[str, Any]:
        """Append a call index entry; the actual result stays in the artifact store."""
        item = current_stage() or self._default or self.ensure_stage(visible=False)
        item._require_running()
        if status not in {"completed", "failed"}:
            raise ValueError("Call result status must be completed or failed")
        if not isinstance(summary, str) or len(summary) > 500:
            raise ValueError("Result summary must be text of at most 500 characters")
        if inputs is not None and not all(isinstance(ref, str) for ref in inputs):
            raise TypeError("Input references must be strings")
        entry = {
            "stage_id": item.id, "call_id": call_id, "artifact_id": artifact_id,
            "time": item.current.get("time"), "depth": item.current.get("depth"),
            "status": status, "summary": summary, "inputs": list(inputs or []),
        }
        item.results.append(entry)
        if self.records:
            self.records.update("stage", item.id, result_index=item.results)
        self._event({"type": "stage_result_indexed", "step_id": item.id,
                     "stage_id": item.id, "visible_step": item.visible,
                     "entry": entry.copy()})
        return entry

    @property
    def result_index(self) -> list[dict[str, Any]]:
        return [entry.copy() for item in self.stages for entry in item.results]
