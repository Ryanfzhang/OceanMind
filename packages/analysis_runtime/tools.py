"""Record ordinary Python tool calls without changing their return values."""

from __future__ import annotations

import json
import inspect
import math
from collections.abc import Callable, Mapping
from typing import Any

import xarray as xr

from packages.analysis_runtime.artifacts import ArtifactStore
from packages.analysis_runtime.records import RunRecords
from packages.analysis_runtime.stages import StageManager
from packages.tool_loader.progress import reset_tool_progress_callback, set_tool_progress_callback


def _brief(value: Any) -> Any:
    """Describe a parameter without recording its array values."""
    if isinstance(value, xr.DataArray):
        return {"type": "DataArray", "dims": list(value.dims), "shape": list(value.shape)}
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value[:200]
    if isinstance(value, (tuple, list)) and len(value) <= 10:
        return [_brief(item) for item in value]
    return {"type": type(value).__name__}


def _analysis_geometry(kind: str, points: Any) -> dict[str, Any]:
    """Keep the actual, bounded geometry used by an analysis result."""
    if kind not in {"transect", "polygon"} or not isinstance(points, (list, tuple)):
        raise ValueError("Analysis geometry requires transect or polygon vertices")
    if not (2 if kind == "transect" else 3) <= len(points) <= 1024:
        raise ValueError("Analysis geometry has an invalid vertex count")
    vertices = []
    for point in points:
        if isinstance(point, dict):
            lon, lat = point.get("lon"), point.get("lat")
        elif isinstance(point, (list, tuple)) and len(point) == 2:
            lon, lat = point
        else:
            raise ValueError("Analysis vertices must be [lon, lat] pairs")
        if (not isinstance(lon, (int, float)) or not isinstance(lat, (int, float))
                or not math.isfinite(lon) or not math.isfinite(lat)
                or not -180 <= lon <= 360 or not -90 <= lat <= 90):
            raise ValueError("Analysis vertex is outside longitude/latitude bounds")
        vertices.append([float(lon), float(lat)])
    return {"type": kind, "points": vertices}


class AnalysisTools:
    """Execute discovered tools, save each result, and emit call-level events."""

    def __init__(
        self,
        records: RunRecords,
        artifacts: ArtifactStore,
        stages: StageManager,
        *,
        functions: Mapping[str, Callable[..., Any]] | None = None,
    ) -> None:
        if records.run_id != stages.run_id:
            raise ValueError("Records and stages belong to different runs")
        self.records, self.artifacts, self.stages = records, artifacts, stages
        if functions is None:
            from packages.tool_loader.introspect import get_tools_cached

            functions = get_tools_cached()
        self.functions = dict(functions)
        self._result_refs: dict[int, str] = {}

    def _function(self, name: str) -> Callable[..., Any]:
        if name in self.functions:
            return self.functions[name]
        matches = [func for key, func in self.functions.items() if key.endswith(f".{name}")]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"Unknown or ambiguous tool: {name}")

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self.call(name, *args, **kwargs)

    def ref(self, value: Any) -> str:
        """Get the artifact ID for an unchanged result object."""
        try:
            return self._result_refs[id(value)]
        except KeyError as exc:
            raise ValueError("Result has no artifact ID; pass an explicit input reference") from exc

    def call(
        self, tool_name: str, *args: Any, input_refs: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        func = self._function(tool_name)
        bound = inspect.signature(func).bind(*args, **kwargs)
        invoke = lambda: func(*args, **kwargs)
        result, _ = self._execute(tool_name, invoke, input_refs, dict(bound.arguments),
                                  source_result=True)
        return result

    def _execute(
        self, name: str, func: Callable[..., Any], input_refs: list[str] | None,
        kwargs: dict[str, Any],
        *, source_result: bool = False, presentation: str = "auto",
        geometry: dict[str, Any] | None = None,
    ) -> tuple[Any, str]:
        stage = self.stages.ensure_stage(visible=False)
        refs = input_refs or [
            self._result_refs[id(value)] for value in kwargs.values()
            if id(value) in self._result_refs
        ]
        parameters = {key: _brief(value) for key, value in kwargs.items()}
        if geometry is not None:
            parameters["analysis_geometry"] = _analysis_geometry(
                geometry.get("type"), geometry.get("points"),
            )
        else:
            for argument, kind in (("transect_points", "transect"),
                                   ("polygon_points", "polygon")):
                if isinstance(kwargs.get(argument), (list, tuple)):
                    try:
                        parameters["analysis_geometry"] = _analysis_geometry(kind, kwargs[argument])
                    except ValueError:
                        pass  # Geometry recording must not reject a valid tool call.
                    break
        call_id = self.records.new_call(
            self.stages.attempt_id, stage.id, name,
            inputs=refs, parameters=parameters,
        )
        self.stages._event({"type": "call_started", "stage_id": stage.id, "call_id": call_id,
                            "name": name})

        def progress(payload: dict[str, Any]) -> None:
            self.stages._event({"type": "call_progress", "stage_id": stage.id,
                                "call_id": call_id, "progress": payload})

        token = set_tool_progress_callback(progress)
        try:
            result = func()
            if (presentation == "auto" and isinstance(result, xr.DataArray)
                    and not refs and source_result
                    and not any(isinstance(value, xr.DataArray) for value in kwargs.values())):
                presentation = "summary"
            artifact_id = self.artifacts.publish(
                name, result, run_id=self.records.run_id, attempt_id=self.stages.attempt_id,
                stage_id=stage.id, inputs=refs, call_id=call_id,
                presentation=presentation,
            )
            summary = json.dumps(
                self.artifacts.read_artifact(artifact_id)["summary"], ensure_ascii=False
            )[:500]
            self.records.update("call", call_id, status="completed", artifact_ids=[artifact_id])
            self.stages.register_call_result(
                call_id, status="completed", artifact_id=artifact_id,
                summary=summary, inputs=refs,
            )
            self._result_refs[id(result)] = artifact_id
            self.stages._event({"type": "call_completed", "stage_id": stage.id,
                                "call_id": call_id, "artifact_id": artifact_id})
            return result, artifact_id
        except Exception as exc:
            self.records.update("call", call_id, status="failed", error=str(exc)[:500])
            self.stages.register_call_result(
                call_id, status="failed", summary=str(exc)[:500], inputs=refs,
            )
            self.stages._event({"type": "call_failed", "stage_id": stage.id,
                                "call_id": call_id, "error": str(exc)[:500]})
            raise
        finally:
            reset_tool_progress_callback(token)

    def publish(self, name: str, value: Any, *, inputs: list[str] | None = None,
                presentation: str = "auto", geometry: dict[str, Any] | None = None) -> str:
        """Save one custom calculation in the active stage."""
        _, artifact_id = self._execute(
            f"publish:{name}", lambda: value, [] if inputs is None else inputs, {},
            presentation=presentation, geometry=geometry,
        )
        return artifact_id

    def read_artifact(self, artifact_id: str, **limits: Any) -> dict:
        metadata = self.artifacts.read_artifact(artifact_id, **limits)
        if metadata["status"] != "missing" and metadata.get("run_id") != self.records.run_id:
            raise ValueError("Artifact belongs to another run")
        return metadata

    def load_result(self, artifact_id: str) -> Any:
        metadata = self.artifacts.read_artifact(artifact_id)
        if metadata["status"] != "missing" and metadata.get("run_id") != self.records.run_id:
            raise ValueError("Artifact belongs to another run")
        return self.artifacts.load_result(artifact_id)
