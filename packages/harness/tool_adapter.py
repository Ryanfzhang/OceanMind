"""Adapter between harness artifacts and the existing ToolOrchestrator."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from packages.harness.artifact_store import ArtifactRecord, ArtifactStore
from packages.harness.contracts import spec_from_xarray
from packages.harness.specs import ArtifactKind, ArtifactSpec, FrontendType
from packages.tool_loader.orchestrator import ToolOrchestrator


class ToolAdapter:
    def __init__(self, orchestrator: Optional[ToolOrchestrator] = None) -> None:
        self.orchestrator = orchestrator or ToolOrchestrator()

    def execute(
        self,
        tool_name: str,
        params: Mapping[str, Any],
        *,
        save_as: Optional[str] = None,
        store: Optional[ArtifactStore] = None,
        output_spec: Optional[ArtifactSpec] = None,
    ) -> ArtifactRecord:
        resolved_params = self._unwrap_params(dict(params))
        execution = self.orchestrator.execute_tool(
            tool_name=tool_name,
            params=resolved_params,
            save_as=save_as,
        )
        result_id = str(execution["result_id"])
        normalized_result = execution["result"]
        spec = output_spec or self._spec_from_normalized_result(
            artifact_id=result_id,
            normalized_result=normalized_result,
            tool_name=tool_name,
        )
        record = ArtifactRecord(
            artifact_id=result_id,
            spec=spec,
            value=normalized_result,
            summary=execution.get("summary") or {},
            provenance={"tool": tool_name, "ref_id": execution.get("ref_id")},
        )
        if store is not None:
            store.put(record)
        return record

    def _unwrap_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {key: self._unwrap_value(value) for key, value in params.items()}

    def _unwrap_value(self, value: Any) -> Any:
        if isinstance(value, ArtifactRecord):
            normalized = value.value
            if isinstance(normalized, dict) and "data" in normalized:
                return normalized["data"]
            return normalized
        if isinstance(value, dict):
            return {key: self._unwrap_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._unwrap_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._unwrap_value(item) for item in value)
        return value

    def _spec_from_normalized_result(
        self,
        *,
        artifact_id: str,
        normalized_result: Mapping[str, Any],
        tool_name: str,
    ) -> ArtifactSpec:
        output_type = str(normalized_result.get("output_type") or "generic_result")
        data = normalized_result.get("data")
        if data is not None and hasattr(data, "dims"):
            return spec_from_xarray(
                artifact_id,
                data,
                kind=ArtifactKind.FIELD,
                frontend_type=FrontendType(output_type)
                if output_type in FrontendType._value2member_map_
                else FrontendType.DATA_CONTAINER,
                provenance={"tool": tool_name},
            )
        return ArtifactSpec(
            artifact_id=artifact_id,
            kind=ArtifactKind.GENERIC,
            shape=spec_from_xarray(artifact_id, _ScalarLike()).shape,
            frontend_type=FrontendType(output_type)
            if output_type in FrontendType._value2member_map_
            else FrontendType.GENERIC,
            provenance={"tool": tool_name},
        )


class _ScalarLike:
    dims = ()
    coords: Dict[str, Any] = {}
    attrs: Dict[str, Any] = {}
    name = None

