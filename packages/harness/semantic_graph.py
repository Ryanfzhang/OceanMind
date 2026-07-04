"""Semantic artifact graph for OceanMind harness plans.

This layer is intentionally above executable steps.  It records what data
artifacts are required, what intermediate artifacts are produced, their shapes,
and which implementation strategy was chosen for each task.  The legacy
``steps`` list is then just a compatibility projection for the current executor
and frontend progress protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from packages.harness.ir import ExecutionStrategy, TaskGraph, TaskNode
from packages.harness.shapes import ShapeClass
from packages.harness.specs import ArtifactKind, ArtifactSpec, ReadSpec, VerticalSpec


@dataclass(frozen=True)
class DataRequirementContract:
    requirement_id: str
    artifact_id: str
    variables: Tuple[str, ...]
    dataset: str
    region: Mapping[str, Any]
    time_range: Optional[Tuple[str, str]]
    vertical: Mapping[str, Any]
    expected_shape: str
    mask_requirements: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MaskRequirementContract:
    mask_id: str
    artifact_id: str
    dims: Tuple[str, ...]
    shape_class: str
    role: str
    applies_to: Tuple[str, ...] = ()
    broadcast_policy: str = "coordinate_aligned"


@dataclass(frozen=True)
class ArtifactContract:
    artifact_id: str
    kind: str
    shape_class: str
    dims: Tuple[str, ...]
    role: str
    frontend_type: str
    variable: Optional[str] = None
    units: str = ""
    produced_by: Optional[str] = None
    masks: Tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticTask:
    task_id: str
    task_type: str
    intent: str
    operation: str
    inputs: Mapping[str, str]
    outputs: Tuple[str, ...]
    implementation: Mapping[str, Any]
    input_contract: Mapping[str, Any] = field(default_factory=dict)
    output_contract: Mapping[str, Any] = field(default_factory=dict)
    validation_rules: Tuple[str, ...] = ()
    origin: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticTaskGraph:
    graph_id: str
    data_requirements: Tuple[DataRequirementContract, ...]
    mask_requirements: Tuple[MaskRequirementContract, ...]
    artifacts: Tuple[ArtifactContract, ...]
    tasks: Tuple[SemanticTask, ...]
    final_artifacts: Tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def semantic_graph_from_task_graph(
    graph: TaskGraph,
    *,
    user_request: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> SemanticTaskGraph:
    """Build a semantic artifact graph from the executable harness IR."""

    artifacts: Dict[str, ArtifactContract] = {}
    data_requirements = []
    mask_requirements = []
    tasks = []

    for node in graph.nodes:
        if node.output is not None:
            artifact = _artifact_contract_from_spec(node.output, produced_by=node.node_id)
            artifacts[artifact.artifact_id] = artifact
            for auxiliary_artifact in _vertical_auxiliary_artifacts(node):
                artifacts[auxiliary_artifact.artifact_id] = auxiliary_artifact
            if node.output.kind == ArtifactKind.MASK:
                mask_requirements.append(_mask_requirement_from_artifact(node.output))
            mask_requirements.extend(_vertical_auxiliary_masks(node))

        if node.read_spec is not None and node.output is not None:
            data_requirements.append(
                _data_requirement_from_read_spec(
                    node.read_spec,
                    requirement_id=f"read:{node.output.artifact_id}",
                    artifact_id=node.output.artifact_id,
                )
            )

        for input_artifact in node.inputs.values():
            if input_artifact not in artifacts:
                artifacts[input_artifact] = _placeholder_artifact_contract(input_artifact)

        tasks.append(_semantic_task_from_node(node, graph_metadata=graph.metadata))

    graph_metadata: Dict[str, Any] = {
        "planning_model": "artifact_first",
        "user_request": user_request,
        "steps_are_projection": True,
        "step_projection_target": "legacy_executor_frontend_protocol",
        "manual_usage": {
            "selected_manual": graph.metadata.get("manual_id"),
            "selected_recipe": graph.metadata.get("recipe_id"),
            "matched_manuals": list(graph.metadata.get("matched_manuals") or []),
        },
        "mask_policy": {
            "depth_selection": "vertical_selection_not_roi_mask",
            "local_bottom": "bottom/bottom_band use per-cell deepest valid wet level, not one global depth",
            "roi_masks": "spatial/event/wet/threshold masks only",
            "broadcast": "by named coordinates and compatible shape class",
        },
    }
    graph_metadata.update(dict(graph.metadata))
    if metadata:
        graph_metadata.update(dict(metadata))

    return SemanticTaskGraph(
        graph_id=f"{graph.graph_id}:semantic",
        data_requirements=tuple(data_requirements),
        mask_requirements=tuple(mask_requirements),
        artifacts=tuple(artifacts.values()),
        tasks=tuple(tasks),
        final_artifacts=tuple(graph.final_artifacts),
        metadata=graph_metadata,
    )


def semantic_graph_to_dict(graph: SemanticTaskGraph) -> Dict[str, Any]:
    return {
        "graph_id": graph.graph_id,
        "metadata": _jsonify(graph.metadata),
        "data_requirements": [_data_requirement_to_dict(item) for item in graph.data_requirements],
        "mask_requirements": [_mask_requirement_to_dict(item) for item in graph.mask_requirements],
        "artifacts": [_artifact_contract_to_dict(item) for item in graph.artifacts],
        "tasks": [_semantic_task_to_dict(item) for item in graph.tasks],
        "final_artifacts": list(graph.final_artifacts),
    }


def _semantic_task_from_node(node: TaskNode, *, graph_metadata: Mapping[str, Any]) -> SemanticTask:
    execution = node.execution
    strategy = execution.strategy.value if execution is not None else ExecutionStrategy.HARNESS.value
    implementation: Dict[str, Any] = {
        "strategy": strategy,
        "operation": node.operation,
    }
    if execution is not None:
        implementation["tool_name"] = execution.tool_name
        implementation["params"] = _jsonify(execution.params)
        if execution.strategy == ExecutionStrategy.CODE:
            implementation["code_contract"] = {
                "entrypoint": "run(inputs, params) -> dict",
                "allowed_libraries": ["numpy", "pandas", "xarray", "scipy"],
                "blocked_capabilities": ["file", "network", "shell"],
                "io_contract": _jsonify(execution.params.get("io_contract", {})),
            }

    output_id = node.output.artifact_id if node.output is not None else node.node_id
    output_contract = _artifact_contract_to_dict(
        _artifact_contract_from_spec(node.output, produced_by=node.node_id)
    ) if node.output is not None else {}

    origin: Dict[str, Any] = {}
    for key in ("manual_id", "recipe_id", "event_type"):
        if graph_metadata.get(key) is not None:
            origin[key] = graph_metadata.get(key)
    if node.read_spec is not None:
        origin["data_requirement"] = f"read:{output_id}"

    return SemanticTask(
        task_id=node.node_id,
        task_type=node.node_type.value,
        intent=node.intent,
        operation=node.operation,
        inputs=dict(node.inputs),
        outputs=(output_id,),
        implementation=implementation,
        input_contract={
            key: {"artifact_id": value, "required": True}
            for key, value in node.inputs.items()
        },
        output_contract=output_contract,
        validation_rules=tuple(node.validation_rules),
        origin=origin,
    )


def _data_requirement_from_read_spec(
    read_spec: ReadSpec,
    *,
    requirement_id: str,
    artifact_id: str,
) -> DataRequirementContract:
    expected_shape = (
        read_spec.expected_shape.value
        if isinstance(read_spec.expected_shape, ShapeClass)
        else str(read_spec.expected_shape or ShapeClass.UNKNOWN.value)
    )
    return DataRequirementContract(
        requirement_id=requirement_id,
        artifact_id=artifact_id,
        variables=tuple(read_spec.variables),
        dataset=read_spec.dataset,
        region=_jsonify(read_spec.region or {}),
        time_range=tuple(read_spec.time_range) if read_spec.time_range is not None else None,
        vertical=_vertical_to_dict(read_spec.vertical),
        expected_shape=expected_shape,
        mask_requirements=tuple(read_spec.mask_requirements),
    )


def _artifact_contract_from_spec(
    spec: Optional[ArtifactSpec],
    *,
    produced_by: Optional[str],
) -> ArtifactContract:
    if spec is None:
        return _placeholder_artifact_contract(str(produced_by or "unknown"))
    return ArtifactContract(
        artifact_id=spec.artifact_id,
        kind=spec.kind.value,
        shape_class=spec.shape_class.value,
        dims=tuple(spec.dims),
        role=_artifact_role(spec),
        frontend_type=spec.frontend_type.value,
        variable=spec.variable,
        units=spec.units,
        produced_by=produced_by,
        masks=tuple(_mask_label(index, mask) for index, mask in enumerate(spec.masks)),
        provenance=_jsonify(spec.provenance),
    )


def _placeholder_artifact_contract(artifact_id: str) -> ArtifactContract:
    return ArtifactContract(
        artifact_id=artifact_id,
        kind=ArtifactKind.GENERIC.value,
        shape_class=ShapeClass.UNKNOWN.value,
        dims=(),
        role="external_or_prior_artifact",
        frontend_type="generic_result",
    )


def _mask_requirement_from_artifact(spec: ArtifactSpec) -> MaskRequirementContract:
    return MaskRequirementContract(
        mask_id=f"mask:{spec.artifact_id}",
        artifact_id=spec.artifact_id,
        dims=tuple(spec.dims),
        shape_class=spec.shape_class.value,
        role="analysis_mask",
    )


def _vertical_auxiliary_artifacts(node: TaskNode) -> Tuple[ArtifactContract, ...]:
    if node.operation != "select_vertical" or node.output is None or node.execution is None:
        return ()
    params = node.execution.params
    mode = str(params.get("mode") or "").strip().lower()
    relative_to = str(params.get("relative_to") or "").strip().lower()
    if mode not in {"bottom", "bottom_band", "relative_to_bottom"} and relative_to != "bottom":
        return ()

    output_id = node.output.artifact_id
    auxiliaries = [
        ArtifactContract(
            artifact_id=f"{output_id}_bottom_depth",
            kind=ArtifactKind.MAP.value,
            shape_class=ShapeClass.MAP_2D.value,
            dims=("lat", "lon"),
            role="auxiliary_vertical_metadata",
            frontend_type="data_container_result",
            produced_by=node.node_id,
            provenance={
                "vertical_mode": mode or "bottom",
                "meaning": "Per-cell deepest valid wet depth used for local-bottom selection.",
                "time_dependent_allowed": True,
                "possible_dims": [["lat", "lon"], ["time", "lat", "lon"]],
            },
        )
    ]
    if mode in {"bottom_band", "relative_to_bottom"} or relative_to == "bottom":
        auxiliaries.append(
            ArtifactContract(
                artifact_id=f"{output_id}_valid_bottom_band_mask",
                kind=ArtifactKind.MASK.value,
                shape_class=ShapeClass.FIELD_3D_DEPTH_MAP.value,
                dims=("depth", "lat", "lon"),
                role="mask_artifact",
                frontend_type="data_container_result",
                produced_by=node.node_id,
                provenance={
                    "vertical_mode": "bottom_band",
                    "meaning": "Boolean mask selecting depth cells within the requested distance above local bottom.",
                    "broadcast_policy": "broadcast to time/depth/lat/lon field by named coordinates",
                },
            )
        )
    return tuple(auxiliaries)


def _vertical_auxiliary_masks(node: TaskNode) -> Tuple[MaskRequirementContract, ...]:
    if node.operation != "select_vertical" or node.output is None or node.execution is None:
        return ()
    params = node.execution.params
    mode = str(params.get("mode") or "").strip().lower()
    relative_to = str(params.get("relative_to") or "").strip().lower()
    if mode not in {"bottom_band", "relative_to_bottom"} and relative_to != "bottom":
        return ()
    output_id = node.output.artifact_id
    return (
        MaskRequirementContract(
            mask_id=f"mask:{output_id}_valid_bottom_band_mask",
            artifact_id=f"{output_id}_valid_bottom_band_mask",
            dims=("depth", "lat", "lon"),
            shape_class=ShapeClass.FIELD_3D_DEPTH_MAP.value,
            role="valid_bottom_band_mask",
            applies_to=(output_id,),
            broadcast_policy="coordinate_aligned_to_selected_field",
        ),
    )


def _mask_label(index: int, mask: Any) -> str:
    role = getattr(mask, "role", "") or ""
    source = getattr(mask, "source", "") or ""
    return str(role or source or f"mask_{index + 1}")


def _artifact_role(spec: ArtifactSpec) -> str:
    if spec.kind in {ArtifactKind.FIELD, ArtifactKind.DATA, ArtifactKind.DATASET}:
        return "data_artifact"
    if spec.kind == ArtifactKind.MASK:
        return "mask_artifact"
    if spec.frontend_type.value != "data_container_result":
        return "frontend_result"
    return "intermediate_artifact"


def _vertical_to_dict(vertical: VerticalSpec) -> Dict[str, Any]:
    payload = {
        "mode": vertical.mode,
        "depth_value": vertical.depth_value,
        "depth_range": list(vertical.depth_range) if vertical.depth_range is not None else None,
        "relative_to": vertical.relative_to,
        "band_thickness_m": vertical.band_thickness_m,
        "aggregation": vertical.aggregation,
        "retain_depth": vertical.retain_depth,
        "source_text": vertical.source_text,
    }
    mode = str(vertical.mode or "").strip().lower()
    if mode in {"bottom", "bottom_band"} or str(vertical.relative_to or "").strip().lower() == "bottom":
        payload.update(
            {
                "coordinate_mode": "per_cell_local_bottom",
                "selector_type": "deepest_valid_wet_level",
                "bottom_depth_auxiliary": "bottom_depth(lat, lon) or bottom_depth(time, lat, lon)",
                "is_roi_mask": False,
            }
        )
    elif mode in {"surface", "fixed_depth", "depth_range"}:
        payload.update(
            {
                "coordinate_mode": "global_depth_coordinate",
                "selector_type": mode,
                "is_roi_mask": False,
            }
        )
    if mode == "bottom_band":
        payload.update(
            {
                "bottom_band_mask_auxiliary": "valid_bottom_band_mask(depth, lat, lon)",
                "bottom_band_semantics": "cells within band_thickness_m above each local bottom depth",
            }
        )
    return {key: value for key, value in payload.items() if value is not None and value != ""}


def _data_requirement_to_dict(item: DataRequirementContract) -> Dict[str, Any]:
    return {
        "requirement_id": item.requirement_id,
        "artifact_id": item.artifact_id,
        "variables": list(item.variables),
        "dataset": item.dataset,
        "region": _jsonify(item.region),
        "time_range": list(item.time_range) if item.time_range is not None else None,
        "vertical": _jsonify(item.vertical),
        "expected_shape": item.expected_shape,
        "mask_requirements": list(item.mask_requirements),
    }


def _mask_requirement_to_dict(item: MaskRequirementContract) -> Dict[str, Any]:
    return {
        "mask_id": item.mask_id,
        "artifact_id": item.artifact_id,
        "dims": list(item.dims),
        "shape_class": item.shape_class,
        "role": item.role,
        "applies_to": list(item.applies_to),
        "broadcast_policy": item.broadcast_policy,
    }


def _artifact_contract_to_dict(item: ArtifactContract) -> Dict[str, Any]:
    return {
        "artifact_id": item.artifact_id,
        "kind": item.kind,
        "shape_class": item.shape_class,
        "dims": list(item.dims),
        "role": item.role,
        "frontend_type": item.frontend_type,
        "variable": item.variable,
        "units": item.units,
        "produced_by": item.produced_by,
        "masks": list(item.masks),
        "provenance": _jsonify(item.provenance),
    }


def _semantic_task_to_dict(item: SemanticTask) -> Dict[str, Any]:
    return {
        "task_id": item.task_id,
        "task_type": item.task_type,
        "intent": item.intent,
        "operation": item.operation,
        "inputs": dict(item.inputs),
        "outputs": list(item.outputs),
        "implementation": _jsonify(item.implementation),
        "input_contract": _jsonify(item.input_contract),
        "output_contract": _jsonify(item.output_contract),
        "validation_rules": list(item.validation_rules),
        "origin": _jsonify(item.origin),
    }


def _jsonify(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return getattr(value, "value")
    return value
