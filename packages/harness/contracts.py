"""Validation contracts for OceanMind harness artifacts and task graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from packages.harness.ir import TaskGraph
from packages.harness.masks import MaskSpec, mask_compatible_with_shape, mask_spec_from_dims
from packages.harness.shapes import ShapeClass, normalize_dims, shape_spec_from_dims
from packages.harness.specs import ArtifactKind, ArtifactSpec, FrontendType
from packages.harness.tool_contracts import PortSpec, get_tool_shape_contract


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    message: str
    path: str = ""


@dataclass
class ValidationResult:
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def add(self, level: str, message: str, path: str = "") -> None:
        self.issues.append(ValidationIssue(level=level, message=message, path=path))

    def extend(self, other: "ValidationResult") -> None:
        self.issues.extend(other.issues)

    def raise_for_errors(self) -> None:
        errors = [issue for issue in self.issues if issue.level == "error"]
        if errors:
            message = "; ".join(issue.message for issue in errors)
            raise ValueError(message)


def infer_frontend_type(shape_class: ShapeClass, kind: ArtifactKind) -> FrontendType:
    if kind == ArtifactKind.SPECTRUM or shape_class == ShapeClass.SPECTRUM_1D:
        return FrontendType.SPECTRUM
    if kind == ArtifactKind.SERIES or shape_class == ShapeClass.SERIES_1D:
        return FrontendType.TIMESERIES
    if kind == ArtifactKind.PROFILE or shape_class == ShapeClass.PROFILE_1D:
        return FrontendType.PROFILE
    if kind == ArtifactKind.SECTION or shape_class == ShapeClass.SECTION_2D:
        return FrontendType.SECTION
    if kind == ArtifactKind.MAP or shape_class == ShapeClass.MAP_2D:
        return FrontendType.SPATIAL_FIELD
    if kind in {ArtifactKind.DATA, ArtifactKind.FIELD, ArtifactKind.DATASET, ArtifactKind.MASK}:
        return FrontendType.DATA_CONTAINER
    return FrontendType.GENERIC


def spec_from_xarray(
    artifact_id: str,
    data: Any,
    *,
    kind: ArtifactKind = ArtifactKind.FIELD,
    frontend_type: Optional[FrontendType] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> ArtifactSpec:
    dims = tuple(getattr(data, "dims", ()))
    shape = shape_spec_from_dims(dims)
    units = ""
    variable = None
    attrs = getattr(data, "attrs", None)
    if isinstance(attrs, dict):
        units = str(attrs.get("units") or attrs.get("unit") or "")
    name = getattr(data, "name", None)
    if isinstance(name, str) and name:
        variable = name
    coords = _extract_coord_summary(data)
    resolved_frontend_type = frontend_type or infer_frontend_type(shape.shape_class, kind)
    return ArtifactSpec(
        artifact_id=artifact_id,
        kind=kind,
        shape=shape,
        units=units,
        coords=coords,
        frontend_type=resolved_frontend_type,
        variable=variable,
        provenance=provenance or {},
    )


def validate_artifact_spec(spec: ArtifactSpec) -> ValidationResult:
    result = ValidationResult()
    if not spec.artifact_id:
        result.add("error", "ArtifactSpec requires artifact_id", "artifact_id")
    if spec.shape.shape_class == ShapeClass.UNKNOWN:
        result.add("warning", f"Unknown shape for dims {list(spec.dims)}", "shape")
    if spec.kind == ArtifactKind.MASK and not spec.dims:
        result.add("error", "Mask artifact must declare dimensions", "dims")
    return result


def validate_mask_for_artifact(mask: MaskSpec, artifact: ArtifactSpec) -> ValidationResult:
    result = ValidationResult()
    if not mask_compatible_with_shape(mask, artifact.shape_class, artifact.dims):
        result.add(
            "error",
            (
                f"{mask.mask_class.value} with dims {list(mask.dims)} is not compatible "
                f"with {artifact.shape_class.value} dims {list(artifact.dims)}"
            ),
            "mask",
        )
    return result


def validate_xarray_matches_spec(data: Any, spec: ArtifactSpec) -> ValidationResult:
    result = ValidationResult()
    dims = normalize_dims(getattr(data, "dims", ()))
    if dims != spec.dims:
        result.add(
            "error",
            f"Data dims {list(dims)} do not match spec dims {list(spec.dims)}",
            "dims",
        )
    actual_shape = shape_spec_from_dims(dims).shape_class
    if actual_shape != spec.shape_class:
        result.add(
            "error",
            f"Data shape {actual_shape.value} does not match spec shape {spec.shape_class.value}",
            "shape_class",
        )
    return result


def validate_mask_broadcast(mask: Any, artifact: ArtifactSpec, *, role: str = "") -> ValidationResult:
    dims = tuple(getattr(mask, "dims", ()))
    mask_spec = mask_spec_from_dims(dims, role=role)
    return validate_mask_for_artifact(mask_spec, artifact)


def validate_task_graph(graph: TaskGraph) -> ValidationResult:
    result = ValidationResult()
    seen_node_ids: set[str] = set()
    available_artifacts: Dict[str, ArtifactSpec] = {}

    for node in graph.nodes:
        if not node.node_id:
            result.add("error", "Task node requires node_id", "nodes")
            continue
        if node.node_id in seen_node_ids:
            result.add("error", f"Duplicate node_id: {node.node_id}", node.node_id)
        seen_node_ids.add(node.node_id)

        for input_name, artifact_id in node.inputs.items():
            if artifact_id not in available_artifacts:
                result.add(
                    "error",
                    f"Node {node.node_id} input {input_name} references unknown artifact {artifact_id}",
                    node.node_id,
                )
                continue
            expected_shape = node.expected_input_shapes.get(input_name)
            if expected_shape is not None:
                actual_shape = available_artifacts[artifact_id].shape_class
                if actual_shape != expected_shape:
                    result.add(
                        "error",
                        (
                            f"Node {node.node_id} input {input_name} expected "
                            f"{expected_shape.value}, got {actual_shape.value}"
                        ),
                        node.node_id,
                    )

        if node.output is not None:
            spec_result = validate_artifact_spec(node.output)
            result.extend(spec_result)
            if node.output.artifact_id in available_artifacts:
                result.add(
                    "error",
                    f"Duplicate artifact_id: {node.output.artifact_id}",
                    node.node_id,
                )
            available_artifacts[node.output.artifact_id] = node.output

    for artifact_id in graph.final_artifacts:
        if artifact_id not in available_artifacts:
            result.add("error", f"Final artifact is not produced: {artifact_id}", "final_artifacts")

    return result


def validate_task_graph_contracts(graph: TaskGraph) -> ValidationResult:
    """Validate task graph refs against centralized tool shape contracts."""
    result = validate_task_graph(graph)
    available_artifacts: Dict[str, ArtifactSpec] = {}

    for node in graph.nodes:
        contract = get_tool_shape_contract((node.execution.tool_name if node.execution else None) or node.operation)
        params = node.execution.params if node.execution is not None else {}
        param_refs = _collect_param_refs(params)

        for path, artifact_id in param_refs:
            artifact = available_artifacts.get(artifact_id)
            if artifact is None:
                continue
            if contract is None:
                continue
            root = _root_param(path)
            port = _port_for_param(contract.inputs, root)
            if port is None:
                continue
            _validate_port_artifact(result, node.node_id, root, port, artifact)

        if contract is not None:
            _validate_mask_param_compatibility(result, node.node_id, contract.inputs, param_refs, available_artifacts)
            if node.output is not None and contract.output is not None:
                output = contract.output
                if node.output.kind != output.kind:
                    result.add(
                        "error",
                        (
                            f"Node {node.node_id} output {node.output.artifact_id} expected kind "
                            f"{output.kind.value}, got {node.output.kind.value}"
                        ),
                        node.node_id,
                    )
                if output.shapes and node.output.shape_class not in output.shapes:
                    result.add(
                        "error",
                        (
                            f"Node {node.node_id} output {node.output.artifact_id} expected one of "
                            f"{[shape.value for shape in output.shapes]}, got {node.output.shape_class.value}"
                        ),
                        node.node_id,
                    )

        if node.output is not None:
            available_artifacts[node.output.artifact_id] = node.output

    return result


def validate_tool_runtime_result(
    tool_name: str,
    artifact_id: str,
    data: Any,
    *,
    output_type: str = "",
) -> ValidationResult:
    """Validate a tool's concrete xarray output against its shape contract."""
    result = ValidationResult()
    contract = get_tool_shape_contract(tool_name)
    if contract is None or contract.output is None or data is None or not hasattr(data, "dims"):
        return result
    output = contract.output
    spec = spec_from_xarray(
        artifact_id,
        data,
        kind=output.kind,
        frontend_type=FrontendType(output_type) if output_type in FrontendType._value2member_map_ else output.frontend_type,
    )
    if output.shapes and spec.shape_class not in output.shapes:
        result.add(
            "error",
            (
                f"Tool {tool_name} output {artifact_id} expected one of "
                f"{[shape.value for shape in output.shapes]}, got {spec.shape_class.value}"
            ),
            artifact_id,
        )
    if output.kind == ArtifactKind.MASK:
        mask_role = output.role or str(getattr(data, "attrs", {}).get("mask_type") or "mask")
        mask_spec = mask_spec_from_dims(spec.dims, role=mask_role)
        if mask_spec.mask_class.value == "UnknownMask":
            result.add("error", f"Tool {tool_name} produced mask with unsupported dims {list(spec.dims)}", artifact_id)
    if output.boolean_required:
        dtype = getattr(data, "dtype", None)
        if dtype is None or dtype != bool:
            result.add("error", f"Tool {tool_name} must produce boolean mask data", artifact_id)
    return result


def _extract_coord_summary(data: Any) -> Mapping[str, Any]:
    coords = getattr(data, "coords", None)
    if coords is None:
        return {}
    summary: Dict[str, Any] = {}
    for name in coords:
        coord = coords[name]
        try:
            size = int(coord.size)
        except Exception:
            size = None
        entry: Dict[str, Any] = {}
        if size is not None:
            entry["size"] = size
        try:
            values = coord.values
            if size and size > 0:
                entry["start"] = _safe_coord_value(values[0])
                entry["end"] = _safe_coord_value(values[-1])
        except Exception:
            pass
        summary[str(name)] = entry
    return summary


def _collect_param_refs(value: Any, path: str = "") -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = []
    if isinstance(value, str) and value.startswith("$ref:"):
        artifact_id = value[5:].split(".", 1)[0]
        refs.append((path, artifact_id))
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            nested_path = f"{path}.{key}" if path else str(key)
            refs.extend(_collect_param_refs(nested, nested_path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            nested_path = f"{path}[{index}]" if path else f"[{index}]"
            refs.extend(_collect_param_refs(nested, nested_path))
    return refs


def _root_param(path: str) -> str:
    root = path.split(".", 1)[0]
    return root.split("[", 1)[0]


def _port_for_param(inputs: Mapping[str, PortSpec], root: str) -> Optional[PortSpec]:
    if root in inputs:
        return inputs[root]
    return None


def _validate_port_artifact(
    result: ValidationResult,
    node_id: str,
    param_name: str,
    port: PortSpec,
    artifact: ArtifactSpec,
) -> None:
    if port.kind is not None and artifact.kind != port.kind:
        result.add(
            "error",
            (
                f"Node {node_id} param {param_name} expected {port.kind.value} artifact, "
                f"got {artifact.kind.value} artifact {artifact.artifact_id}"
            ),
            node_id,
        )
    if port.shapes and artifact.shape_class not in port.shapes:
        result.add(
            "error",
            (
                f"Node {node_id} param {param_name} expected shape in "
                f"{[shape.value for shape in port.shapes]}, got {artifact.shape_class.value} "
                f"from {artifact.artifact_id}"
            ),
            node_id,
        )


def _validate_mask_param_compatibility(
    result: ValidationResult,
    node_id: str,
    inputs: Mapping[str, PortSpec],
    param_refs: Iterable[Tuple[str, str]],
    available_artifacts: Mapping[str, ArtifactSpec],
) -> None:
    refs_by_root: Dict[str, List[str]] = {}
    for path, artifact_id in param_refs:
        refs_by_root.setdefault(_root_param(path), []).append(artifact_id)

    data_candidates = [
        artifact_id
        for root in ("data", "field", "temp", "oxygen", "chlorophyll", "u", "v")
        for artifact_id in refs_by_root.get(root, [])
    ]
    target_id = data_candidates[0] if data_candidates else None
    target = available_artifacts.get(target_id) if target_id else None

    for root in ("mask", "analysis_mask", "event_mask"):
        for mask_id in refs_by_root.get(root, []):
            mask = available_artifacts.get(mask_id)
            if mask is None:
                continue
            role = inputs.get(root).role if root in inputs else root
            if target is not None:
                result.extend(validate_mask_for_artifact(mask_spec_from_dims(mask.dims, role=role), target))

    if "masks" in refs_by_root:
        mask_specs = [
            available_artifacts[artifact_id]
            for artifact_id in refs_by_root["masks"]
            if artifact_id in available_artifacts
        ]
        if len(mask_specs) >= 2:
            union_dims = set(mask_specs[0].dims)
            for spec in mask_specs[1:]:
                union_dims.update(spec.dims)
            for spec in mask_specs:
                if not set(spec.dims).issubset(union_dims):
                    result.add("error", f"Mask {spec.artifact_id} cannot be combined with sibling masks", node_id)


def _safe_coord_value(value: Any) -> Any:
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass
    return str(value)
