import numpy as np
import pytest
import xarray as xr

from domain.ocean.harness_ops import build_condition_mask
from packages.harness import (
    ArtifactKind,
    ExecutionSpec,
    ExecutionStrategy,
    FrontendType,
    MaskClass,
    NodeType,
    ShapeClass,
    TaskNode,
    task_graph_from_nodes,
    mask_spec_from_dims,
    shape_spec_from_dims,
    validate_mask_for_artifact,
    validate_task_graph_contracts,
)
from packages.harness.specs import ArtifactSpec


def test_shape_taxonomy_distinguishes_map_section_and_series():
    assert shape_spec_from_dims(("time", "depth", "lat", "lon")).shape_class == ShapeClass.FIELD_4D
    assert shape_spec_from_dims(("time", "lat", "lon")).shape_class == ShapeClass.FIELD_3D_TIME_MAP
    assert shape_spec_from_dims(("lat", "lon")).shape_class == ShapeClass.MAP_2D
    assert shape_spec_from_dims(("depth", "distance")).shape_class == ShapeClass.SECTION_2D
    assert shape_spec_from_dims(("time",)).shape_class == ShapeClass.SERIES_1D
    assert shape_spec_from_dims(("frequency",)).shape_class == ShapeClass.SPECTRUM_1D


def test_mask_taxonomy_and_compatibility():
    artifact = ArtifactSpec(
        artifact_id="oxygen",
        kind=ArtifactKind.FIELD,
        shape=shape_spec_from_dims(("time", "depth", "lat", "lon")),
        frontend_type=FrontendType.DATA_CONTAINER,
    )

    horizontal = mask_spec_from_dims(("lat", "lon"), role="roi")
    bathymetry = mask_spec_from_dims(("lat", "lon"), role="bathymetry")
    vertical = mask_spec_from_dims(("depth",), role="vertical")
    event = mask_spec_from_dims(("time", "depth", "lat", "lon"), role="event")

    assert horizontal.mask_class == MaskClass.HORIZONTAL
    assert bathymetry.mask_class == MaskClass.BATHYMETRY
    assert vertical.mask_class == MaskClass.VERTICAL
    assert event.mask_class == MaskClass.EVENT
    assert validate_mask_for_artifact(horizontal, artifact).ok
    assert validate_mask_for_artifact(bathymetry, artifact).ok
    assert validate_mask_for_artifact(vertical, artifact).ok
    assert validate_mask_for_artifact(event, artifact).ok


def test_xarray_dim_aliases_are_normalized():
    field = xr.DataArray(
        np.zeros((2, 3)),
        dims=("latitude", "longitude"),
        coords={"latitude": [1, 2], "longitude": [3, 4, 5]},
    )
    spec = shape_spec_from_dims(field.dims)
    assert spec.shape_class == ShapeClass.MAP_2D
    assert spec.dims == ("lat", "lon")


def test_condition_mask_from_multiple_aligned_fields():
    time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    lat = np.array([18.0, 19.0])
    lon = np.array([110.0, 111.0])
    oxygen = xr.DataArray(
        np.array([[[50, 70], [55, 65]], [[45, 80], [62, 58]]], dtype=float),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="oxygen",
    )
    temp = xr.DataArray(
        np.array([[[29, 29], [27, 30]], [[30, 26], [29, 31]]], dtype=float),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="temp",
    )

    mask = build_condition_mask(
        {"oxygen": oxygen, "temp": temp},
        "oxygen < 60 and temp > 28",
        mask_name="hot_low_oxygen",
    )

    assert mask.dtype == bool
    assert mask.dims == ("time", "lat", "lon")
    assert mask.attrs["mask_type"] == "condition"
    assert mask.attrs["source_variables"] == ["oxygen", "temp"]
    assert mask.values.tolist()[0] == [[True, False], [False, False]]


def test_condition_mask_rejects_unsafe_expression():
    field = xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"))

    with pytest.raises(ValueError):
        build_condition_mask({"oxygen": field}, "__import__('os').system('echo nope')")


def test_task_graph_contract_rejects_incompatible_mask_dims():
    field = ArtifactSpec(
        artifact_id="oxygen",
        kind=ArtifactKind.FIELD,
        shape=shape_spec_from_dims(("time", "lat", "lon")),
        frontend_type=FrontendType.DATA_CONTAINER,
    )
    bad_mask = ArtifactSpec(
        artifact_id="bad_mask",
        kind=ArtifactKind.MASK,
        shape=shape_spec_from_dims(("depth",)),
        frontend_type=FrontendType.DATA_CONTAINER,
    )
    nodes = [
        TaskNode(
            node_id="read_oxygen",
            node_type=NodeType.READ,
            intent="read",
            operation="load_dataset",
            output=field,
            execution=ExecutionSpec(strategy=ExecutionStrategy.TOOL, tool_name="load_dataset", params={}),
        ),
        TaskNode(
            node_id="bad_mask",
            node_type=NodeType.DERIVE,
            intent="mask",
            operation="build_threshold_mask",
            inputs={"data": "oxygen"},
            output=bad_mask,
            execution=ExecutionSpec(
                strategy=ExecutionStrategy.TOOL,
                tool_name="build_threshold_mask",
                params={"data": "$ref:oxygen.data", "threshold": 60},
            ),
        ),
        TaskNode(
            node_id="apply_bad_mask",
            node_type=NodeType.DERIVE,
            intent="apply",
            operation="apply_mask",
            inputs={"data": "oxygen", "mask": "bad_mask"},
            output=field,
            execution=ExecutionSpec(
                strategy=ExecutionStrategy.TOOL,
                tool_name="apply_mask",
                params={"data": "$ref:oxygen.data", "mask": "$ref:bad_mask.data"},
            ),
        ),
    ]

    validation = validate_task_graph_contracts(task_graph_from_nodes("g", nodes, final_artifacts=["oxygen"]))
    assert not validation.ok
    assert any("not compatible" in issue.message or "expected one of" in issue.message for issue in validation.issues)
