"""Central shape contracts for harness tool planning and runtime checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

from packages.harness.shapes import ShapeClass
from packages.harness.specs import ArtifactKind, FrontendType


FIELD_SHAPES: Tuple[ShapeClass, ...] = (
    ShapeClass.FIELD_4D,
    ShapeClass.FIELD_3D_TIME_MAP,
    ShapeClass.FIELD_3D_DEPTH_MAP,
    ShapeClass.MAP_2D,
)
MASK_SHAPES: Tuple[ShapeClass, ...] = FIELD_SHAPES + (
    ShapeClass.SERIES_1D,
    ShapeClass.PROFILE_1D,
)
SERIES_SHAPES: Tuple[ShapeClass, ...] = (ShapeClass.SERIES_1D,)
MAP_SHAPES: Tuple[ShapeClass, ...] = (ShapeClass.MAP_2D,)
TABLE_SHAPES: Tuple[ShapeClass, ...] = (ShapeClass.TABLE, ShapeClass.SCALAR, ShapeClass.UNKNOWN)


@dataclass(frozen=True)
class PortSpec:
    kind: Optional[ArtifactKind] = None
    shapes: Tuple[ShapeClass, ...] = ()
    role: str = ""
    allow_literal: bool = False


@dataclass(frozen=True)
class OutputSpec:
    kind: ArtifactKind
    shapes: Tuple[ShapeClass, ...] = ()
    frontend_type: FrontendType = FrontendType.GENERIC
    role: str = ""
    boolean_required: bool = False


@dataclass(frozen=True)
class ToolShapeContract:
    tool_name: str
    inputs: Mapping[str, PortSpec] = field(default_factory=dict)
    output: Optional[OutputSpec] = None
    accepts_mask_params: Tuple[str, ...] = ()


TOOL_SHAPE_CONTRACTS: Mapping[str, ToolShapeContract] = {
    "load_dataset": ToolShapeContract(
        "load_dataset",
        output=OutputSpec(ArtifactKind.FIELD, (ShapeClass.FIELD_4D,), FrontendType.DATA_CONTAINER, role="field"),
    ),
    "assemble_dataset": ToolShapeContract(
        "assemble_dataset",
        inputs={"variables": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="variables")},
        output=OutputSpec(ArtifactKind.FIELD, FIELD_SHAPES, FrontendType.DATA_CONTAINER, role="assembled_dataset"),
    ),
    "select_vertical": ToolShapeContract(
        "select_vertical",
        inputs={"data": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field")},
        output=OutputSpec(ArtifactKind.FIELD, FIELD_SHAPES, FrontendType.DATA_CONTAINER, role="field"),
    ),
    "build_threshold_mask": ToolShapeContract(
        "build_threshold_mask",
        inputs={"data": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field")},
        output=OutputSpec(ArtifactKind.MASK, MASK_SHAPES, FrontendType.DATA_CONTAINER, role="threshold_mask", boolean_required=True),
    ),
    "build_condition_mask": ToolShapeContract(
        "build_condition_mask",
        inputs={"fields": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="condition_fields")},
        output=OutputSpec(ArtifactKind.MASK, MASK_SHAPES, FrontendType.DATA_CONTAINER, role="condition_mask", boolean_required=True),
    ),
    "build_polygon_mask": ToolShapeContract(
        "build_polygon_mask",
        inputs={"data": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="reference_grid")},
        output=OutputSpec(ArtifactKind.MASK, MAP_SHAPES, FrontendType.DATA_CONTAINER, role="polygon_mask", boolean_required=True),
    ),
    "build_isobath_mask": ToolShapeContract(
        "build_isobath_mask",
        inputs={
            "data": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="reference_grid"),
            "bathymetry": PortSpec(ArtifactKind.FIELD, MAP_SHAPES, role="bathymetry"),
        },
        output=OutputSpec(ArtifactKind.MASK, MAP_SHAPES, FrontendType.DATA_CONTAINER, role="isobath_mask", boolean_required=True),
    ),
    "combine_masks": ToolShapeContract(
        "combine_masks",
        inputs={"masks": PortSpec(ArtifactKind.MASK, MASK_SHAPES, role="mask_list")},
        output=OutputSpec(ArtifactKind.MASK, MASK_SHAPES, FrontendType.DATA_CONTAINER, role="combined_mask", boolean_required=True),
    ),
    "apply_mask": ToolShapeContract(
        "apply_mask",
        inputs={
            "data": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field"),
            "mask": PortSpec(ArtifactKind.MASK, MASK_SHAPES, role="analysis_mask"),
        },
        output=OutputSpec(ArtifactKind.FIELD, FIELD_SHAPES, FrontendType.DATA_CONTAINER, role="masked_field"),
        accepts_mask_params=("mask",),
    ),
    "compute_masked_area_fraction_timeseries": ToolShapeContract(
        "compute_masked_area_fraction_timeseries",
        inputs={
            "event_mask": PortSpec(ArtifactKind.MASK, MASK_SHAPES, role="event_mask"),
            "analysis_mask": PortSpec(ArtifactKind.MASK, MASK_SHAPES, role="analysis_mask"),
        },
        output=OutputSpec(ArtifactKind.SERIES, SERIES_SHAPES, FrontendType.TIMESERIES),
        accepts_mask_params=("analysis_mask",),
    ),
    "compute_masked_mean_timeseries": ToolShapeContract(
        "compute_masked_mean_timeseries",
        inputs={
            "data": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field"),
            "analysis_mask": PortSpec(ArtifactKind.MASK, MASK_SHAPES, role="analysis_mask"),
            "event_mask": PortSpec(ArtifactKind.MASK, MASK_SHAPES, role="event_mask"),
        },
        output=OutputSpec(ArtifactKind.SERIES, SERIES_SHAPES, FrontendType.TIMESERIES),
        accepts_mask_params=("analysis_mask", "event_mask"),
    ),
    "compute_speed_from_uv": ToolShapeContract(
        "compute_speed_from_uv",
        inputs={
            "u": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field"),
            "v": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field"),
        },
        output=OutputSpec(ArtifactKind.FIELD, FIELD_SHAPES, FrontendType.DATA_CONTAINER),
    ),
    "extract_timeseries": ToolShapeContract(
        "extract_timeseries",
        inputs={"data": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field")},
        output=OutputSpec(ArtifactKind.SERIES, SERIES_SHAPES, FrontendType.TIMESERIES),
    ),
    "extract_regional_mean": ToolShapeContract(
        "extract_regional_mean",
        inputs={"data": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field")},
        output=OutputSpec(ArtifactKind.SERIES, SERIES_SHAPES, FrontendType.TIMESERIES),
    ),
    "compute_spatial_field": ToolShapeContract(
        "compute_spatial_field",
        inputs={
            "data": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field"),
            "mask": PortSpec(ArtifactKind.MASK, MASK_SHAPES, role="analysis_mask"),
        },
        output=OutputSpec(ArtifactKind.MAP, MAP_SHAPES, FrontendType.SPATIAL_FIELD),
        accepts_mask_params=("mask",),
    ),
    "compute_transport_streamfunction_map": ToolShapeContract(
        "compute_transport_streamfunction_map",
        inputs={
            "u": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="zonal_velocity"),
            "v": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="meridional_velocity"),
        },
        output=OutputSpec(ArtifactKind.MAP, MAP_SHAPES, FrontendType.SPATIAL_FIELD, role="transport_streamfunction"),
    ),
    "compute_field_trend": ToolShapeContract(
        "compute_field_trend",
        inputs={"data": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field")},
        output=OutputSpec(ArtifactKind.MAP, MAP_SHAPES, FrontendType.FIELD_TREND),
    ),
    "compute_density": ToolShapeContract(
        "compute_density",
        inputs={"data": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="assembled_temperature_salinity")},
        output=OutputSpec(ArtifactKind.FIELD, FIELD_SHAPES, FrontendType.DATA_CONTAINER, role="density"),
    ),
    "compute_vertical_stability_timeseries": ToolShapeContract(
        "compute_vertical_stability_timeseries",
        inputs={
            "data": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field"),
            "temp": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="temperature"),
            "salt": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="salinity"),
            "density": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="density"),
        },
        output=OutputSpec(ArtifactKind.SERIES, SERIES_SHAPES, FrontendType.TIMESERIES, role="stability_timeseries"),
    ),
    "compute_event_summary_map": ToolShapeContract(
        "compute_event_summary_map",
        inputs={
            "event_detection": PortSpec(ArtifactKind.TABLE, TABLE_SHAPES, role="event_detection"),
            "data": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field"),
        },
        output=OutputSpec(ArtifactKind.MAP, MAP_SHAPES, FrontendType.SPATIAL_FIELD),
    ),
    "compute_event_frequency_map": ToolShapeContract(
        "compute_event_frequency_map",
        inputs={
            "event_detection": PortSpec(ArtifactKind.TABLE, TABLE_SHAPES, role="event_detection"),
            "data": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field"),
        },
        output=OutputSpec(ArtifactKind.MAP, MAP_SHAPES, FrontendType.SPATIAL_FIELD),
    ),
    "detect_heatwaves": ToolShapeContract(
        "detect_heatwaves",
        inputs={
            "temp": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field"),
            "analysis_mask": PortSpec(ArtifactKind.MASK, MASK_SHAPES, role="analysis_mask"),
        },
        output=OutputSpec(ArtifactKind.TABLE, TABLE_SHAPES, FrontendType.GENERIC, role="event_detection"),
        accepts_mask_params=("analysis_mask",),
    ),
    "detect_hypoxia": ToolShapeContract(
        "detect_hypoxia",
        inputs={
            "oxygen": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field"),
            "analysis_mask": PortSpec(ArtifactKind.MASK, MASK_SHAPES, role="analysis_mask"),
        },
        output=OutputSpec(ArtifactKind.TABLE, TABLE_SHAPES, FrontendType.GENERIC, role="event_detection"),
        accepts_mask_params=("analysis_mask",),
    ),
    "detect_algal_blooms": ToolShapeContract(
        "detect_algal_blooms",
        inputs={
            "chlorophyll": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field"),
            "analysis_mask": PortSpec(ArtifactKind.MASK, MASK_SHAPES, role="analysis_mask"),
        },
        output=OutputSpec(ArtifactKind.TABLE, TABLE_SHAPES, FrontendType.GENERIC, role="event_detection"),
        accepts_mask_params=("analysis_mask",),
    ),
    "detect_upwelling": ToolShapeContract(
        "detect_upwelling",
        inputs={
            "temp": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field"),
            "analysis_mask": PortSpec(ArtifactKind.MASK, MASK_SHAPES, role="analysis_mask"),
        },
        output=OutputSpec(ArtifactKind.TABLE, TABLE_SHAPES, FrontendType.GENERIC, role="event_detection"),
        accepts_mask_params=("analysis_mask",),
    ),
    "detect_eutrophication": ToolShapeContract(
        "detect_eutrophication",
        inputs={
            "chlorophyll": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field"),
            "oxygen": PortSpec(ArtifactKind.FIELD, FIELD_SHAPES, role="field"),
            "analysis_mask": PortSpec(ArtifactKind.MASK, MASK_SHAPES, role="analysis_mask"),
        },
        output=OutputSpec(ArtifactKind.TABLE, TABLE_SHAPES, FrontendType.GENERIC, role="event_detection"),
        accepts_mask_params=("analysis_mask",),
    ),
}


def get_tool_shape_contract(tool_name: str) -> Optional[ToolShapeContract]:
    return TOOL_SHAPE_CONTRACTS.get(tool_name)
