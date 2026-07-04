"""Core specs for OceanMind harness reads and artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from packages.harness.masks import MaskSpec
from packages.harness.shapes import ShapeClass, ShapeSpec, shape_spec_from_dims


class ArtifactKind(str, Enum):
    DATA = "data"
    FIELD = "field"
    DATASET = "dataset"
    MASK = "mask"
    SERIES = "series"
    PROFILE = "profile"
    SPECTRUM = "spectrum"
    HOVMOLLER = "hovmoller"
    MAP = "map"
    SECTION = "section"
    TABLE = "table"
    SCALAR = "scalar"
    REPORT = "report"
    GENERIC = "generic"


class FrontendType(str, Enum):
    DATA_CONTAINER = "data_container_result"
    TIMESERIES = "timeseries_result"
    TREND = "trend_result"
    SPATIAL_FIELD = "spatial_field_result"
    FIELD_TREND = "field_trend_result"
    SPECTRUM = "spectrum_result"
    LAG_CORRELATION = "lag_correlation_result"
    REGRESSION_MAP = "regression_map_result"
    SECTION = "section_result"
    PROFILE = "profile_result"
    HOVMOLLER = "hovmoller_result"
    EVIDENCE_REPORT = "evidence_report_result"
    GENERIC = "generic_result"


@dataclass(frozen=True)
class VerticalSpec:
    mode: str = "unspecified"
    depth_value: Optional[float] = None
    depth_range: Optional[Tuple[float, float]] = None
    relative_to: Optional[str] = None
    band_thickness_m: Optional[float] = None
    aggregation: Optional[str] = None
    retain_depth: bool = False
    source_text: str = ""


@dataclass(frozen=True)
class ReadSpec:
    variables: Tuple[str, ...]
    region: Optional[Mapping[str, Any]] = None
    time_range: Optional[Tuple[str, str]] = None
    vertical: VerticalSpec = field(default_factory=VerticalSpec)
    expected_shape: Optional[ShapeClass] = None
    mask_requirements: Tuple[str, ...] = ()
    dataset: str = "current"


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    kind: ArtifactKind
    shape: ShapeSpec
    units: str = ""
    coords: Mapping[str, Any] = field(default_factory=dict)
    frontend_type: FrontendType = FrontendType.GENERIC
    variable: Optional[str] = None
    masks: Tuple[MaskSpec, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dims(self) -> Tuple[str, ...]:
        return self.shape.dims

    @property
    def shape_class(self) -> ShapeClass:
        return self.shape.shape_class


@dataclass
class DataBundle:
    data: Any
    spec: ArtifactSpec
    mask: Any = None
    masks: Dict[str, Any] = field(default_factory=dict)
    read_spec: Optional[ReadSpec] = None
    validation: List[Mapping[str, Any]] = field(default_factory=list)


def artifact_spec_from_dims(
    *,
    artifact_id: str,
    dims: Tuple[str, ...],
    kind: ArtifactKind = ArtifactKind.FIELD,
    units: str = "",
    frontend_type: FrontendType = FrontendType.GENERIC,
    variable: Optional[str] = None,
    coords: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> ArtifactSpec:
    return ArtifactSpec(
        artifact_id=artifact_id,
        kind=kind,
        shape=shape_spec_from_dims(dims),
        units=units,
        coords=coords or {},
        frontend_type=frontend_type,
        variable=variable,
        provenance=provenance or {},
    )
