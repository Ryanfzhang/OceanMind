"""Shape taxonomy for OceanMind harness artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple


class ShapeClass(str, Enum):
    FIELD_4D = "Field4D"
    FIELD_3D_TIME_MAP = "Field3DTimeMap"
    FIELD_3D_DEPTH_MAP = "Field3DDepthMap"
    MAP_2D = "Map2D"
    SECTION_2D = "Section2D"
    HOVMOLLER_2D = "Hovmoller2D"
    SERIES_1D = "Series1D"
    PROFILE_1D = "Profile1D"
    SPECTRUM_1D = "Spectrum1D"
    TABLE = "Table"
    SCALAR = "Scalar"
    UNKNOWN = "Unknown"


CANONICAL_DIM_ORDER: Tuple[str, ...] = ("time", "depth", "lat", "lon")

_DIM_ALIASES = {
    "latitude": "lat",
    "y": "lat",
    "longitude": "lon",
    "x": "lon",
    "z": "depth",
    "lev": "depth",
    "level": "depth",
    "depth_m": "depth",
    "date": "time",
    "datetime": "time",
    "freq": "frequency",
    "period": "period",
    "distance_km": "distance",
    "sample": "distance",
}


@dataclass(frozen=True)
class ShapeSpec:
    """Canonical shape description independent of the concrete array object."""

    shape_class: ShapeClass
    dims: Tuple[str, ...]

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def has_dim(self, dim: str) -> bool:
        return normalize_dim_name(dim) in self.dims


def normalize_dim_name(name: object) -> str:
    lowered = str(name).strip().lower()
    return _DIM_ALIASES.get(lowered, lowered)


def normalize_dims(dims: Iterable[object]) -> Tuple[str, ...]:
    return tuple(normalize_dim_name(dim) for dim in dims)


def classify_dims(dims: Iterable[object]) -> ShapeClass:
    """Classify an ordered dimension tuple by semantic dimensions, not only rank."""

    normalized = normalize_dims(dims)
    dim_set = set(normalized)

    if not normalized:
        return ShapeClass.SCALAR
    if normalized == ("time", "depth", "lat", "lon"):
        return ShapeClass.FIELD_4D
    if dim_set == {"time", "depth", "lat", "lon"}:
        return ShapeClass.FIELD_4D
    if dim_set == {"time", "lat", "lon"}:
        return ShapeClass.FIELD_3D_TIME_MAP
    if dim_set == {"depth", "lat", "lon"}:
        return ShapeClass.FIELD_3D_DEPTH_MAP
    if dim_set == {"lat", "lon"}:
        return ShapeClass.MAP_2D
    if dim_set == {"depth", "distance"}:
        return ShapeClass.SECTION_2D
    if dim_set in ({"time", "distance"}, {"time", "depth"}):
        return ShapeClass.HOVMOLLER_2D
    if normalized == ("time",):
        return ShapeClass.SERIES_1D
    if normalized == ("depth",):
        return ShapeClass.PROFILE_1D
    if normalized == ("frequency",):
        return ShapeClass.SPECTRUM_1D

    return ShapeClass.UNKNOWN


def shape_spec_from_dims(dims: Iterable[object]) -> ShapeSpec:
    normalized = normalize_dims(dims)
    return ShapeSpec(shape_class=classify_dims(normalized), dims=normalized)


def canonicalize_field_dims(dims: Iterable[object]) -> Tuple[str, ...]:
    """Return field-like dims ordered as time/depth/lat/lon when possible."""

    normalized = normalize_dims(dims)
    present = set(normalized)
    ordered = tuple(dim for dim in CANONICAL_DIM_ORDER if dim in present)
    extras = tuple(dim for dim in normalized if dim not in CANONICAL_DIM_ORDER)
    return ordered + extras

