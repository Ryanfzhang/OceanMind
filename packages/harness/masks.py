"""Mask taxonomy and compatibility rules for OceanMind harness artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple

from packages.harness.shapes import ShapeClass, normalize_dims


class MaskClass(str, Enum):
    HORIZONTAL = "HorizontalMask"
    VERTICAL = "VerticalMask"
    VOLUME = "VolumeMask"
    TIME = "TimeMask"
    EVENT = "EventMask"
    SECTION = "SectionMask"
    BATHYMETRY = "BathymetryMask"
    VALID_DATA = "ValidDataMask"
    UNKNOWN = "UnknownMask"


@dataclass(frozen=True)
class MaskSpec:
    mask_class: MaskClass
    dims: Tuple[str, ...]
    role: str = ""
    source: str = ""

    @property
    def ndim(self) -> int:
        return len(self.dims)


def classify_mask_dims(dims: Iterable[object], *, role: str = "") -> MaskClass:
    normalized = normalize_dims(dims)
    dim_set = set(normalized)
    role_lower = role.strip().lower()

    if role_lower in {"bathymetry", "isobath", "bottom_depth"} and dim_set == {"lat", "lon"}:
        return MaskClass.BATHYMETRY
    if role_lower in {"event", "diagnostic", "threshold"}:
        if dim_set in ({"time", "lat", "lon"}, {"time", "depth", "lat", "lon"}):
            return MaskClass.EVENT
    if role_lower in {"valid", "wet", "valid_data", "valid_bottom_band"}:
        return MaskClass.VALID_DATA

    if dim_set == {"lat", "lon"}:
        return MaskClass.HORIZONTAL
    if dim_set == {"depth"}:
        return MaskClass.VERTICAL
    if dim_set == {"depth", "lat", "lon"}:
        return MaskClass.VOLUME
    if dim_set == {"time"}:
        return MaskClass.TIME
    if dim_set in ({"time", "lat", "lon"}, {"time", "depth", "lat", "lon"}):
        return MaskClass.EVENT
    if dim_set == {"depth", "distance"}:
        return MaskClass.SECTION

    return MaskClass.UNKNOWN


def mask_spec_from_dims(dims: Iterable[object], *, role: str = "", source: str = "") -> MaskSpec:
    normalized = normalize_dims(dims)
    return MaskSpec(
        mask_class=classify_mask_dims(normalized, role=role),
        dims=normalized,
        role=role,
        source=source,
    )


def mask_broadcasts_to_dims(mask_dims: Iterable[object], target_dims: Iterable[object]) -> bool:
    normalized_mask = set(normalize_dims(mask_dims))
    normalized_target = set(normalize_dims(target_dims))
    return bool(normalized_mask) and normalized_mask.issubset(normalized_target)


def mask_compatible_with_shape(mask: MaskSpec, shape_class: ShapeClass, target_dims: Iterable[object]) -> bool:
    if mask.mask_class == MaskClass.UNKNOWN:
        return False
    if shape_class in {ShapeClass.TABLE, ShapeClass.SCALAR, ShapeClass.UNKNOWN}:
        return mask.mask_class == MaskClass.VALID_DATA and mask_broadcasts_to_dims(mask.dims, target_dims)
    if mask.mask_class == MaskClass.HORIZONTAL:
        return shape_class in {
            ShapeClass.FIELD_4D,
            ShapeClass.FIELD_3D_TIME_MAP,
            ShapeClass.FIELD_3D_DEPTH_MAP,
            ShapeClass.MAP_2D,
        } and mask_broadcasts_to_dims(mask.dims, target_dims)
    if mask.mask_class == MaskClass.BATHYMETRY:
        return shape_class in {
            ShapeClass.FIELD_4D,
            ShapeClass.FIELD_3D_TIME_MAP,
            ShapeClass.FIELD_3D_DEPTH_MAP,
            ShapeClass.MAP_2D,
        } and mask_broadcasts_to_dims(mask.dims, target_dims)
    if mask.mask_class == MaskClass.VERTICAL:
        return shape_class in {
            ShapeClass.FIELD_4D,
            ShapeClass.FIELD_3D_DEPTH_MAP,
            ShapeClass.PROFILE_1D,
            ShapeClass.HOVMOLLER_2D,
        } and mask_broadcasts_to_dims(mask.dims, target_dims)
    if mask.mask_class == MaskClass.VOLUME:
        return shape_class in {
            ShapeClass.FIELD_4D,
            ShapeClass.FIELD_3D_DEPTH_MAP,
        } and mask_broadcasts_to_dims(mask.dims, target_dims)
    if mask.mask_class == MaskClass.TIME:
        return shape_class in {
            ShapeClass.FIELD_4D,
            ShapeClass.FIELD_3D_TIME_MAP,
            ShapeClass.SERIES_1D,
            ShapeClass.HOVMOLLER_2D,
        } and mask_broadcasts_to_dims(mask.dims, target_dims)
    if mask.mask_class == MaskClass.EVENT:
        return shape_class in {
            ShapeClass.FIELD_4D,
            ShapeClass.FIELD_3D_TIME_MAP,
        } and mask_broadcasts_to_dims(mask.dims, target_dims)
    if mask.mask_class == MaskClass.SECTION:
        return shape_class == ShapeClass.SECTION_2D and mask_broadcasts_to_dims(mask.dims, target_dims)
    if mask.mask_class == MaskClass.VALID_DATA:
        return mask_broadcasts_to_dims(mask.dims, target_dims)
    return False

