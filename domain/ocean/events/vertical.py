"""
Vertical selection helpers for event detection.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import xarray as xr

from domain.ocean.data_access.load import (
    _build_coord_slice,
    _normalize_depth_range,
    has_pending_bottom_selection,
    resolve_pending_bottom_selection,
    _select_deepest_valid_depth,
)


VerticalMode = Literal["surface", "bottom", "fixed_depth", "depth_range"]
DepthAggregation = Literal["mean", "min", "max", "median", "surface", "bottom"]


def prepare_event_vertical_field(
    data: xr.DataArray,
    *,
    default_mode: VerticalMode,
    vertical_mode: Optional[str] = None,
    depth_value: Optional[float] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    depth_aggregation: str = "mean",
) -> xr.DataArray:
    """
    Collapse a 4D (time, depth, lat, lon) event field into the 3D field required
    by detection algorithms while preserving a configurable depth semantic.
    """
    if "depth" not in data.dims:
        return data

    mode = (vertical_mode or default_mode).lower()

    if mode == "surface":
        selected = data.isel(depth=0)
    elif mode == "bottom":
        if has_pending_bottom_selection(data):
            selected = resolve_pending_bottom_selection(
                data,
                label="event bottom field",
                start=0.08,
                end=0.12,
            )
        else:
            selected = _select_deepest_valid_depth(data, "depth")
    elif mode == "fixed_depth":
        target_depth = _resolve_depth_value(depth_value=depth_value, depth_range=depth_range)
        if target_depth is None:
            raise ValueError("vertical_mode='fixed_depth' requires depth_value or a single-valued depth_range.")
        selected = data.sel(depth=target_depth, method="nearest")
    elif mode == "depth_range":
        if depth_range is None:
            raise ValueError("vertical_mode='depth_range' requires depth_range.")
        depth_values = data["depth"].values
        normalized_range = _normalize_depth_range(depth_values, depth_range)
        window = data.sel(depth=_build_coord_slice(depth_values, normalized_range))
        if window.sizes.get("depth", 0) == 0:
            raise ValueError(
                "No depth levels fall within the requested depth_range "
                f"[{depth_range[0]}, {depth_range[1]}]."
            )
        selected = _aggregate_depth_window(window, depth_aggregation)
    else:
        raise ValueError(f"Unsupported vertical_mode: {vertical_mode}")

    if "depth" in selected.dims and selected.sizes.get("depth") == 1:
        selected = selected.isel(depth=0)

    selected.attrs = {
        **getattr(selected, "attrs", {}),
        "vertical_mode": mode,
        "depth_value": depth_value,
        "depth_range": list(depth_range) if depth_range is not None else None,
        "depth_aggregation": depth_aggregation,
    }
    return selected


def _resolve_depth_value(
    *,
    depth_value: Optional[float],
    depth_range: Optional[Tuple[float, float]],
) -> Optional[float]:
    if depth_value is not None:
        return float(depth_value)
    if depth_range is None:
        return None
    if len(depth_range) != 2:
        return None
    if abs(float(depth_range[0]) - float(depth_range[1])) > 1e-6:
        return None
    return float(depth_range[0])


def _aggregate_depth_window(data: xr.DataArray, depth_aggregation: str) -> xr.DataArray:
    aggregation = depth_aggregation.lower()
    if aggregation == "mean":
        return data.mean(dim="depth", skipna=True)
    if aggregation == "min":
        return data.min(dim="depth", skipna=True)
    if aggregation == "max":
        return data.max(dim="depth", skipna=True)
    if aggregation == "median":
        return data.median(dim="depth", skipna=True)
    if aggregation == "surface":
        return data.isel(depth=0)
    if aggregation == "bottom":
        return data.isel(depth=-1)
    raise ValueError(f"Unsupported depth_aggregation: {depth_aggregation}")
