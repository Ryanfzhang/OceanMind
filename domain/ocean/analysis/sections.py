"""
Section extraction and section-based Hovmoller diagnostics.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import xarray as xr

from domain.ocean.analysis._transect import _interp_along_transect
from domain.ocean.data_access.load import get_depth_dim
from domain.ocean.data_access.partitioned import materialize_partitioned_xarray
from domain.ocean.result_payload import as_numeric_array


def extract_transect_section(
    data: xr.DataArray,
    transect_points,
    n_samples: int = 200,
    method: Literal["linear", "nearest"] = "linear",
) -> Dict:
    """
    Extract a sampled section along an arbitrary lon/lat transect.
    """
    data = materialize_partitioned_xarray(data)

    section, sample_info = _interp_along_transect(
        data=data,
        transect_points=transect_points,
        n_samples=n_samples,
        method=method,
    )
    depth_dim = get_depth_dim(section)

    dims: list[str] = []
    if "time" in section.dims:
        dims.append("time")
    if depth_dim is not None and depth_dim in section.dims:
        dims.append(depth_dim)
    dims.append("distance")

    ordered = section.transpose(*dims)
    values = np.asarray(ordered.values, dtype=float)

    result = {
        "distance_km": [float(value) for value in ordered.distance.values],
        "sample_points": sample_info["sample_points"],
        "values": values,
        "metadata": {
            "variable": data.name or "unknown",
            "units": data.attrs.get("units", ""),
            "transect_points": sample_info["transect_points"],
            "n_samples": int(n_samples),
            "method": method,
            "dims": dims,
            "statistics": _compute_statistics(values),
        },
    }

    if "time" in ordered.dims:
        result["time"] = [str(value) for value in ordered.time.values]
    if depth_dim is not None and depth_dim in ordered.dims:
        result["depth"] = [float(value) for value in ordered[depth_dim].values]

    return result


def compute_section_hovmoller(
    section: Dict,
    diagram_type: Literal["time_distance", "time_depth"],
    fixed_depth: Optional[float] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    fixed_distance_km: Optional[float] = None,
    distance_range_km: Optional[Tuple[float, float]] = None,
    aggregate_method: Literal["mean", "max", "min"] = "mean",
) -> Dict:
    """
    Collapse a section result into a two-dimensional Hovmoller matrix.
    """
    data = _section_result_to_array(section)
    if "time" not in data.dims:
        raise ValueError("compute_section_hovmoller requires a section with a time dimension")

    depth_dim = get_depth_dim(data)

    if diagram_type == "time_distance":
        if depth_dim is None:
            ordered = data.transpose("time", "distance")
        else:
            if fixed_depth is not None and depth_range is not None:
                raise ValueError("fixed_depth and depth_range are mutually exclusive")
            if fixed_depth is not None:
                collapsed = data.sel({depth_dim: fixed_depth}, method="nearest")
            elif depth_range is not None:
                collapsed = _select_depth_range(data, depth_dim, depth_range)
                collapsed = _apply_aggregation(collapsed, aggregate_method, dims=[depth_dim])
            else:
                raise ValueError("time_distance requires fixed_depth or depth_range when the section contains depth")
            ordered = collapsed.transpose("time", "distance")
        spatial_coord = ordered.distance.values
        spatial_dim = "distance"
    elif diagram_type == "time_depth":
        if depth_dim is None:
            raise ValueError("time_depth requires a section with a depth dimension")
        if fixed_distance_km is not None and distance_range_km is not None:
            raise ValueError("fixed_distance_km and distance_range_km are mutually exclusive")
        if fixed_distance_km is not None:
            collapsed = data.sel(distance=fixed_distance_km, method="nearest")
        elif distance_range_km is not None:
            collapsed = data.sel(distance=_build_coord_slice(data.distance.values, distance_range_km))
            collapsed = _apply_aggregation(collapsed, aggregate_method, dims=["distance"])
        else:
            raise ValueError("time_depth requires fixed_distance_km or distance_range_km")
        if depth_range is not None:
            collapsed = _select_depth_range(collapsed, depth_dim, depth_range)
        ordered = collapsed.transpose("time", depth_dim)
        spatial_coord = ordered[depth_dim].values
        spatial_dim = depth_dim
    else:
        raise ValueError(f"Unknown diagram_type: {diagram_type}")

    values = np.asarray(ordered.values, dtype=float)
    return {
        "time": [str(value) for value in ordered.time.values],
        "spatial_coord": [float(value) for value in spatial_coord],
        "values": values,
        "metadata": {
            "diagram_type": diagram_type,
            "spatial_dim": spatial_dim,
            "aggregate_method": aggregate_method,
            "variable": data.name or section.get("metadata", {}).get("variable", "unknown"),
            "units": section.get("metadata", {}).get("units", ""),
            "statistics": _compute_statistics(values),
        },
    }


def _section_result_to_array(section: Dict) -> xr.DataArray:
    metadata = section.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("section metadata is required")

    values = as_numeric_array(section.get("values"))
    coords = {}
    dims = []

    if "time" in section:
        coords["time"] = np.asarray(section["time"])
        dims.append("time")
    if "depth" in section:
        coords["depth"] = np.asarray(section["depth"], dtype=float)
        dims.append("depth")

    coords["distance"] = np.asarray(section["distance_km"], dtype=float)
    dims.append("distance")

    expected_shape = tuple(len(coords[dim]) for dim in dims)
    if values.shape != expected_shape:
        raise ValueError(f"Section values shape {values.shape} does not match coordinates {expected_shape}")

    array = xr.DataArray(
        values,
        coords=coords,
        dims=dims,
        name=metadata.get("variable", "unknown"),
        attrs={
            "units": metadata.get("units", ""),
            "transect_points": metadata.get("transect_points"),
        },
    )
    return array


def _build_coord_slice(values, coord_range: Tuple[float, float]) -> slice:
    start, end = coord_range
    values = np.asarray(values, dtype=float)
    ascending = values[0] <= values[-1]
    if ascending:
        return slice(min(start, end), max(start, end))
    return slice(max(start, end), min(start, end))


def _select_depth_range(
    data: xr.DataArray,
    depth_dim: str,
    depth_range: Tuple[float, float],
) -> xr.DataArray:
    from domain.ocean.data_access.load import _normalize_depth_range

    depth_values = np.asarray(data[depth_dim].values, dtype=float)
    normalized_range = _normalize_depth_range(depth_values, depth_range)
    return data.sel({depth_dim: _build_coord_slice(depth_values, normalized_range)})


def _apply_aggregation(data: xr.DataArray, aggregation: str, dims: list[str]) -> xr.DataArray:
    if aggregation == "mean":
        return data.mean(dim=dims, skipna=True)
    if aggregation == "max":
        return data.max(dim=dims, skipna=True)
    if aggregation == "min":
        return data.min(dim=dims, skipna=True)
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def _compute_statistics(values: np.ndarray) -> Dict:
    values = as_numeric_array(values)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {"n_valid": 0}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "n_valid": int(valid.size),
    }
