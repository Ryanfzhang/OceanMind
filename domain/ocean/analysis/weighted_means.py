"""
Weighted regional means for gridded ocean-model fields.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import xarray as xr

from domain.ocean.data_access.load import get_depth_dim, _normalize_depth_range, resolve_pending_bottom_selection
from domain.ocean.data_access.partitioned import find_partitioned_values, materialize_partitioned_xarray
from domain.ocean.dask_utils import dataarray_to_numpy, report_phase


EARTH_RADIUS_M = 6_371_000.0


def compute_area_weighted_mean(
    data: xr.DataArray,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    depth_aggregation: Literal["mean", "max", "min", "surface", "integral"] = "mean",
) -> Dict:
    """
    Compute an area-weighted regional mean time series on a lon/lat grid.

    If lon_range/lat_range are omitted, use the full horizontal extent already
    present in ``data``. This lets upstream load_dataset subsetting define the
    region without requiring the planner to repeat the same bounds.
    """
    if find_partitioned_values((data,)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_area_weighted_mean",
            tool_func=compute_area_weighted_mean,
            params={
                "data": data,
                "lon_range": lon_range,
                "lat_range": lat_range,
                "depth_range": depth_range,
                "depth_aggregation": depth_aggregation,
            },
        )
    data = materialize_partitioned_xarray(data)
    data = resolve_pending_bottom_selection(
        data,
        label="area-weighted bottom layer",
        start=0.05,
        end=0.15,
    )

    if "time" not in data.dims:
        raise ValueError("compute_area_weighted_mean requires a time dimension")
    _require_horizontal_coords(data)
    lon_range = _resolve_horizontal_range(data, "lon", lon_range)
    lat_range = _resolve_horizontal_range(data, "lat", lat_range)

    subset = _subset_horizontal(data, lon_range, lat_range)
    subset = _aggregate_depth(subset, depth_range, depth_aggregation)
    subset = _normalize_to_timeseries_field(subset)

    weights = _horizontal_cell_area(subset["lon"].values, subset["lat"].values)
    mean = _weighted_mean(subset, weights, dims=("lat", "lon"))
    values = dataarray_to_numpy(
        mean,
        label="area-weighted mean time series",
        dtype=float,
        start=0.2,
        end=0.9,
    )
    return {
        "times": [str(value) for value in mean.time.values],
        "values": values.tolist(),
        "metadata": {
            "variable": data.name or "unknown",
            "unit": data.attrs.get("units", ""),
            "units": data.attrs.get("units", ""),
            "region": {"lon_range": list(lon_range), "lat_range": list(lat_range)},
            "depth_range": list(depth_range) if depth_range is not None else None,
            "depth_aggregation": depth_aggregation if get_depth_dim(data) is not None else None,
            "weighting": "area_weighted",
            "statistics": _compute_series_statistics(values),
        },
    }


def compute_volume_weighted_mean(
    data: xr.DataArray,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    depth_range: Tuple[float, float],
) -> Dict:
    """
    Compute a volume-weighted regional mean time series using grid-cell area and
    layer thickness.
    """
    if find_partitioned_values((data,)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_volume_weighted_mean",
            tool_func=compute_volume_weighted_mean,
            params={
                "data": data,
                "lon_range": lon_range,
                "lat_range": lat_range,
                "depth_range": depth_range,
            },
        )
    data = materialize_partitioned_xarray(data)
    data = resolve_pending_bottom_selection(
        data,
        label="area-integral bottom layer",
        start=0.05,
        end=0.15,
    )

    if "time" not in data.dims:
        raise ValueError("compute_volume_weighted_mean requires a time dimension")
    depth_dim = get_depth_dim(data)
    if depth_dim is None:
        raise ValueError("compute_volume_weighted_mean requires a depth dimension")
    _require_horizontal_coords(data)

    subset = _subset_horizontal(data, lon_range, lat_range)
    normalized_range = _normalize_depth_range(np.asarray(subset[depth_dim].values, dtype=float), depth_range)
    subset = subset.sel({depth_dim: _build_coord_slice(subset[depth_dim].values, normalized_range)})
    if subset.sizes.get(depth_dim, 0) == 0:
        raise ValueError("Selected depth_range does not include any levels")

    area_weights = _horizontal_cell_area(subset["lon"].values, subset["lat"].values)
    thickness = _layer_thickness(subset[depth_dim].values)
    volume_weights = xr.DataArray(
        thickness,
        coords={depth_dim: subset[depth_dim]},
        dims=(depth_dim,),
    ) * area_weights

    mean = _weighted_mean(subset, volume_weights, dims=(depth_dim, "lat", "lon"))
    values = dataarray_to_numpy(
        mean,
        label="volume-weighted mean time series",
        dtype=float,
        start=0.2,
        end=0.9,
    )
    return {
        "times": [str(value) for value in mean.time.values],
        "values": values.tolist(),
        "metadata": {
            "variable": data.name or "unknown",
            "unit": data.attrs.get("units", ""),
            "units": data.attrs.get("units", ""),
            "region": {"lon_range": list(lon_range), "lat_range": list(lat_range)},
            "depth_range": list(depth_range),
            "weighting": "volume_weighted",
            "statistics": _compute_series_statistics(values),
        },
    }


def compute_area_integral(
    data: xr.DataArray,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    depth_range: Optional[Tuple[float, float]] = None,
    depth_aggregation: Literal["mean", "max", "min", "surface", "integral"] = "mean",
) -> Dict:
    """
    Compute an area-integrated regional inventory time series on a lon/lat grid.
    """
    if find_partitioned_values((data,)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_area_integral",
            tool_func=compute_area_integral,
            params={
                "data": data,
                "lon_range": lon_range,
                "lat_range": lat_range,
                "depth_range": depth_range,
                "depth_aggregation": depth_aggregation,
            },
        )
    data = materialize_partitioned_xarray(data)

    if "time" not in data.dims:
        raise ValueError("compute_area_integral requires a time dimension")
    _require_horizontal_coords(data)

    subset = _subset_horizontal(data, lon_range, lat_range)
    subset = _aggregate_depth(subset, depth_range, depth_aggregation)
    subset = _normalize_to_timeseries_field(subset)

    weights = _horizontal_cell_area(subset["lon"].values, subset["lat"].values)
    integral = _weighted_sum(subset, weights, dims=("lat", "lon"))
    values = dataarray_to_numpy(
        integral,
        label="area integral time series",
        dtype=float,
        start=0.2,
        end=0.9,
    )
    return {
        "times": [str(value) for value in integral.time.values],
        "values": values.tolist(),
        "metadata": {
            "variable": data.name or "unknown",
            "unit": _multiply_units(data.attrs.get("units", ""), "m^2"),
            "units": _multiply_units(data.attrs.get("units", ""), "m^2"),
            "region": {"lon_range": list(lon_range), "lat_range": list(lat_range)},
            "depth_range": list(depth_range) if depth_range is not None else None,
            "depth_aggregation": depth_aggregation if get_depth_dim(data) is not None else None,
            "reduction": "area_integral",
            "statistics": _compute_series_statistics(values),
        },
    }


def compute_volume_integral(
    data: xr.DataArray,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    depth_range: Tuple[float, float],
) -> Dict:
    """
    Compute a volume-integrated regional inventory time series.
    """
    if find_partitioned_values((data,)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_volume_integral",
            tool_func=compute_volume_integral,
            params={
                "data": data,
                "lon_range": lon_range,
                "lat_range": lat_range,
                "depth_range": depth_range,
            },
        )
    data = materialize_partitioned_xarray(data)

    if "time" not in data.dims:
        raise ValueError("compute_volume_integral requires a time dimension")
    depth_dim = get_depth_dim(data)
    if depth_dim is None:
        raise ValueError("compute_volume_integral requires a depth dimension")
    _require_horizontal_coords(data)

    subset = _subset_horizontal(data, lon_range, lat_range)
    normalized_range = _normalize_depth_range(np.asarray(subset[depth_dim].values, dtype=float), depth_range)
    subset = subset.sel({depth_dim: _build_coord_slice(subset[depth_dim].values, normalized_range)})
    if subset.sizes.get(depth_dim, 0) == 0:
        raise ValueError("Selected depth_range does not include any levels")

    area_weights = _horizontal_cell_area(subset["lon"].values, subset["lat"].values)
    thickness = _layer_thickness(subset[depth_dim].values)
    volume_weights = xr.DataArray(
        thickness,
        coords={depth_dim: subset[depth_dim]},
        dims=(depth_dim,),
    ) * area_weights

    integral = _weighted_sum(subset, volume_weights, dims=(depth_dim, "lat", "lon"))
    values = dataarray_to_numpy(
        integral,
        label="volume integral time series",
        dtype=float,
        start=0.2,
        end=0.9,
    )
    return {
        "times": [str(value) for value in integral.time.values],
        "values": values.tolist(),
        "metadata": {
            "variable": data.name or "unknown",
            "unit": _multiply_units(data.attrs.get("units", ""), "m^3"),
            "units": _multiply_units(data.attrs.get("units", ""), "m^3"),
            "region": {"lon_range": list(lon_range), "lat_range": list(lat_range)},
            "depth_range": list(depth_range),
            "reduction": "volume_integral",
            "statistics": _compute_series_statistics(values),
        },
    }


def _subset_horizontal(
    data: xr.DataArray,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
) -> xr.DataArray:
    subset = data.sel(
        lon=_build_coord_slice(data.lon.values, lon_range),
        lat=_build_coord_slice(data.lat.values, lat_range),
    )
    if subset.sizes.get("lon", 0) == 0 or subset.sizes.get("lat", 0) == 0:
        raise ValueError("Selected lon/lat range does not include any grid cells")
    return subset


def _resolve_horizontal_range(
    data: xr.DataArray,
    coord_name: str,
    coord_range: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    if coord_range is not None:
        if len(coord_range) != 2:
            raise ValueError(f"{coord_name}_range must contain exactly two values")
        return float(coord_range[0]), float(coord_range[1])

    values = np.asarray(data[coord_name].values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Cannot infer {coord_name}_range from empty or non-finite coordinates")
    return float(np.nanmin(finite)), float(np.nanmax(finite))


def _aggregate_depth(
    data: xr.DataArray,
    depth_range: Optional[Tuple[float, float]],
    depth_aggregation: str,
) -> xr.DataArray:
    depth_dim = get_depth_dim(data)
    if depth_dim is None:
        return data

    field = data
    if depth_range is not None:
        normalized_range = _normalize_depth_range(np.asarray(field[depth_dim].values, dtype=float), depth_range)
        field = field.sel({depth_dim: _build_coord_slice(field[depth_dim].values, normalized_range)})

    if depth_aggregation == "surface":
        return field.isel({depth_dim: 0})
    if depth_aggregation == "integral":
        return _vertical_integral(field, depth_dim)
    if depth_aggregation == "mean":
        return field.mean(dim=depth_dim, skipna=True)
    if depth_aggregation == "max":
        return field.max(dim=depth_dim, skipna=True)
    if depth_aggregation == "min":
        return field.min(dim=depth_dim, skipna=True)

    raise ValueError(f"Unknown depth_aggregation: {depth_aggregation}")


def _normalize_to_timeseries_field(data: xr.DataArray) -> xr.DataArray:
    for dim in list(data.dims):
        if dim == "time":
            continue
        if data.sizes[dim] == 1 and dim not in {"lat", "lon"}:
            data = data.squeeze(dim, drop=True)

    remaining = [dim for dim in data.dims if dim not in {"time", "lat", "lon"}]
    if remaining:
        raise ValueError(
            f"Weighted mean expects only time/lat/lon after depth aggregation; remaining dims: {remaining}"
        )
    return data


def _weighted_mean(data: xr.DataArray, weights: xr.DataArray, dims: Tuple[str, ...]) -> xr.DataArray:
    aligned_data, aligned_weights = xr.broadcast(data, weights)
    valid = np.isfinite(aligned_data)
    weighted_data = xr.where(valid, aligned_data, 0.0) * aligned_weights
    effective_weights = xr.where(valid, aligned_weights, 0.0)
    numerator = weighted_data.sum(dim=dims, skipna=True)
    denominator = effective_weights.sum(dim=dims, skipna=True)
    return xr.where(denominator > 0.0, numerator / denominator, np.nan)


def _weighted_sum(data: xr.DataArray, weights: xr.DataArray, dims: Tuple[str, ...]) -> xr.DataArray:
    aligned_data, aligned_weights = xr.broadcast(data, weights)
    valid = np.isfinite(aligned_data)
    weighted_data = xr.where(valid, aligned_data, 0.0) * aligned_weights
    return weighted_data.sum(dim=dims, skipna=True)


def _horizontal_cell_area(lon_values, lat_values) -> xr.DataArray:
    lon = np.asarray(lon_values, dtype=float)
    lat = np.asarray(lat_values, dtype=float)
    lon_bounds = np.deg2rad(_coord_bounds(lon))
    lat_bounds = np.deg2rad(_coord_bounds(lat))

    lon_width = np.abs(np.diff(lon_bounds))
    lat_strip = np.abs(np.sin(lat_bounds[1:]) - np.sin(lat_bounds[:-1]))
    area = (EARTH_RADIUS_M ** 2) * lat_strip[:, None] * lon_width[None, :]
    return xr.DataArray(
        area,
        coords={"lat": lat_values, "lon": lon_values},
        dims=("lat", "lon"),
        name="cell_area",
    )


def _layer_thickness(depth_values) -> np.ndarray:
    depth = np.asarray(depth_values, dtype=float)
    if depth.size == 1:
        return np.array([1.0], dtype=float)
    depth_abs = np.abs(depth)
    bounds = _coord_bounds(depth_abs)
    return np.abs(np.diff(bounds))


def _coord_bounds(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)

    bounds = np.empty(values.size + 1, dtype=float)
    bounds[1:-1] = 0.5 * (values[:-1] + values[1:])
    bounds[0] = values[0] - 0.5 * (values[1] - values[0])
    bounds[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return bounds


def _build_coord_slice(values, coord_range: Tuple[float, float]) -> slice:
    start, end = coord_range
    values = np.asarray(values, dtype=float)
    ascending = values[0] <= values[-1]
    if ascending:
        return slice(min(start, end), max(start, end))
    return slice(max(start, end), min(start, end))


def _coord_extent(values) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values[0]), float(values[-1])


def _vertical_integral(data: xr.DataArray, depth_dim: str) -> xr.DataArray:
    depth_values = np.asarray(data[depth_dim].values, dtype=float)
    if depth_values.size < 2:
        raise ValueError("Vertical integration requires at least two depth levels")

    integrated = data.integrate(depth_dim)
    integrated = np.abs(integrated)
    integrated.attrs = {
        **data.attrs,
        "long_name": f"Vertical integral of {data.name or 'field'}",
        "units": f"{data.attrs.get('units', '')} * m",
        "aggregation": "vertical_integral",
    }
    return integrated


def _require_horizontal_coords(data: xr.DataArray) -> None:
    missing = [name for name in ("lon", "lat") if name not in data.coords]
    if missing:
        raise ValueError(f"Missing required horizontal coordinates: {missing}")


def _compute_series_statistics(values: np.ndarray) -> Dict:
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


def _multiply_units(base_units: str, factor_units: str) -> str:
    base_units = (base_units or "").strip()
    if not base_units:
        return factor_units
    return f"{base_units} * {factor_units}"
