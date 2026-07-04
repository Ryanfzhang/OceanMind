"""
空间分析工具

提供二维空间场、Hovmoller 图和统一时间序列提取接口。
"""

import os
from typing import Dict, Optional, Tuple, Literal

import numpy as np
import xarray as xr

from domain.ocean.diagnostics.compute import compute_vertical_integral
from domain.ocean.data_access.partitioned import (
    find_partitioned_values,
    materialize_partitioned_xarray,
)
from domain.ocean.data_access.load import get_depth_dim, resolve_pending_bottom_selection
from domain.ocean.dask_utils import dataarray_to_numpy, is_dask_backed, report_phase
from domain.ocean.result_payload import as_numeric_array


def compute_spatial_field(
    data: xr.DataArray,
    time_range: Optional[Tuple[str, str]] = None,
    time_aggregation: Literal['mean', 'max', 'min', 'std', 'median'] = 'mean',
    depth_range: Optional[Tuple[float, float]] = None,
    depth_aggregation: Literal['mean', 'max', 'min', 'integral', 'surface'] = 'mean',
    mask: Optional[xr.DataArray] = None
) -> Dict:
    """
    计算二维空间分布场。
    """
    if find_partitioned_values((data, mask)):
        from packages.tool_loader.partitioned_execution import _compute_partitioned_spatial_field

        return _compute_partitioned_spatial_field(
            {
                "data": data,
                "time_range": time_range,
                "time_aggregation": time_aggregation,
                "depth_range": depth_range,
                "depth_aggregation": depth_aggregation,
                "mask": mask,
            }
        )
    data = materialize_partitioned_xarray(data)
    mask = materialize_partitioned_xarray(mask)
    data = resolve_pending_bottom_selection(
        data,
        label="spatial field bottom layer",
        start=0.05,
        end=0.18,
    )

    field = data

    if time_range is not None and 'time' in field.coords:
        field = field.sel(time=slice(*time_range))

    if mask is not None:
        field = field.where(mask)

    field = _aggregate_depth(field, depth_range, depth_aggregation)
    field = _aggregate_time(field, time_aggregation)

    if 'lat' not in field.dims or 'lon' not in field.dims:
        raise ValueError("Result must retain lat and lon dimensions after aggregation")

    values = dataarray_to_numpy(
        field,
        label="spatial field",
        dtype=float,
        start=0.2,
        end=0.9,
    )

    metadata = {
        'variable': data.name or 'unknown',
        'units': data.attrs.get('units', ''),
        'time_range': list(time_range) if time_range is not None else None,
        'time_aggregation': time_aggregation if 'time' in data.dims else None,
        'depth_range': list(depth_range) if depth_range is not None else None,
        'depth_aggregation': depth_aggregation if get_depth_dim(data) is not None else None,
        'statistics': _compute_statistics(values),
    }
    for attr_name in ('vertical_mode', 'bottom_selection', 'bottom_depth_coordinate'):
        attr_value = data.attrs.get(attr_name)
        if attr_value is not None:
            metadata[attr_name] = attr_value

    return {
        'lon': field.lon.values.tolist(),
        'lat': field.lat.values.tolist(),
        'values': values,
        'metadata': metadata,
    }


def extract_timeseries(
    data: xr.DataArray,
    lon: Optional[float] = None,
    lat: Optional[float] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    spatial_aggregation: Literal['mean', 'max', 'min', 'median'] = 'mean',
    depth_range: Optional[Tuple[float, float]] = None,
    depth_aggregation: Literal['mean', 'max', 'min', 'surface'] = 'mean',
    method: Literal['nearest', 'linear'] = 'nearest',
    mask: Optional[xr.DataArray] = None
) -> Dict:
    """
    统一提取点位或区域时间序列。
    """
    if find_partitioned_values((data, mask)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="extract_timeseries",
            tool_func=extract_timeseries,
            params={
                "data": data,
                "lon": lon,
                "lat": lat,
                "lon_range": lon_range,
                "lat_range": lat_range,
                "spatial_aggregation": spatial_aggregation,
                "depth_range": depth_range,
                "depth_aggregation": depth_aggregation,
                "method": method,
                "mask": mask,
            },
        )
    data = materialize_partitioned_xarray(data)
    mask = materialize_partitioned_xarray(mask)
    data = resolve_pending_bottom_selection(
        data,
        label="time series bottom layer",
        start=0.03,
        end=0.08,
    )

    if 'time' not in data.dims:
        raise ValueError("Data must have time dimension")

    is_point_mode = lon is not None and lat is not None
    is_region_mode = lon_range is not None and lat_range is not None

    if is_point_mode == is_region_mode:
        raise ValueError("Provide either (lon, lat) or (lon_range, lat_range)")

    series = data

    if mask is not None:
        series = series.where(mask)

    series = _aggregate_depth(series, depth_range, depth_aggregation)
    report_phase(
        phase="preparing_timeseries",
        message="Preparing extracted time series",
        percent=0.05,
        compute_backend="dask" if is_dask_backed(series) else "xarray",
    )

    if is_point_mode:
        if method == 'nearest':
            series = series.sel(lon=lon, lat=lat, method='nearest')
        else:
            series = series.interp(lon=lon, lat=lat, method=method)
        metadata = {
            'location': {'lon': lon, 'lat': lat},
            'mode': 'point',
            'method': method,
        }
    else:
        series = series.sel(
            lon=_build_coord_slice(series.lon.values, lon_range),
            lat=_build_coord_slice(series.lat.values, lat_range),
        )
        series = _apply_aggregation(series, spatial_aggregation, dims=['lat', 'lon'])
        metadata = {
            'region': {'lon_range': lon_range, 'lat_range': lat_range},
            'mode': 'region',
            'spatial_aggregation': spatial_aggregation,
        }

    values = dataarray_to_numpy(
        series,
        label="extracted time series",
        dtype=float,
        start=0.2,
        end=0.9,
    )
    report_phase(
        phase="timeseries_complete",
        message="Extracted time series complete",
        percent=1.0,
        compute_backend="numpy",
    )

    return {
        'times': [str(t) for t in series.time.values],
        'values': values.tolist(),
        'metadata': {
            'variable': data.name or 'unknown',
            'unit': data.attrs.get('units', ''),
            **metadata,
            'statistics': _compute_statistics(values),
        }
    }


def compute_hovmoller(
    data: xr.DataArray,
    diagram_type: Literal['time_lon', 'time_lat', 'time_depth'],
    fixed_lat: Optional[float] = None,
    fixed_lon: Optional[float] = None,
    fixed_lat_range: Optional[Tuple[float, float]] = None,
    fixed_lon_range: Optional[Tuple[float, float]] = None,
    aggregate_dim: Literal['mean', 'max', 'min'] = 'mean',
    spatial_weighting: Literal['equal', 'area_weighted', 'area'] = 'equal',
    depth: Optional[float] = None,
    depth_range: Optional[Tuple[float, float]] = None
) -> Dict:
    """
    计算 Hovmoller 图所需的二维矩阵。
    """
    if find_partitioned_values((data,)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_hovmoller",
            tool_func=compute_hovmoller,
            params={
                "data": data,
                "diagram_type": diagram_type,
                "fixed_lat": fixed_lat,
                "fixed_lon": fixed_lon,
                "fixed_lat_range": fixed_lat_range,
                "fixed_lon_range": fixed_lon_range,
                "aggregate_dim": aggregate_dim,
                "spatial_weighting": spatial_weighting,
                "depth": depth,
                "depth_range": depth_range,
            },
        )
    data = materialize_partitioned_xarray(data)
    data = resolve_pending_bottom_selection(
        data,
        label="Hovmoller bottom layer",
        start=0.03,
        end=0.08,
    )

    if 'time' not in data.dims:
        raise ValueError("Data must have time dimension")

    field = data
    aggregate_dim = _normalize_aggregate_dim(aggregate_dim)
    spatial_weighting = _normalize_spatial_weighting(spatial_weighting)
    fixed_lat = _normalize_optional_scalar(fixed_lat)
    fixed_lon = _normalize_optional_scalar(fixed_lon)
    fixed_lat_range = _normalize_optional_range(fixed_lat_range)
    fixed_lon_range = _normalize_optional_range(fixed_lon_range)
    depth_dim = get_depth_dim(field)

    if diagram_type in {'time_lon', 'time_lat'} and depth_dim is not None:
        if depth is not None and depth_range is not None:
            raise ValueError("depth and depth_range are mutually exclusive")
        if depth is not None:
            field = field.isel({depth_dim: _nearest_index(field[depth_dim].values, depth)})
        elif depth_range is not None:
            field = _select_depth_range(field, depth_dim, depth_range)
            field = _apply_aggregation(field, aggregate_dim, dims=[depth_dim])
        else:
            field = field.isel({depth_dim: 0})

    if diagram_type == 'time_lon':
        field = _select_or_aggregate_lat(field, fixed_lat, fixed_lat_range, aggregate_dim)
        if 'lat' in field.dims:
            field = _apply_spatial_aggregation(field, aggregate_dim, dims=['lat'], spatial_weighting=spatial_weighting)
        spatial_dim = 'lon'
    elif diagram_type == 'time_lat':
        field = _select_or_aggregate_lon(field, fixed_lon, fixed_lon_range, aggregate_dim)
        if 'lon' in field.dims:
            field = _apply_spatial_aggregation(field, aggregate_dim, dims=['lon'], spatial_weighting=spatial_weighting)
        spatial_dim = 'lat'
    elif diagram_type == 'time_depth':
        if depth_dim is None:
            raise ValueError("Data must have depth dimension for time_depth Hovmoller")
        if is_dask_backed(field):
            values, time_values, spatial_coord = _compute_time_depth_hovmoller_dask_batches(
                field,
                depth_dim=depth_dim,
                fixed_lon=fixed_lon,
                fixed_lat=fixed_lat,
                fixed_lon_range=fixed_lon_range,
                fixed_lat_range=fixed_lat_range,
                aggregate_dim=aggregate_dim,
                spatial_weighting=spatial_weighting,
                depth_range=depth_range,
            )
            return {
                'time': time_values,
                'spatial_coord': spatial_coord,
                'values': values,
                'metadata': {
                    'diagram_type': diagram_type,
                    'spatial_dim': depth_dim,
                    'aggregate_dim': aggregate_dim,
                    'spatial_weighting': spatial_weighting,
                    'variable': data.name or 'unknown',
                    'units': data.attrs.get('units', ''),
                    'compute_strategy': "time_batch_dask",
                    'statistics': _compute_statistics(values),
                }
            }
        field = _select_time_depth_location(
            field,
            fixed_lon,
            fixed_lat,
            fixed_lon_range,
            fixed_lat_range,
            aggregate_dim,
            spatial_weighting,
        )
        if depth_range is not None:
            field = _select_depth_range(field, depth_dim, depth_range)
        spatial_dim = depth_dim
    else:
        raise ValueError(f"Unknown diagram_type: {diagram_type}")

    if spatial_dim in {'depth', 'z'}:
        field = _drop_sentinel_depth_coordinates(field, spatial_dim)

    if 'time' not in field.dims or spatial_dim not in field.dims:
        raise ValueError("Hovmoller result must retain time and the target spatial dimension")

    ordered = field.transpose('time', spatial_dim)
    if diagram_type == 'time_depth' and is_dask_backed(ordered):
        values = _compute_hovmoller_time_batches(ordered)
        compute_strategy = "time_batch_dask"
    else:
        values = dataarray_to_numpy(
            ordered,
            label="Hovmoller matrix",
            dtype=float,
            start=0.2,
            end=0.9,
        )
        compute_strategy = "single_compute"

    return {
        'time': [str(t) for t in ordered.time.values],
        'spatial_coord': ordered[spatial_dim].values.tolist(),
        'values': values,
        'metadata': {
            'diagram_type': diagram_type,
            'spatial_dim': spatial_dim,
            'aggregate_dim': aggregate_dim,
            'spatial_weighting': spatial_weighting,
            'variable': data.name or 'unknown',
            'units': data.attrs.get('units', ''),
            'compute_strategy': compute_strategy,
            'statistics': _compute_statistics(values),
        }
    }


def _compute_time_depth_hovmoller_dask_batches(
    field: xr.DataArray,
    *,
    depth_dim: str,
    fixed_lon: Optional[float],
    fixed_lat: Optional[float],
    fixed_lon_range: Optional[Tuple[float, float]],
    fixed_lat_range: Optional[Tuple[float, float]],
    aggregate_dim: str,
    spatial_weighting: str,
    depth_range: Optional[Tuple[float, float]],
) -> tuple[np.ndarray, list[str], list[float]]:
    """Build and compute each time-depth Hovmoller batch independently."""
    batches = _hovmoller_time_batch_slices(field)
    total_batches = len(batches)
    report_phase(
        phase="preparing_hovmoller_batches",
        message="Preparing Hovmoller time batches",
        percent=0.03,
        completed_units=0,
        total_units=total_batches,
        unit_label="time batch",
        compute_backend="dask",
        chunks=_hovmoller_chunk_summary(field),
    )

    if total_batches == 0:
        return np.empty((0, 0), dtype=float), [], []

    arrays: list[np.ndarray] = []
    time_values: list[str] = []
    spatial_coord: Optional[list[float]] = None

    for batch_index, batch_slice in enumerate(batches):
        current_unit = _format_time_batch_label(field, batch_slice)
        batch_start = 0.08 + 0.84 * (batch_index / total_batches)
        batch_end = 0.08 + 0.84 * ((batch_index + 1) / total_batches)
        report_phase(
            phase="computing_hovmoller_batch",
            message="Preparing Hovmoller time batch",
            percent=batch_start,
            completed_units=batch_index,
            total_units=total_batches,
            unit_label="time batch",
            current_unit=current_unit,
            compute_backend="dask",
        )
        batch = field.isel(time=batch_slice)
        batch = _select_time_depth_location(
            batch,
            fixed_lon,
            fixed_lat,
            fixed_lon_range,
            fixed_lat_range,
            aggregate_dim,
            spatial_weighting,
        )
        if depth_range is not None:
            batch = _select_depth_range(batch, depth_dim, depth_range)
        batch = _drop_sentinel_depth_coordinates(batch, depth_dim)
        if 'time' not in batch.dims or depth_dim not in batch.dims:
            raise ValueError("Hovmoller result must retain time and the target spatial dimension")

        ordered = batch.transpose('time', depth_dim)
        if spatial_coord is None:
            spatial_coord = ordered[depth_dim].values.tolist()
        batch_values = dataarray_to_numpy(
            ordered,
            label=f"Hovmoller matrix batch {batch_index + 1}/{total_batches}",
            dtype=float,
            start=batch_start,
            end=batch_end,
        )
        arrays.append(batch_values)
        time_values.extend(str(t) for t in ordered.time.values)
        report_phase(
            phase="computing_hovmoller_batch",
            message="Computed Hovmoller time batch",
            percent=batch_end,
            completed_units=batch_index + 1,
            total_units=total_batches,
            unit_label="time batch",
            current_unit=current_unit,
            compute_backend="dask",
        )

    report_phase(
        phase="assembling_hovmoller_matrix",
        message="Assembling Hovmoller matrix",
        percent=0.96,
        completed_units=total_batches,
        total_units=total_batches,
        unit_label="time batch",
        compute_backend="dask",
    )
    values = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 0), dtype=float)
    report_phase(
        phase="assembling_hovmoller_matrix",
        message="Assembled Hovmoller matrix",
        percent=0.98,
        completed_units=total_batches,
        total_units=total_batches,
        unit_label="time batch",
        compute_backend="dask",
    )
    return values, time_values, spatial_coord or []


def _compute_hovmoller_time_batches(ordered: xr.DataArray) -> np.ndarray:
    """Compute a time-depth Hovmoller matrix in bounded time batches."""
    batches = _hovmoller_time_batch_slices(ordered)
    total_batches = len(batches)
    report_phase(
        phase="preparing_hovmoller_batches",
        message="Preparing Hovmoller time batches",
        percent=0.05,
        completed_units=0,
        total_units=total_batches,
        unit_label="time batch",
        compute_backend="dask",
        chunks=_hovmoller_chunk_summary(ordered),
    )

    if total_batches == 0:
        return np.empty((0, int(ordered.sizes.get(ordered.dims[1], 0))), dtype=float)

    arrays = []
    for batch_index, batch_slice in enumerate(batches):
        current_unit = _format_time_batch_label(ordered, batch_slice)
        batch_start = 0.1 + 0.8 * (batch_index / total_batches)
        batch_end = 0.1 + 0.8 * ((batch_index + 1) / total_batches)
        report_phase(
            phase="computing_hovmoller_batch",
            message="Computing Hovmoller time batch",
            percent=batch_start,
            completed_units=batch_index,
            total_units=total_batches,
            unit_label="time batch",
            current_unit=current_unit,
            compute_backend="dask",
        )
        batch = ordered.isel(time=batch_slice)
        batch_values = dataarray_to_numpy(
            batch,
            label=f"Hovmoller matrix batch {batch_index + 1}/{total_batches}",
            dtype=float,
            start=batch_start,
            end=batch_end,
        )
        arrays.append(batch_values)
        report_phase(
            phase="computing_hovmoller_batch",
            message="Computed Hovmoller time batch",
            percent=batch_end,
            completed_units=batch_index + 1,
            total_units=total_batches,
            unit_label="time batch",
            current_unit=current_unit,
            compute_backend="dask",
        )

    report_phase(
        phase="assembling_hovmoller_matrix",
        message="Assembling Hovmoller matrix",
        percent=0.95,
        completed_units=total_batches,
        total_units=total_batches,
        unit_label="time batch",
        compute_backend="dask",
    )
    values = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 0), dtype=float)
    report_phase(
        phase="assembling_hovmoller_matrix",
        message="Assembled Hovmoller matrix",
        percent=0.98,
        completed_units=total_batches,
        total_units=total_batches,
        unit_label="time batch",
        compute_backend="dask",
    )
    return values


def _hovmoller_time_batch_slices(ordered: xr.DataArray) -> list[slice]:
    n_time = int(ordered.sizes.get("time", 0))
    if n_time <= 0:
        return []

    chunks = getattr(ordered.data, "chunks", None)
    if chunks is not None and "time" in ordered.dims:
        time_axis = ordered.get_axis_num("time")
        if time_axis < len(chunks):
            time_chunks = [int(value) for value in chunks[time_axis] if int(value) > 0]
            if time_chunks and sum(time_chunks) == n_time:
                slices = []
                start = 0
                for chunk_size in time_chunks:
                    stop = start + chunk_size
                    slices.append(slice(start, stop))
                    start = stop
                return slices

    batch_size = _hovmoller_fallback_time_batch_size()
    return [slice(start, min(start + batch_size, n_time)) for start in range(0, n_time, batch_size)]


def _hovmoller_fallback_time_batch_size() -> int:
    raw_value = os.environ.get("OCEAN_HOVMOLLER_TIME_BATCH_DAYS", "30")
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = 30
    return max(1, value)


def _format_time_batch_label(ordered: xr.DataArray, batch_slice: slice) -> str:
    start = int(batch_slice.start or 0)
    stop = int(batch_slice.stop or start)
    if stop <= start:
        return "empty"
    time_values = ordered["time"].values
    first = str(time_values[start])
    last = str(time_values[stop - 1])
    return first if first == last else f"{first} to {last}"


def _hovmoller_chunk_summary(ordered: xr.DataArray) -> Dict:
    chunks = getattr(ordered.data, "chunks", None)
    if chunks is None:
        return {}
    return {
        str(dim): [int(value) for value in dim_chunks]
        for dim, dim_chunks in zip(ordered.dims, chunks)
    }


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
        field = _select_depth_range(field, depth_dim, depth_range)

    if depth_aggregation == 'surface':
        return field.isel({depth_dim: 0})
    if depth_aggregation == 'integral':
        return compute_vertical_integral(field, depth_range or _coord_extent(field[depth_dim].values))
    if depth_aggregation in {'mean', 'max', 'min'}:
        return _apply_aggregation(field, depth_aggregation, dims=[depth_dim])

    raise ValueError(f"Unknown depth_aggregation: {depth_aggregation}")


def _aggregate_time(data: xr.DataArray, time_aggregation: str) -> xr.DataArray:
    if 'time' not in data.dims:
        return data
    if time_aggregation in {'mean', 'max', 'min', 'std', 'median'}:
        return _apply_aggregation(data, time_aggregation, dims=['time'])
    raise ValueError(f"Unknown time_aggregation: {time_aggregation}")


def _apply_aggregation(data: xr.DataArray, aggregation: str, dims: list[str]) -> xr.DataArray:
    aggregation = _normalize_aggregate_dim(aggregation)
    if aggregation == 'mean':
        return data.mean(dim=dims, skipna=True)
    if aggregation == 'max':
        return data.max(dim=dims, skipna=True)
    if aggregation == 'min':
        return data.min(dim=dims, skipna=True)
    if aggregation == 'std':
        return data.std(dim=dims, skipna=True)
    if aggregation == 'median':
        return data.median(dim=dims, skipna=True)
    raise ValueError(f"Unknown aggregation: {aggregation}")


def _apply_spatial_aggregation(
    data: xr.DataArray,
    aggregation: str,
    dims: list[str],
    *,
    spatial_weighting: str,
) -> xr.DataArray:
    aggregation = _normalize_aggregate_dim(aggregation)
    if aggregation == 'mean' and spatial_weighting == 'area_weighted':
        spatial_dims = [dim for dim in dims if dim in {'lat', 'lon'} and dim in data.dims]
        if spatial_dims:
            return _area_weighted_mean(data, spatial_dims)
    return _apply_aggregation(data, aggregation, dims)


def _area_weighted_mean(data: xr.DataArray, dims: list[str]) -> xr.DataArray:
    weights = _horizontal_area_weights(data)
    aligned_data, aligned_weights = xr.broadcast(data, weights)
    valid = np.isfinite(aligned_data)
    weighted = xr.where(valid, aligned_data, 0.0) * aligned_weights
    effective_weights = xr.where(valid, aligned_weights, 0.0)
    numerator = weighted.sum(dim=dims, skipna=True)
    denominator = effective_weights.sum(dim=dims, skipna=True)
    return xr.where(denominator > 0.0, numerator / denominator, np.nan)


def _horizontal_area_weights(data: xr.DataArray) -> xr.DataArray:
    if 'lat' not in data.coords:
        raise ValueError("area_weighted Hovmoller aggregation requires lat coordinates")

    lat = np.asarray(data.lat.values, dtype=float)
    lat_bounds = np.deg2rad(_coord_bounds(lat))
    lat_strip = np.abs(np.sin(lat_bounds[1:]) - np.sin(lat_bounds[:-1]))

    if 'lon' in data.coords and 'lon' in data.dims:
        lon = np.asarray(data.lon.values, dtype=float)
        lon_bounds = np.deg2rad(_coord_bounds(lon))
        lon_width = np.abs(np.diff(lon_bounds))
        weights = lat_strip[:, None] * lon_width[None, :]
        return xr.DataArray(weights, coords={'lat': data.lat, 'lon': data.lon}, dims=('lat', 'lon'))

    return xr.DataArray(lat_strip, coords={'lat': data.lat}, dims=('lat',))


def _coord_bounds(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        spacing = 1.0
        return np.asarray([values[0] - spacing / 2.0, values[0] + spacing / 2.0], dtype=float)

    mids = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _select_or_aggregate_lat(
    data: xr.DataArray,
    fixed_lat: Optional[float],
    fixed_lat_range: Optional[Tuple[float, float]],
    aggregate_dim: str,
) -> xr.DataArray:
    fixed_lat = _normalize_optional_scalar(fixed_lat)
    fixed_lat_range = _normalize_optional_range(fixed_lat_range)
    if fixed_lat is not None and fixed_lat_range is not None:
        raise ValueError("fixed_lat and fixed_lat_range are mutually exclusive")
    if fixed_lat is not None:
        return data.sel(lat=fixed_lat, method='nearest')
    if fixed_lat_range is not None:
        selected = data.sel(lat=_build_coord_slice(data.lat.values, fixed_lat_range))
        return _apply_aggregation(selected, aggregate_dim, dims=['lat'])
    raise ValueError("time_lon requires fixed_lat or fixed_lat_range")


def _select_or_aggregate_lon(
    data: xr.DataArray,
    fixed_lon: Optional[float],
    fixed_lon_range: Optional[Tuple[float, float]],
    aggregate_dim: str,
) -> xr.DataArray:
    fixed_lon = _normalize_optional_scalar(fixed_lon)
    fixed_lon_range = _normalize_optional_range(fixed_lon_range)
    if fixed_lon is not None and fixed_lon_range is not None:
        raise ValueError("fixed_lon and fixed_lon_range are mutually exclusive")
    if fixed_lon is not None:
        return data.sel(lon=fixed_lon, method='nearest')
    if fixed_lon_range is not None:
        selected = data.sel(lon=_build_coord_slice(data.lon.values, fixed_lon_range))
        return _apply_aggregation(selected, aggregate_dim, dims=['lon'])
    raise ValueError("time_lat requires fixed_lon or fixed_lon_range")


def _select_time_depth_location(
    data: xr.DataArray,
    fixed_lon: Optional[float],
    fixed_lat: Optional[float],
    fixed_lon_range: Optional[Tuple[float, float]],
    fixed_lat_range: Optional[Tuple[float, float]],
    aggregate_dim: str,
    spatial_weighting: str,
) -> xr.DataArray:
    fixed_lon = _normalize_optional_scalar(fixed_lon)
    fixed_lat = _normalize_optional_scalar(fixed_lat)
    fixed_lon_range = _normalize_optional_range(fixed_lon_range)
    fixed_lat_range = _normalize_optional_range(fixed_lat_range)
    aggregate_dim = _normalize_aggregate_dim(aggregate_dim)
    is_point = fixed_lon is not None and fixed_lat is not None
    is_region = fixed_lon_range is not None and fixed_lat_range is not None
    if is_point == is_region:
        raise ValueError("time_depth requires either fixed_lon/fixed_lat or fixed_lon_range/fixed_lat_range")

    if is_point:
        return data.sel(lon=fixed_lon, lat=fixed_lat, method='nearest')

    selected = data.sel(
        lon=_build_coord_slice(data.lon.values, fixed_lon_range),
        lat=_build_coord_slice(data.lat.values, fixed_lat_range),
    )
    return _apply_spatial_aggregation(
        selected,
        aggregate_dim,
        dims=['lat', 'lon'],
        spatial_weighting=spatial_weighting,
    )


def _drop_sentinel_depth_coordinates(data: xr.DataArray, depth_dim: str) -> xr.DataArray:
    depth_values = np.asarray(data[depth_dim].values, dtype=float)
    valid_indices = np.where(np.isfinite(depth_values) & (np.abs(depth_values) < 9000))[0]
    if valid_indices.size == depth_values.size:
        return data
    return data.isel({depth_dim: valid_indices})


def _select_depth_range(
    data: xr.DataArray,
    depth_dim: str,
    depth_range: Tuple[float, float],
) -> xr.DataArray:
    from domain.ocean.data_access.load import _normalize_depth_range

    depth_values = np.asarray(data[depth_dim].values, dtype=float)
    normalized_range = _normalize_depth_range(depth_values, depth_range)
    return data.sel({depth_dim: _build_coord_slice(depth_values, normalized_range)})


def _build_coord_slice(values, target_range: Tuple[float, float]) -> slice:
    start, end = target_range
    values = np.asarray(values, dtype=float)
    ascending = len(values) < 2 or values[0] <= values[-1]
    if ascending:
        return slice(min(start, end), max(start, end))
    return slice(max(start, end), min(start, end))


def _coord_extent(values) -> Tuple[float, float]:
    return float(np.min(values)), float(np.max(values))


def _nearest_index(values, target: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(values, dtype=float) - target)))


def _normalize_aggregate_dim(aggregation: Optional[str]) -> str:
    if aggregation is None:
        return 'mean'

    normalized = str(aggregation).strip().lower()
    if normalized in {'mean', 'max', 'min', 'std', 'median'}:
        return normalized

    # Planner sometimes emits dimension labels instead of a reduction op.
    if normalized in {'lon_lat', 'lat_lon', 'xy', 'spatial', 'area', 'region'}:
        return 'mean'

    return normalized


def _normalize_spatial_weighting(value: Optional[str]) -> str:
    normalized = str(value or 'equal').strip().lower().replace('-', '_')
    compact = normalized.replace('_', '')
    if compact in {'equal', 'unweighted', 'grid', 'gridpoint', 'gridcell'}:
        return 'equal'
    if compact in {'area', 'areaweighted', 'spherical'}:
        return 'area_weighted'
    raise ValueError(f"Unknown spatial_weighting: {value}")


def _normalize_optional_scalar(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        return float(value[0])
    return float(value)


def _normalize_optional_range(value: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) == 1:
            scalar = float(value[0])
            return (scalar, scalar)
        return (float(value[0]), float(value[1]))
    scalar = float(value)
    return (scalar, scalar)


def _compute_statistics(values: np.ndarray) -> Dict:
    values = as_numeric_array(values)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'n_valid': 0,
        }

    return {
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'min': float(np.min(valid)),
        'max': float(np.max(valid)),
        'n_valid': int(valid.size),
    }
