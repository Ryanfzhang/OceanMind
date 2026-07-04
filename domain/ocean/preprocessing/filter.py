"""
Preprocessing helpers for temporal/spatial filtering, masking, and interpolation.
"""

import re
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import butter, filtfilt

from domain.ocean.data_access.partitioned import (
    find_partitioned_values,
    materialize_partitioned_xarray,
)
from domain.ocean.data_access.load import get_depth_dim
from domain.ocean.dask_utils import dataarray_to_numpy, is_dask_backed, report_phase


def filter_data(
    data: xr.DataArray,
    filter_type: Literal['lowpass', 'highpass', 'bandpass'],
    cutoff_period: Union[float, Tuple[float, float]],
    dimension: Literal['time', 'spatial'] = 'time',
    method: Literal['butterworth', 'gaussian', 'running_mean'] = 'butterworth',
    order: int = 4
) -> xr.DataArray:
    """
    Filter a DataArray along time or horizontal space.

    Args:
        data: Input field.
        filter_type: Filter family.
        cutoff_period: Window or period in samples. For bandpass provide
            `(low_period, high_period)`.
        dimension: Time or spatial filtering mode.
        method: Numerical implementation.
        order: Butterworth order.

    Returns:
        Filtered DataArray with the same dims and coordinates.
    """
    if find_partitioned_values((data,)):
        if _normalize_filter_dimension(dimension) == "time":
            data = materialize_partitioned_xarray(data)
        else:
            from packages.tool_loader.partitioned_execution import execute_partition_aware

            return execute_partition_aware(
                tool_name="filter_data",
                tool_func=filter_data,
                params={
                    "data": data,
                    "filter_type": filter_type,
                    "cutoff_period": cutoff_period,
                    "dimension": dimension,
                    "method": method,
                    "order": order,
                },
            )
    else:
        data = materialize_partitioned_xarray(data)

    filter_type, method = _normalize_filter_spec(filter_type, method)
    dimension = _normalize_filter_dimension(dimension)
    cutoff_period = _normalize_cutoff_period_arg(cutoff_period)

    if dimension == 'time':
        if 'time' not in data.dims:
            raise ValueError("filter_data with dimension='time' requires a time dimension")
        axis = data.get_axis_num('time')
        return _filter_time(data, axis, filter_type, cutoff_period, method, order)

    return _filter_spatial(data, filter_type, cutoff_period, method, order)


def interpolate_data(
    data: xr.DataArray,
    lon_points: Optional[np.ndarray] = None,
    lat_points: Optional[np.ndarray] = None,
    depth_points: Optional[np.ndarray] = None,
    time_points: Optional[np.ndarray] = None,
    method: Literal['linear', 'nearest'] = 'linear'
) -> xr.DataArray:
    """
    Interpolate a DataArray to a new coordinate grid.

    At least one target coordinate sequence must be provided.
    """
    data = materialize_partitioned_xarray(data)

    coords = {}
    if lon_points is not None:
        coords['lon'] = lon_points
    if lat_points is not None:
        coords['lat'] = lat_points
    if depth_points is not None:
        depth_dim = get_depth_dim(data)
        if depth_dim is None:
            raise ValueError("depth_points provided but input has no depth dimension")
        coords[depth_dim] = depth_points
    if time_points is not None:
        coords['time'] = time_points

    if not coords:
        raise ValueError("interpolate_data requires at least one target coordinate array")

    return data.interp(coords, method=method)


def apply_mask(
    data: xr.DataArray,
    mask: xr.DataArray,
    fill_value: float = np.nan
) -> xr.DataArray:
    """Apply a boolean mask to the data field."""
    if find_partitioned_values((data, mask)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="apply_mask",
            tool_func=apply_mask,
            params={"data": data, "mask": mask, "fill_value": fill_value},
        )
    data = materialize_partitioned_xarray(data)
    mask = materialize_partitioned_xarray(mask)

    aligned_data, aligned_mask = xr.align(data, mask, join='inner')
    return aligned_data.where(aligned_mask, fill_value)


def build_polygon_mask(
    data: xr.DataArray,
    polygon_points: Sequence[Sequence[float]],
    invert: bool = False,
) -> xr.DataArray:
    """
    Build a boolean mask from a lon/lat polygon on the grid of `data`.

    Args:
        data: Reference field that provides lon/lat coordinates.
        polygon_points: Polygon vertices as `[[lon, lat], ...]`.
        invert: When true, flip inside/outside.
    """
    data = materialize_partitioned_xarray(data)

    if 'lon' not in data.coords or 'lat' not in data.coords:
        raise ValueError("build_polygon_mask requires lon/lat coordinates")
    if len(polygon_points) < 3:
        raise ValueError("polygon_points must contain at least three vertices")

    polygon = np.asarray(polygon_points, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon_points must be shaped like [[lon, lat], ...]")

    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])

    lon_values = np.asarray(data.lon.values, dtype=float)
    lat_values = np.asarray(data.lat.values, dtype=float)
    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)

    mask = _points_in_polygon(lon_grid, lat_grid, polygon[:, 0], polygon[:, 1])
    if invert:
        mask = ~mask

    mask_field = xr.DataArray(
        mask,
        coords={"lat": data.lat, "lon": data.lon},
        dims=("lat", "lon"),
        name="polygon_mask",
        attrs={
            "long_name": "Polygon mask",
            "mask_type": "polygon",
            "invert": bool(invert),
            "polygon_points": polygon.tolist(),
        },
    )
    return mask_field


def build_isobath_mask(
    data: xr.DataArray,
    isobath_depth: float,
    comparison: Literal['shallower_or_equal', 'deeper_or_equal'] = 'shallower_or_equal',
    bathymetry: Optional[xr.DataArray] = None,
    invert: bool = False,
) -> xr.DataArray:
    """
    Build a horizontal mask relative to an isobath depth.

    When `bathymetry` is absent, infer bottom depth from the deepest valid level of
    the provided 3D/4D field.
    """
    data = materialize_partitioned_xarray(data)
    bathymetry = materialize_partitioned_xarray(bathymetry)

    bottom_depth, bathymetry_source, bathymetry_warning = _resolve_bottom_depth(data, bathymetry)
    threshold = abs(float(isobath_depth))

    if comparison == 'shallower_or_equal':
        mask = bottom_depth <= threshold
    elif comparison == 'deeper_or_equal':
        mask = bottom_depth >= threshold
    else:
        raise ValueError(f"Unsupported comparison: {comparison}")

    if invert:
        mask = ~mask

    attrs = {
        "long_name": "Isobath mask",
        "mask_type": "isobath",
        "isobath_depth": threshold,
        "comparison": comparison,
        "invert": bool(invert),
        "bathymetry_source": bathymetry_source,
    }
    if bathymetry_warning is not None:
        attrs["bathymetry_warning"] = bathymetry_warning

    mask_values = dataarray_to_numpy(
        mask.astype(bool),
        label="isobath mask",
        dtype=bool,
        start=0.2,
        end=0.9,
    )

    return xr.DataArray(
        mask_values.astype(bool),
        coords={"lat": bottom_depth.lat, "lon": bottom_depth.lon},
        dims=("lat", "lon"),
        name="isobath_mask",
        attrs=attrs,
    )


def combine_masks(
    masks: Sequence[xr.DataArray],
    operation: Literal['and', 'or', 'xor'] = 'and',
    invert: bool = False,
) -> xr.DataArray:
    """
    Combine multiple boolean masks on a shared horizontal grid.
    """
    masks = [materialize_partitioned_xarray(mask) for mask in masks]

    if not masks:
        raise ValueError("combine_masks requires at least one mask")

    aligned_masks = [mask.astype(bool) for mask in xr.align(*masks, join='inner')]
    combined = aligned_masks[0]
    for mask in aligned_masks[1:]:
        if operation == 'and':
            combined = combined & mask
        elif operation == 'or':
            combined = combined | mask
        elif operation == 'xor':
            combined = combined ^ mask
        else:
            raise ValueError(f"Unsupported mask operation: {operation}")

    if invert:
        combined = ~combined

    combined.name = "combined_mask"
    combined.attrs = {
        "long_name": "Combined mask",
        "mask_type": "combined",
        "operation": operation,
        "invert": bool(invert),
        "n_masks": len(aligned_masks),
    }
    return combined


def _filter_time(
    data: xr.DataArray,
    axis: int,
    filter_type: str,
    cutoff_period: Union[float, Tuple[float, float]],
    method: str,
    order: int
) -> xr.DataArray:
    """Apply a 1D filter along the time axis."""
    if method == 'running_mean':
        return _running_mean_filter(data, filter_type, cutoff_period)
    if method == 'gaussian':
        return _gaussian_time_filter(data, filter_type, cutoff_period, axis)
    if method == 'butterworth':
        filtered = filtfilt(
            *butter(order, _normalize_cutoff(cutoff_period, filter_type), btype=_butter_type(filter_type)),
            data.values,
            axis=axis,
            method='gust',
        )
        result = data.copy(data=filtered)
        result.attrs = dict(data.attrs)
        result.attrs['filter_method'] = method
        result.attrs['filter_type'] = filter_type
        return result
    raise ValueError(f"Unsupported filter method: {method}")


def _normalize_filter_spec(filter_type: str, method: str) -> Tuple[str, str]:
    normalized_filter_type = _canonical_filter_type(filter_type)
    normalized_method = _canonical_filter_method(method)

    if normalized_filter_type is None:
        combined = _split_combined_filter_token(filter_type)
        if combined is not None:
            combined_filter_type, combined_method = combined
            normalized_filter_type = combined_filter_type
            normalized_method = normalized_method or combined_method

    if normalized_method is None:
        combined = _split_combined_filter_token(method)
        if combined is not None:
            combined_filter_type, combined_method = combined
            normalized_method = combined_method
            normalized_filter_type = normalized_filter_type or combined_filter_type

    if normalized_filter_type is None:
        raise ValueError(f"Unsupported filter type: {filter_type}")
    if normalized_method is None:
        raise ValueError(f"Unsupported filter method: {method}")

    return normalized_filter_type, normalized_method


def _normalize_filter_dimension(dimension: str) -> str:
    key = _compact_token(dimension)
    aliases = {
        'time': 'time',
        'temporal': 'time',
        'timeseries': 'time',
        'spatial': 'spatial',
        'space': 'spatial',
        'horizontal': 'spatial',
    }
    normalized = aliases.get(key)
    if normalized is None:
        raise ValueError(f"Unsupported filter dimension: {dimension}")
    return normalized


def _normalize_cutoff_period_arg(
    cutoff_period: Union[float, Tuple[float, float]]
) -> Union[float, Tuple[float, float]]:
    if isinstance(cutoff_period, str):
        numeric = _coerce_cutoff_value(cutoff_period)
        if numeric is not None:
            return numeric

    if isinstance(cutoff_period, (list, tuple)):
        numeric_values = [
            numeric
            for item in cutoff_period
            for numeric in [_coerce_cutoff_value(item)]
            if numeric is not None
        ]
        if len(numeric_values) == 1:
            return numeric_values[0]
        if len(numeric_values) == 2:
            return (numeric_values[0], numeric_values[1])
        raise ValueError("filtering requires one cutoff value or a pair of cutoff values")
    return cutoff_period


def _canonical_filter_type(value: str) -> Optional[str]:
    return {
        'low': 'lowpass',
        'lowpass': 'lowpass',
        'high': 'highpass',
        'highpass': 'highpass',
        'band': 'bandpass',
        'bandpass': 'bandpass',
    }.get(_compact_token(value))


def _canonical_filter_method(value: str) -> Optional[str]:
    return {
        'butter': 'butterworth',
        'butterworth': 'butterworth',
        'gauss': 'gaussian',
        'gaussian': 'gaussian',
        'conv': 'gaussian',
        'convolution': 'gaussian',
        'runningmean': 'running_mean',
        'movingaverage': 'running_mean',
        'boxcar': 'running_mean',
    }.get(_compact_token(value))


def _split_combined_filter_token(value: str) -> Optional[Tuple[str, str]]:
    compact = _compact_token(value)
    if not compact:
        return None

    method_aliases = {
        'butter': 'butterworth',
        'butterworth': 'butterworth',
        'gauss': 'gaussian',
        'gaussian': 'gaussian',
        'conv': 'gaussian',
        'convolution': 'gaussian',
        'runningmean': 'running_mean',
        'movingaverage': 'running_mean',
        'boxcar': 'running_mean',
    }
    filter_aliases = {
        'low': 'lowpass',
        'lowpass': 'lowpass',
        'high': 'highpass',
        'highpass': 'highpass',
        'band': 'bandpass',
        'bandpass': 'bandpass',
    }

    for method_key, normalized_method in method_aliases.items():
        for filter_key, normalized_filter_type in filter_aliases.items():
            if compact in {f"{method_key}{filter_key}", f"{filter_key}{method_key}"}:
                return normalized_filter_type, normalized_method

    return None


def _compact_token(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r'[^a-z]+', '', value.lower())


def _coerce_cutoff_value(value: Union[str, float, int]) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    match = re.search(r'[-+]?\d*\.?\d+', value)
    if match is None:
        return None
    return float(match.group(0))


def _filter_spatial(
    data: xr.DataArray,
    filter_type: str,
    cutoff_period: Union[float, Tuple[float, float]],
    method: str,
    order: int
) -> xr.DataArray:
    """Apply a horizontal filter over lat/lon."""
    if 'lat' not in data.dims or 'lon' not in data.dims:
        raise ValueError("filter_data with dimension='spatial' requires lat/lon dimensions")

    sigma = _cutoff_to_sigma(cutoff_period)
    lat_axis = data.get_axis_num('lat')
    lon_axis = data.get_axis_num('lon')

    if method in {'gaussian', 'butterworth'}:
        smooth = gaussian_filter(data.values, sigma=_build_spatial_sigma(data.ndim, lat_axis, lon_axis, sigma))
    elif method == 'running_mean':
        smooth = data.rolling(lat=max(int(round(sigma)), 1), lon=max(int(round(sigma)), 1), center=True).mean().values
    else:
        raise ValueError(f"Unsupported filter method: {method}")

    if filter_type == 'lowpass':
        filtered = smooth
    elif filter_type == 'highpass':
        filtered = data.values - smooth
    elif filter_type == 'bandpass':
        if not isinstance(cutoff_period, tuple):
            raise ValueError("bandpass spatial filtering requires a cutoff_period tuple")
        low_sigma = _cutoff_to_sigma(cutoff_period[0])
        high_sigma = _cutoff_to_sigma(cutoff_period[1])
        low_smooth = gaussian_filter(data.values, sigma=_build_spatial_sigma(data.ndim, lat_axis, lon_axis, low_sigma))
        high_smooth = gaussian_filter(data.values, sigma=_build_spatial_sigma(data.ndim, lat_axis, lon_axis, high_sigma))
        filtered = high_smooth - low_smooth
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

    result = data.copy(data=filtered)
    result.attrs = dict(data.attrs)
    result.attrs['filter_method'] = method
    result.attrs['filter_type'] = filter_type
    return result


def _resolve_bottom_depth(
    data: xr.DataArray,
    bathymetry: Optional[xr.DataArray],
) -> Tuple[xr.DataArray, str, Optional[str]]:
    if 'lon' not in data.coords or 'lat' not in data.coords:
        raise ValueError("build_isobath_mask requires lon/lat coordinates")

    bathymetry_warning: Optional[str] = None
    if bathymetry is not None:
        normalized_bathymetry, bathymetry_source = _normalize_bathymetry_field(bathymetry)
        if normalized_bathymetry is not None and bathymetry_source is not None:
            aligned_bathy, _ = xr.align(
                np.abs(normalized_bathymetry),
                _horizontal_reference(data),
                join='inner',
            )
            aligned_bathy = aligned_bathy.astype(float).transpose('lat', 'lon')
            aligned_bathy.name = "bottom_depth"
            return aligned_bathy, bathymetry_source, None
        bathymetry_warning = "ignored_non_2d_bathymetry"

    depth_dim = get_depth_dim(data)
    if depth_dim is None:
        raise ValueError("build_isobath_mask requires either bathymetry or a depth-resolved field")

    inference_data, representative_warning = _representative_bathymetry_inference_field(data, depth_dim)
    bathymetry_warning = _combine_warnings(bathymetry_warning, representative_warning)

    valid = np.isfinite(inference_data)
    reduce_dims = [dim for dim in inference_data.dims if dim not in {depth_dim, 'lat', 'lon'}]
    for dim in reduce_dims:
        valid = valid.any(dim=dim)

    raw_depth_values = np.asarray(inference_data[depth_dim].values, dtype=float)
    valid_depth = xr.DataArray(
        np.isfinite(raw_depth_values) & (np.abs(raw_depth_values) < 9000),
        coords={depth_dim: inference_data[depth_dim]},
        dims=(depth_dim,),
    )

    valid = valid & valid_depth

    depth_values = xr.DataArray(
        np.abs(raw_depth_values),
        coords={depth_dim: inference_data[depth_dim]},
        dims=(depth_dim,),
    )
    depth_grid = depth_values.broadcast_like(valid)
    bottom_depth = depth_grid.where(valid).max(dim=depth_dim, skipna=True).transpose('lat', 'lon')
    bottom_depth.name = "bottom_depth"
    return bottom_depth, "inferred_from_valid_depth", bathymetry_warning


def _representative_bathymetry_inference_field(
    data: xr.DataArray,
    depth_dim: str,
) -> Tuple[xr.DataArray, Optional[str]]:
    """Use one representative non-spatial slice for lazy bathymetry inference."""
    if not is_dask_backed(data):
        return data, None

    indexers = {
        dim: 0
        for dim in data.dims
        if dim not in {depth_dim, 'lat', 'lon'} and int(data.sizes.get(dim, 0)) > 0
    }
    if not indexers:
        return data, None

    report_phase(
        phase="inferring_bathymetry",
        message="Inferring isobath mask from a representative wet-cell slice",
        percent=0.05,
        compute_backend="dask",
    )
    return data.isel(indexers, drop=True), "inferred_from_representative_slice"


def _combine_warnings(first: Optional[str], second: Optional[str]) -> Optional[str]:
    if first and second:
        if first == second:
            return first
        return f"{first};{second}"
    return first or second


def _normalize_bathymetry_field(
    bathymetry: xr.DataArray,
) -> Tuple[Optional[xr.DataArray], Optional[str]]:
    if 'lon' not in bathymetry.coords or 'lat' not in bathymetry.coords:
        raise ValueError("bathymetry must include lon/lat coordinates")

    extra_dims = [dim for dim in bathymetry.dims if dim not in {'lat', 'lon'}]
    normalized = bathymetry
    source = "provided"

    if extra_dims:
        if any(bathymetry.sizes[dim] != 1 for dim in extra_dims):
            return None, None
        normalized = bathymetry.isel({dim: 0 for dim in extra_dims}, drop=True)
        source = "provided_squeezed"

    if normalized.ndim != 2 or set(normalized.dims) != {'lat', 'lon'}:
        return None, None

    return normalized.transpose('lat', 'lon'), source


def _horizontal_reference(data: xr.DataArray) -> xr.DataArray:
    extra_dims = {dim: 0 for dim in data.dims if dim not in {'lat', 'lon'}}
    if not extra_dims:
        return data.transpose('lat', 'lon')
    return data.isel(extra_dims, drop=True).transpose('lat', 'lon')


def _running_mean_filter(
    data: xr.DataArray,
    filter_type: str,
    cutoff_period: Union[float, Tuple[float, float]]
) -> xr.DataArray:
    """Apply running-mean style low/high/band-pass filtering."""
    if filter_type == 'bandpass':
        if not isinstance(cutoff_period, tuple):
            raise ValueError("bandpass filtering requires a cutoff_period tuple")
        short_period, long_period = cutoff_period
        low = data.rolling(time=max(int(round(long_period)), 1), center=True).mean()
        high = data.rolling(time=max(int(round(short_period)), 1), center=True).mean()
        result = high - low
    else:
        period = cutoff_period if not isinstance(cutoff_period, tuple) else cutoff_period[-1]
        smooth = data.rolling(time=max(int(round(float(period))), 1), center=True).mean()
        result = smooth if filter_type == 'lowpass' else (data - smooth)

    result.attrs = dict(data.attrs)
    result.attrs['filter_method'] = 'running_mean'
    result.attrs['filter_type'] = filter_type
    return result


def _gaussian_time_filter(
    data: xr.DataArray,
    filter_type: str,
    cutoff_period: Union[float, Tuple[float, float]],
    axis: int
) -> xr.DataArray:
    """Apply gaussian smoothing along time."""
    sigma = _cutoff_to_sigma(cutoff_period)
    smooth = gaussian_filter1d(data.values, sigma=sigma, axis=axis)

    if filter_type == 'lowpass':
        filtered = smooth
    elif filter_type == 'highpass':
        filtered = data.values - smooth
    elif filter_type == 'bandpass':
        if not isinstance(cutoff_period, tuple):
            raise ValueError("bandpass filtering requires a cutoff_period tuple")
        short_sigma = _cutoff_to_sigma(cutoff_period[0])
        long_sigma = _cutoff_to_sigma(cutoff_period[1])
        short_smooth = gaussian_filter1d(data.values, sigma=short_sigma, axis=axis)
        long_smooth = gaussian_filter1d(data.values, sigma=long_sigma, axis=axis)
        filtered = short_smooth - long_smooth
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

    result = data.copy(data=filtered)
    result.attrs = dict(data.attrs)
    result.attrs['filter_method'] = 'gaussian'
    result.attrs['filter_type'] = filter_type
    return result


def _normalize_cutoff(
    cutoff_period: Union[float, Tuple[float, float]],
    filter_type: str
) -> Union[float, Tuple[float, float]]:
    """Convert sample-period style cutoffs into Butterworth normalized frequencies."""
    if filter_type == 'bandpass':
        if not isinstance(cutoff_period, tuple):
            raise ValueError("bandpass filtering requires a cutoff_period tuple")
        low_period, high_period = sorted(float(value) for value in cutoff_period)
        low = max(min(2.0 / max(high_period, 2.0), 0.99), 1e-4)
        high = max(min(2.0 / max(low_period, 2.0), 0.99), low + 1e-4)
        return (low, high)

    period = float(cutoff_period if not isinstance(cutoff_period, tuple) else cutoff_period[-1])
    return max(min(2.0 / max(period, 2.0), 0.99), 1e-4)


def _butter_type(filter_type: str) -> str:
    return {
        'lowpass': 'lowpass',
        'highpass': 'highpass',
        'bandpass': 'bandpass',
    }[filter_type]


def _cutoff_to_sigma(cutoff_period: Union[float, Tuple[float, float]]) -> float:
    """Map sample periods to a gaussian sigma."""
    if isinstance(cutoff_period, tuple):
        return max(float(max(cutoff_period)) / 2.0, 1.0)
    return max(float(cutoff_period) / 2.0, 1.0)


def _build_spatial_sigma(ndim: int, lat_axis: int, lon_axis: int, sigma: float):
    sigma_values = [0.0] * ndim
    sigma_values[lat_axis] = sigma
    sigma_values[lon_axis] = sigma
    return sigma_values


def _points_in_polygon(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    polygon_lon: np.ndarray,
    polygon_lat: np.ndarray,
) -> np.ndarray:
    """Vectorized ray-casting polygon inclusion test with boundary support."""
    inside = np.zeros(lon_grid.shape, dtype=bool)
    boundary = np.zeros(lon_grid.shape, dtype=bool)
    tolerance = 1e-10

    for index in range(len(polygon_lon) - 1):
        x0 = polygon_lon[index]
        y0 = polygon_lat[index]
        x1 = polygon_lon[index + 1]
        y1 = polygon_lat[index + 1]

        cross = (lon_grid - x0) * (y1 - y0) - (lat_grid - y0) * (x1 - x0)
        dot = (lon_grid - x0) * (x1 - x0) + (lat_grid - y0) * (y1 - y0)
        segment_sq = (x1 - x0) ** 2 + (y1 - y0) ** 2
        boundary |= (
            (np.abs(cross) <= tolerance)
            & (dot >= -tolerance)
            & (dot <= segment_sq + tolerance)
        )

        denominator = (y1 - y0) if abs(y1 - y0) > tolerance else np.nan
        intersects = ((y0 > lat_grid) != (y1 > lat_grid)) & (
            lon_grid < ((x1 - x0) * (lat_grid - y0) / denominator + x0)
        )
        inside ^= np.nan_to_num(intersects, nan=False)

    return inside | boundary
