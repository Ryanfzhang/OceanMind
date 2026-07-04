"""Shared Dask/progress helpers for event detection tools."""

from __future__ import annotations

from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import xarray as xr

from domain.ocean.dask_utils import chunk_summary, dataarray_to_numpy, is_dask_backed, report_phase


EARTH_RADIUS_KM = 6371.0


def report_detection_input(label: str, data: xr.DataArray, *, percent: float) -> None:
    report_phase(
        phase="preparing_event_detection",
        message=f"Preparing {label} event detection input",
        percent=percent,
        compute_backend="dask" if is_dask_backed(data) else "xarray",
        chunks=chunk_summary(data) if is_dask_backed(data) else None,
    )


def compute_event_array(
    data: xr.DataArray,
    *,
    label: str,
    dtype: Any = float,
    start: float,
    end: float,
) -> np.ndarray:
    return dataarray_to_numpy(data, label=label, dtype=dtype, start=start, end=end)


def ensure_time_reduction_chunks(data: xr.DataArray, *, label: str) -> xr.DataArray:
    """Rechunk time to one block before xarray quantile-style reductions if needed."""
    chunks = getattr(data.data, "chunks", None)
    if chunks is None or "time" not in data.dims:
        return data
    time_axis = data.get_axis_num("time")
    time_chunks = chunks[time_axis]
    if len(time_chunks) <= 1:
        return data
    report_phase(
        phase="preparing_event_chunks",
        message=f"Rechunking time for {label} event threshold",
        percent=0.09,
        compute_backend="dask",
        chunks=chunk_summary(data),
    )
    return data.chunk({"time": -1})


def compute_duration_mask(
    mask: np.ndarray,
    min_duration: int,
    *,
    event_label: str,
    start: float = 0.45,
    end: float = 0.55,
) -> np.ndarray:
    """Vectorized persistence mask for boolean arrays with shape time x lat x lon."""
    mask = np.asarray(mask, dtype=bool)
    if min_duration <= 1:
        return mask

    n_time, n_lat, n_lon = mask.shape
    flat_mask = mask.reshape(n_time, n_lat * n_lon)
    flat_persistence = np.zeros_like(flat_mask, dtype=bool)
    consecutive = np.zeros(flat_mask.shape[1], dtype=np.int32)
    report_every = max(1, n_time // 20)

    for t in range(n_time):
        active = flat_mask[t]
        consecutive = np.where(active, consecutive + 1, 0)
        persistent = consecutive >= min_duration
        flat_persistence[t, persistent] = True
        just_reached = consecutive == min_duration
        if np.any(just_reached):
            first_t = max(0, t - min_duration + 1)
            for previous_t in range(first_t, t):
                flat_persistence[previous_t, just_reached] = True
        if t == 0 or t == n_time - 1 or (t + 1) % report_every == 0:
            report_phase(
                phase="computing_event_persistence",
                message=f"Computing {event_label} persistence mask",
                percent=start + (end - start) * ((t + 1) / max(1, n_time)),
                completed_units=t + 1,
                total_units=n_time,
                unit_label="time step",
            )

    return flat_persistence.reshape(mask.shape)


def extract_bbox_from_mask(mask: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> dict[str, Optional[float]]:
    y_indices, x_indices = np.where(mask)
    return extract_bbox_from_indices(y_indices, x_indices, lon, lat)


def extract_bbox_from_indices(
    y_indices: np.ndarray,
    x_indices: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> dict[str, Optional[float]]:
    y_indices = np.asarray(y_indices)
    x_indices = np.asarray(x_indices)
    if y_indices.size == 0 or x_indices.size == 0:
        return {"lon_min": None, "lon_max": None, "lat_min": None, "lat_max": None}

    lon_values = _coordinate_values_at_indices(lon, y_indices, x_indices, axis="x")
    lat_values = _coordinate_values_at_indices(lat, y_indices, x_indices, axis="y")
    return {
        "lon_min": float(np.min(lon_values)),
        "lon_max": float(np.max(lon_values)),
        "lat_min": float(np.min(lat_values)),
        "lat_max": float(np.max(lat_values)),
    }


def estimate_area_km2(mask: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return 0.0

    dlon, dlat = _mean_grid_delta_degrees(lon, lat)
    if dlon <= 0.0 or dlat <= 0.0:
        return 0.0

    y_indices, x_indices = np.where(mask)
    lat_values = _coordinate_values_at_indices(lat, y_indices, x_indices, axis="y")
    mean_lat = float(np.mean(lat_values)) if lat_values.size else float(np.mean(lat))
    dx = EARTH_RADIUS_KM * np.deg2rad(dlon) * np.cos(np.deg2rad(mean_lat))
    dy = EARTH_RADIUS_KM * np.deg2rad(dlat)
    return float(mask.sum() * abs(dx) * abs(dy))


def estimate_grid_spacing_km(lon: np.ndarray, lat: np.ndarray) -> float:
    dlon, dlat = _mean_grid_delta_degrees(lon, lat)
    if dlon <= 0.0 or dlat <= 0.0:
        return 0.0
    mean_lat = float(np.mean(lat))
    dx = EARTH_RADIUS_KM * np.deg2rad(dlon) * np.cos(np.deg2rad(mean_lat))
    dy = EARTH_RADIUS_KM * np.deg2rad(dlat)
    return float((abs(dx) + abs(dy)) / 2.0)


def iter_time_slices(
    u: xr.DataArray,
    v: xr.DataArray,
) -> Iterator[Tuple[Optional[int], xr.DataArray, xr.DataArray]]:
    if "time" in u.dims:
        for index in range(u.sizes["time"]):
            yield index, u.isel(time=index), v.isel(time=index)
    else:
        yield None, u, v


def time_batch_slices(data: xr.DataArray, *, fallback_batch_size: int = 30) -> List[slice]:
    n_time = int(data.sizes.get("time", 0))
    if n_time <= 0:
        return []
    chunks = getattr(data.data, "chunks", None)
    if chunks is not None and "time" in data.dims:
        time_axis = data.get_axis_num("time")
        slices: List[slice] = []
        start = 0
        for chunk_size in chunks[time_axis]:
            stop = min(n_time, start + max(1, int(chunk_size)))
            slices.append(slice(start, stop))
            start = stop
        if slices:
            return slices

    return [
        slice(start, min(n_time, start + fallback_batch_size))
        for start in range(0, n_time, fallback_batch_size)
    ]


def _coordinate_values_at_indices(
    coord: np.ndarray,
    y_indices: np.ndarray,
    x_indices: np.ndarray,
    *,
    axis: str,
) -> np.ndarray:
    coord = np.asarray(coord)
    if coord.ndim >= 2:
        return coord[y_indices, x_indices]
    return coord[x_indices] if axis == "x" else coord[y_indices]


def _mean_grid_delta_degrees(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.ndim >= 2:
        lon_delta = np.diff(lon, axis=1)
    else:
        lon_delta = np.diff(lon)
    if lat.ndim >= 2:
        lat_delta = np.diff(lat, axis=0)
    else:
        lat_delta = np.diff(lat)
    if lon_delta.size == 0 or lat_delta.size == 0:
        return 0.0, 0.0
    return float(abs(np.nanmean(lon_delta))), float(abs(np.nanmean(lat_delta)))
