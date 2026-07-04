"""
Shared transect helpers for section extraction and transport diagnostics.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import xarray as xr


EARTH_RADIUS_M = 6_371_000.0


def _normalize_transect_points(transect_points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    if len(transect_points) < 2:
        raise ValueError("transect_points must contain at least two [lon, lat] points")

    normalized: list[tuple[float, float]] = []
    for point in transect_points:
        if len(point) != 2:
            raise ValueError("Each transect point must contain exactly two values: [lon, lat]")
        lon, lat = float(point[0]), float(point[1])
        normalized.append((lon, lat))

    return normalized


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_rad, lat1_rad = np.deg2rad([lon1, lat1])
    lon2_rad, lat2_rad = np.deg2rad([lon2, lat2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * (EARTH_RADIUS_M / 1000.0) * np.arcsin(np.sqrt(a)))


def _build_transect_samples(
    transect_points: Sequence[Sequence[float]],
    n_samples: int,
) -> dict:
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    points = _normalize_transect_points(transect_points)
    segment_lengths = [
        _haversine_km(lon1, lat1, lon2, lat2)
        for (lon1, lat1), (lon2, lat2) in zip(points[:-1], points[1:])
    ]
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_distance = float(cumulative[-1])

    if total_distance <= 0.0:
        raise ValueError("transect_points must span a non-zero distance")

    sample_distances = np.linspace(0.0, total_distance, n_samples)
    sample_lons = np.empty(n_samples, dtype=float)
    sample_lats = np.empty(n_samples, dtype=float)

    for index, distance in enumerate(sample_distances):
        segment_index = int(np.searchsorted(cumulative, distance, side="right") - 1)
        segment_index = min(max(segment_index, 0), len(points) - 2)
        start_distance = cumulative[segment_index]
        segment_length = segment_lengths[segment_index]
        frac = 0.0 if segment_length == 0.0 else (distance - start_distance) / segment_length

        (lon1, lat1) = points[segment_index]
        (lon2, lat2) = points[segment_index + 1]
        sample_lons[index] = lon1 + frac * (lon2 - lon1)
        sample_lats[index] = lat1 + frac * (lat2 - lat1)

    return {
        "transect_points": [[float(lon), float(lat)] for lon, lat in points],
        "distance_km": sample_distances,
        "lon": sample_lons,
        "lat": sample_lats,
        "sample_points": [
            {
                "lon": float(lon),
                "lat": float(lat),
                "distance_km": float(distance),
            }
            for lon, lat, distance in zip(sample_lons, sample_lats, sample_distances)
        ],
    }


def _interp_along_transect(
    data: xr.DataArray,
    transect_points: Sequence[Sequence[float]],
    n_samples: int = 200,
    method: str = "linear",
) -> tuple[xr.DataArray, dict]:
    if "lon" not in data.coords or "lat" not in data.coords:
        raise ValueError("Transect tools require lon and lat coordinates")

    sample_info = _build_transect_samples(transect_points, n_samples)
    distance_coord = xr.DataArray(sample_info["distance_km"], dims="distance")
    sampled = data.interp(
        lon=xr.DataArray(sample_info["lon"], dims="distance", coords={"distance": distance_coord}),
        lat=xr.DataArray(sample_info["lat"], dims="distance", coords={"distance": distance_coord}),
        method=method,
    )
    sampled = sampled.assign_coords(distance=sample_info["distance_km"])
    sampled.coords["distance"].attrs["units"] = "km"
    return sampled, sample_info


def _compute_left_normal_components(lons: Iterable[float], lats: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    lons_arr = np.asarray(list(lons), dtype=float)
    lats_arr = np.asarray(list(lats), dtype=float)
    if lons_arr.size < 2:
        raise ValueError("At least two sampled transect points are required")

    mean_lat_rad = np.deg2rad(np.nanmean(lats_arr))
    x = EARTH_RADIUS_M * np.deg2rad(lons_arr) * np.cos(mean_lat_rad)
    y = EARTH_RADIUS_M * np.deg2rad(lats_arr)

    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.hypot(dx, dy)
    norm[norm == 0.0] = np.nan

    nx = -dy / norm
    ny = dx / norm
    return nx, ny


def _distance_axis_m(distance_km: Sequence[float]) -> np.ndarray:
    distance_axis = np.asarray(distance_km, dtype=float) * 1000.0
    if distance_axis.size < 2:
        raise ValueError("At least two transect distance points are required")
    return distance_axis

