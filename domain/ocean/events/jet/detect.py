"""
Jet detection based on high-speed elongated flow features.
"""

from typing import Dict, List, Optional

import numpy as np
import xarray as xr
from scipy.ndimage import label

from domain.ocean.data_access.partitioned import materialize_partitioned_xarray
from domain.ocean.dask_utils import compute_together_with_progress, report_phase
from domain.ocean.events.detection_utils import (
    estimate_grid_spacing_km,
    iter_time_slices,
    report_detection_input,
)


def detect_jets(
    u: xr.DataArray,
    v: xr.DataArray,
    speed_threshold: Optional[float] = None,
    percentile_threshold: float = 90,
    min_length_km: float = 100.0,
    min_aspect_ratio: float = 3.0,
    min_pixels: int = 12
) -> Dict:
    """
    Detect jet-like high-speed elongated structures from velocity fields.
    """
    u = materialize_partitioned_xarray(u)
    v = materialize_partitioned_xarray(v)
    report_detection_input("jet", u, percent=0.02)

    u, v = xr.align(u, v, join='inner')
    u, v = _prepare_horizontal_velocity_fields(u, v)
    u, v = compute_together_with_progress(
        (u, v),
        label="jet velocity fields",
        start=0.1,
        end=0.55,
    )
    slices = iter_time_slices(u, v)
    events: List[Dict] = []
    mask_stack = []

    for time_index, u_slice, v_slice in slices:
        report_phase(phase="extracting_event_properties", message="Detecting jet components", percent=0.6)
        speed = np.sqrt(u_slice.values ** 2 + v_slice.values ** 2)
        threshold = speed_threshold
        valid = speed[~np.isnan(speed)]
        if threshold is None:
            threshold = float(np.percentile(valid, percentile_threshold)) if valid.size else 0.0

        jet_mask = speed >= threshold
        labels, n_labels = label(jet_mask)
        grid_spacing_km = estimate_grid_spacing_km(u_slice.lon.values, u_slice.lat.values)

        mask_stack.append(jet_mask.astype(int))
        events.extend(
            _extract_jets(
                labels=labels,
                n_labels=n_labels,
                speed=speed,
                u=u_slice.values,
                v=v_slice.values,
                lon=u_slice.lon.values,
                lat=u_slice.lat.values,
                min_pixels=min_pixels,
                min_length_km=min_length_km,
                min_aspect_ratio=min_aspect_ratio,
                grid_spacing_km=grid_spacing_km,
                time_index=time_index,
            )
        )

    stats = {
        'total_count': len(events),
        'detection_params': {
            'speed_threshold': speed_threshold,
            'percentile_threshold': percentile_threshold,
            'min_length_km': min_length_km,
            'min_aspect_ratio': min_aspect_ratio,
        },
    }
    if events:
        stats['mean_length_km'] = float(np.mean([event['length_km'] for event in events]))
        stats['mean_max_speed'] = float(np.mean([event['max_speed'] for event in events]))

    return {
        'event_type': 'jet',
        'events': events,
        'statistics': stats,
        'jet_mask': np.stack(mask_stack).tolist() if len(mask_stack) > 1 else mask_stack[0].tolist(),
        'coordinates': {
            'lon': u.lon.values.tolist(),
            'lat': u.lat.values.tolist(),
        },
    }


def _prepare_horizontal_velocity_fields(u: xr.DataArray, v: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    if 'depth' in u.dims:
        u = u.isel(depth=0)
    if 'depth' in v.dims:
        v = v.isel(depth=0)
    return u, v


def _extract_jets(
    labels: np.ndarray,
    n_labels: int,
    speed: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    min_pixels: int,
    min_length_km: float,
    min_aspect_ratio: float,
    grid_spacing_km: float,
    time_index: Optional[int],
) -> List[Dict]:
    events: List[Dict] = []
    for label_index in range(1, n_labels + 1):
        mask = labels == label_index
        n_pixels = int(mask.sum())
        if n_pixels < min_pixels:
            continue

        y_indices, x_indices = np.where(mask)
        length_pixels, width_pixels, orientation = _principal_axes(y_indices, x_indices)
        length_km = float(length_pixels * grid_spacing_km)
        width_km = float(max(width_pixels * grid_spacing_km, grid_spacing_km))
        aspect_ratio = float(length_km / max(width_km, 1e-6))
        if length_km < min_length_km or aspect_ratio < min_aspect_ratio:
            continue

        region_speed = speed[mask]
        mean_u = float(np.mean(u[mask]))
        mean_v = float(np.mean(v[mask]))
        max_idx = int(np.argmax(region_speed))
        center_lat_idx = int(y_indices[max_idx])
        center_lon_idx = int(x_indices[max_idx])

        event = {
            'jet_id': f'jet_{len(events) + 1}',
            'center': {
                'lon': float(lon[center_lon_idx]),
                'lat': float(lat[center_lat_idx]),
            },
            'length_km': length_km,
            'width_km': width_km,
            'aspect_ratio': aspect_ratio,
            'orientation_deg': float(orientation),
            'mean_speed': float(np.mean(region_speed)),
            'max_speed': float(np.max(region_speed)),
            'mean_direction_deg': float(np.degrees(np.arctan2(mean_v, mean_u))),
            'n_pixels': n_pixels,
        }
        if time_index is not None:
            event['time_index'] = int(time_index)
        events.append(event)
    return events


def _principal_axes(y_indices: np.ndarray, x_indices: np.ndarray):
    """Estimate major/minor axis lengths and orientation using covariance PCA."""
    coords = np.vstack([x_indices - np.mean(x_indices), y_indices - np.mean(y_indices)])
    cov = np.cov(coords)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    length = 4.0 * np.sqrt(max(eigenvalues[0], 1e-6))
    width = 4.0 * np.sqrt(max(eigenvalues[1], 1e-6))
    orientation = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    return length, width, orientation

