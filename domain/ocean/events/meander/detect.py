"""
Meander detection from curvature-like flow signatures.
"""

from typing import Dict, List, Optional

import numpy as np
import xarray as xr
from scipy.ndimage import label

from domain.ocean.data_access.partitioned import materialize_partitioned_xarray
from domain.ocean.dask_utils import compute_together_with_progress, report_phase
from domain.ocean.diagnostics.advanced import compute_rossby_number
from domain.ocean.events.detection_utils import (
    compute_event_array,
    estimate_grid_spacing_km,
    iter_time_slices,
    report_detection_input,
)


def detect_meanders(
    u: xr.DataArray,
    v: xr.DataArray,
    curvature_threshold: Optional[float] = None,
    percentile_threshold: float = 90,
    min_length_km: float = 80.0,
    min_pixels: int = 10
) -> Dict:
    """
    Detect meander-like high-curvature current features.
    """
    u = materialize_partitioned_xarray(u)
    v = materialize_partitioned_xarray(v)
    report_detection_input("meander", u, percent=0.02)

    u, v = xr.align(u, v, join='inner')
    if 'depth' in u.dims:
        u = u.isel(depth=0)
    if 'depth' in v.dims:
        v = v.isel(depth=0)
    u, v = compute_together_with_progress(
        (u, v),
        label="meander velocity fields",
        start=0.1,
        end=0.4,
    )

    events: List[Dict] = []
    mask_stack = []

    for time_index, u_slice, v_slice in iter_time_slices(u, v):
        report_phase(phase="extracting_event_properties", message="Detecting meander components", percent=0.45)
        speed = np.sqrt(u_slice.values ** 2 + v_slice.values ** 2)
        rossby = compute_rossby_number(u_slice, v_slice)
        rossby_values = compute_event_array(
            rossby,
            label="meander Rossby number field",
            dtype=float,
            start=0.45,
            end=0.65,
        )
        curvature = np.abs(rossby_values) / np.maximum(speed, 1e-6)
        valid = curvature[~np.isnan(curvature)]
        threshold = curvature_threshold
        if threshold is None:
            threshold = float(np.percentile(valid, percentile_threshold)) if valid.size else 0.0

        meander_mask = curvature >= threshold
        labels, n_labels = label(meander_mask)
        grid_spacing_km = estimate_grid_spacing_km(u_slice.lon.values, u_slice.lat.values)
        mask_stack.append(meander_mask.astype(int))
        events.extend(
            _extract_meanders(
                labels,
                n_labels,
                curvature,
                speed,
                u_slice.lon.values,
                u_slice.lat.values,
                min_pixels,
                min_length_km,
                grid_spacing_km,
                time_index,
            )
        )

    stats = {
        'total_count': len(events),
        'detection_params': {
            'curvature_threshold': curvature_threshold,
            'percentile_threshold': percentile_threshold,
            'min_length_km': min_length_km,
        },
    }
    if events:
        stats['mean_length_km'] = float(np.mean([event['length_km'] for event in events]))
        stats['mean_max_curvature'] = float(np.mean([event['max_curvature'] for event in events]))

    return {
        'event_type': 'meander',
        'events': events,
        'statistics': stats,
        'curvature_mask': np.stack(mask_stack).tolist() if len(mask_stack) > 1 else mask_stack[0].tolist(),
        'coordinates': {
            'lon': u.lon.values.tolist(),
            'lat': u.lat.values.tolist(),
        },
    }


def _extract_meanders(
    labels: np.ndarray,
    n_labels: int,
    curvature: np.ndarray,
    speed: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    min_pixels: int,
    min_length_km: float,
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
        length_km = float(max(np.ptp(x_indices), np.ptp(y_indices), 1) * grid_spacing_km)
        if length_km < min_length_km:
            continue

        values = curvature[mask]
        max_idx = int(np.argmax(values))
        center_lat_idx = int(y_indices[max_idx])
        center_lon_idx = int(x_indices[max_idx])
        amplitude_km = float(np.minimum(np.ptp(x_indices), np.ptp(y_indices)) * grid_spacing_km / 2.0)

        event = {
            'meander_id': f'meander_{len(events) + 1}',
            'center': {
                'lon': float(lon[center_lon_idx]),
                'lat': float(lat[center_lat_idx]),
            },
            'length_km': length_km,
            'amplitude_km': amplitude_km,
            'mean_curvature': float(np.mean(values)),
            'max_curvature': float(np.max(values)),
            'mean_speed': float(np.mean(speed[mask])),
            'n_pixels': n_pixels,
        }
        if time_index is not None:
            event['time_index'] = int(time_index)
        events.append(event)

    return events

