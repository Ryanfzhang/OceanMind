"""
Eutrophication detection from persistent chlorophyll anomalies.
"""

from typing import Dict, List, Optional

import numpy as np
import xarray as xr
from scipy.ndimage import label

from domain.ocean.data_access.partitioned import materialize_partitioned_xarray
from domain.ocean.dask_utils import is_dask_backed, report_phase
from domain.ocean.events.detection_utils import (
    compute_duration_mask,
    compute_event_array,
    ensure_time_reduction_chunks,
    estimate_area_km2,
    extract_bbox_from_mask,
    report_detection_input,
    time_batch_slices,
)
from domain.ocean.events.vertical import prepare_event_vertical_field


def detect_eutrophication(
    chlorophyll: xr.DataArray,
    oxygen: Optional[xr.DataArray] = None,
    chlorophyll_percentile: float = 90,
    oxygen_threshold: Optional[float] = None,
    min_duration_days: int = 5,
    min_area_km2: float = 1000.0,
    vertical_mode: str = "surface",
    depth_value: float | None = None,
    depth_range: tuple[float, float] | None = None,
    depth_aggregation: str = "mean",
    analysis_mask: xr.DataArray | None = None,
) -> Dict:
    """
    Detect persistent eutrophication-like events.

    High chlorophyll is required. If oxygen is provided, low oxygen can be used
    as an additional constraint.
    """
    chlorophyll = materialize_partitioned_xarray(chlorophyll)
    oxygen = materialize_partitioned_xarray(oxygen)
    report_detection_input("eutrophication", chlorophyll, percent=0.02)

    if 'time' not in chlorophyll.dims:
        raise ValueError("detect_eutrophication requires a time dimension")
    chlorophyll = prepare_event_vertical_field(
        chlorophyll,
        default_mode="surface",
        vertical_mode=vertical_mode,
        depth_value=depth_value,
        depth_range=depth_range,
        depth_aggregation=depth_aggregation,
    )
    report_detection_input("eutrophication vertical field", chlorophyll, percent=0.08)
    analysis_mask = materialize_partitioned_xarray(analysis_mask)
    if analysis_mask is not None:
        chlorophyll, analysis_mask = xr.align(chlorophyll, analysis_mask.astype(bool), join="inner")
        chlorophyll = chlorophyll.where(analysis_mask)

    if oxygen is not None:
        oxygen = prepare_event_vertical_field(
            oxygen,
            default_mode="surface",
            vertical_mode=vertical_mode,
            depth_value=depth_value,
            depth_range=depth_range,
            depth_aggregation=depth_aggregation,
        )
        oxygen = oxygen.interp_like(chlorophyll)
        if analysis_mask is not None:
            oxygen = oxygen.where(analysis_mask)

    chlorophyll = ensure_time_reduction_chunks(chlorophyll, label="eutrophication")
    chl_threshold = chlorophyll.quantile(chlorophyll_percentile / 100.0, dim='time')
    threshold_values = compute_event_array(
        chl_threshold,
        label="eutrophication chlorophyll threshold",
        dtype=float,
        start=0.1,
        end=0.28,
    )
    threshold_np = xr.DataArray(threshold_values, coords=chl_threshold.coords, dims=chl_threshold.dims)
    enriched = chlorophyll > threshold_np

    if oxygen is not None and oxygen_threshold is not None:
        enriched = enriched & (oxygen < oxygen_threshold)

    enriched_mask = compute_event_array(
        enriched,
        label="eutrophication threshold mask",
        dtype=bool,
        start=0.3,
        end=0.48,
    )
    duration_mask = compute_duration_mask(enriched_mask, min_duration_days, event_label="eutrophication", start=0.49, end=0.56)
    events: List[Dict] = []
    event_mask = np.zeros_like(duration_mask, dtype=bool)
    lon = np.asarray(chlorophyll.lon.values)
    lat = np.asarray(chlorophyll.lat.values)
    time_values = chlorophyll.time.values if 'time' in chlorophyll.coords else None
    slices = time_batch_slices(chlorophyll)

    for batch_index, time_slice in enumerate(slices, start=1):
        batch_start = int(time_slice.start or 0)
        batch_stop = int(time_slice.stop or chlorophyll.sizes['time'])
        batch_label = f"{batch_start + 1}-{batch_stop}"
        report_phase(
            phase="extracting_event_properties",
            message="Extracting eutrophication event properties",
            percent=0.58 + 0.35 * ((batch_index - 1) / max(1, len(slices))),
            completed_units=batch_index - 1,
            total_units=len(slices),
            unit_label="time batch",
            current_unit=batch_label,
            compute_backend="dask" if is_dask_backed(chlorophyll) else "xarray",
        )
        if not np.any(duration_mask[time_slice]):
            continue
        chlorophyll_batch = compute_event_array(
            chlorophyll.isel(time=time_slice),
            label=f"eutrophication chlorophyll batch {batch_index}/{len(slices)}",
            dtype=float,
            start=0.58 + 0.25 * ((batch_index - 1) / max(1, len(slices))),
            end=0.58 + 0.25 * (batch_index / max(1, len(slices))),
        )
        oxygen_batch = None
        if oxygen is not None:
            oxygen_batch = compute_event_array(
                oxygen.isel(time=time_slice),
                label=f"eutrophication oxygen batch {batch_index}/{len(slices)}",
                dtype=float,
                start=0.83 + 0.1 * ((batch_index - 1) / max(1, len(slices))),
                end=0.83 + 0.1 * (batch_index / max(1, len(slices))),
            )

        for local_t, time_index in enumerate(range(batch_start, batch_stop)):
            if not np.any(duration_mask[time_index]):
                continue

            labels, n_labels = label(duration_mask[time_index])
            for label_index in range(1, n_labels + 1):
                mask = labels == label_index
                area_km2 = estimate_area_km2(mask, lon, lat)
                if area_km2 < min_area_km2:
                    continue

                event_mask[time_index][mask] = True

                values = chlorophyll_batch[local_t][mask]
                y_indices, x_indices = np.where(mask)
                max_idx = int(np.argmax(values))
                center_lat_idx = int(y_indices[max_idx])
                center_lon_idx = int(x_indices[max_idx])

                event = {
                    'event_id': f'eutrophication_{time_index}_{len(events) + 1}',
                    'time_index': int(time_index),
                    'bbox': extract_bbox_from_mask(mask, lon, lat),
                    'center': {
                        'lon': float(lon[center_lon_idx]),
                        'lat': float(lat[center_lat_idx]),
                    },
                    'mean_chlorophyll': float(np.mean(values)),
                    'max_chlorophyll': float(np.max(values)),
                    'area_km2': float(area_km2),
                    'n_pixels': int(mask.sum()),
                }
                if time_values is not None:
                    event['timestamp'] = str(time_values[time_index])
                if oxygen_batch is not None:
                    oxygen_values = oxygen_batch[local_t][mask]
                    event['mean_oxygen'] = float(np.mean(oxygen_values))
                    event['min_oxygen'] = float(np.min(oxygen_values))
                events.append(event)

    stats = {
        'total_count': len(events),
        'detection_params': {
            'chlorophyll_percentile': chlorophyll_percentile,
            'oxygen_threshold': oxygen_threshold,
            'min_duration_days': min_duration_days,
            'min_area_km2': min_area_km2,
            'vertical_mode': vertical_mode,
            'depth_value': depth_value,
            'depth_range': list(depth_range) if depth_range is not None else None,
            'depth_aggregation': depth_aggregation,
            'analysis_mask': analysis_mask is not None,
        },
    }
    if events:
        stats['mean_area_km2'] = float(np.mean([event['area_km2'] for event in events]))
        stats['mean_max_chlorophyll'] = float(np.mean([event['max_chlorophyll'] for event in events]))

    return {
        'event_type': 'eutrophication',
        'events': events,
        'statistics': stats,
        'event_mask': event_mask,
        'threshold_field': threshold_values,
        'coordinates': {
            'lon': chlorophyll.lon.values.tolist(),
            'lat': chlorophyll.lat.values.tolist(),
        },
    }

