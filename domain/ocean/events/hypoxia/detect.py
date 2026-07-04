"""
海洋事件检测工具 - 低氧区检测

检测溶解氧浓度低于临界阈值的缺氧区域
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Tuple
from scipy.ndimage import label

from domain.ocean.data_access.partitioned import materialize_partitioned_xarray
from domain.ocean.dask_utils import chunk_summary, dataarray_to_numpy, is_dask_backed, report_phase
from domain.ocean.events.detection_utils import (
    compute_duration_mask,
    estimate_area_km2,
    extract_bbox_from_mask,
    time_batch_slices,
)
from domain.ocean.events.vertical import prepare_event_vertical_field


def detect_hypoxia(
    oxygen: xr.DataArray,
    oxygen_threshold: float = 60,
    severe_threshold: float = 20,
    min_area_km2: float = 100,
    min_duration_days: int = 3,
    vertical_mode: str = "bottom",
    depth_value: float | None = None,
    depth_range: Tuple[float, float] | None = None,
    depth_aggregation: str = "mean",
    analysis_mask: xr.DataArray | None = None,
) -> Dict:
    """
    检测低氧区（缺氧区）

    识别溶解氧浓度持续低于阈值的区域

    应用场景：
    - 富营养化监测
    - 水质评估
    - 生态健康诊断

    Args:
        oxygen: 溶解氧数据（mmol/m³）
        oxygen_threshold: 缺氧阈值（默认60 mmol/m³）
        severe_threshold: 严重缺氧阈值（默认20 mmol/m³，接近无氧）
        min_area_km2: 最小面积(km²)
        min_duration_days: 最小持续天数

    Returns:
        低氧事件字典

    Example:
        >>> oxygen = ds['oxygen']  # (time, depth, lat, lon)
        >>> hypoxia = detect_hypoxia(oxygen, oxygen_threshold=60)
        >>> print(f"Found {hypoxia['statistics']['total_count']} hypoxic zones")
    """
    oxygen = materialize_partitioned_xarray(oxygen)
    report_phase(
        phase="preparing_hypoxia_detection",
        message="Preparing hypoxia detection input",
        percent=0.02,
        compute_backend="dask" if is_dask_backed(oxygen) else "xarray",
        chunks=chunk_summary(oxygen) if is_dask_backed(oxygen) else None,
    )

    # 如果有深度维度，选择底层或特定深度
    oxygen = prepare_event_vertical_field(
        oxygen,
        default_mode="bottom",
        vertical_mode=vertical_mode,
        depth_value=depth_value,
        depth_range=depth_range,
        depth_aggregation=depth_aggregation,
    )
    report_phase(
        phase="preparing_hypoxia_detection",
        message="Prepared vertical oxygen field for hypoxia detection",
        percent=0.08,
        compute_backend="dask" if is_dask_backed(oxygen) else "xarray",
        chunks=chunk_summary(oxygen) if is_dask_backed(oxygen) else None,
    )
    analysis_mask = materialize_partitioned_xarray(analysis_mask)
    if analysis_mask is not None:
        oxygen, analysis_mask = xr.align(oxygen, analysis_mask.astype(bool), join="inner")
        oxygen = oxygen.where(analysis_mask)

    has_time = 'time' in oxygen.dims

    if has_time:
        # 时间序列检测
        result = _detect_hypoxia_timeseries(
            oxygen, oxygen_threshold, severe_threshold,
            min_area_km2, min_duration_days
        )
    else:
        # 单时间步检测
        result = _detect_hypoxia_single(
            oxygen, oxygen_threshold, severe_threshold,
            min_area_km2
        )

    detection_params = result.setdefault('statistics', {}).setdefault('detection_params', {})
    detection_params.update({
        'vertical_mode': vertical_mode,
        'depth_value': depth_value,
        'depth_range': list(depth_range) if depth_range is not None else None,
        'depth_aggregation': depth_aggregation,
        'analysis_mask': analysis_mask is not None,
    })
    report_phase(
        phase="hypoxia_detection_complete",
        message="Hypoxia detection complete",
        percent=1.0,
    )
    return result


def _detect_hypoxia_single(
    oxygen: xr.DataArray,
    oxygen_threshold: float,
    severe_threshold: float,
    min_area_km2: float
) -> Dict:
    """单时间步的低氧区检测"""
    # 识别低氧区域
    oxygen_values = dataarray_to_numpy(
        oxygen,
        label="hypoxia oxygen field",
        dtype=float,
        start=0.1,
        end=0.65,
    )
    hypoxic_mask = oxygen_values < oxygen_threshold
    severe_mask = oxygen_values < severe_threshold

    # 标记连通区域
    hypoxic_labels, n_hypoxic = label(hypoxic_mask)
    severe_labels, n_severe = label(severe_mask)

    # 提取事件
    hypoxic_events = _extract_hypoxia_events(
        hypoxic_labels, n_hypoxic, oxygen_values,
        np.asarray(oxygen.lon.values), np.asarray(oxygen.lat.values),
        severity='moderate', threshold=oxygen_threshold,
        min_area_km2=min_area_km2
    )

    severe_events = _extract_hypoxia_events(
        severe_labels, n_severe, oxygen_values,
        np.asarray(oxygen.lon.values), np.asarray(oxygen.lat.values),
        severity='severe', threshold=severe_threshold,
        min_area_km2=min_area_km2
    )

    all_events = hypoxic_events + severe_events

    # 统计信息
    statistics = {
        'total_count': len(all_events),
        'moderate_count': len(hypoxic_events),
        'severe_count': len(severe_events),
        'detection_params': {
            'oxygen_threshold': oxygen_threshold,
            'severe_threshold': severe_threshold,
            'min_area_km2': min_area_km2
        }
    }

    if all_events:
        areas = [e['area_km2'] for e in all_events]
        intensities = [e['min_oxygen'] for e in all_events]
        statistics['mean_area_km2'] = float(np.mean(areas))
        statistics['min_oxygen_overall'] = float(np.min(intensities))

    return {
        'event_type': 'hypoxia',
        'events': all_events,
        'statistics': statistics,
        'event_mask': hypoxic_mask,
        'threshold_field': float(oxygen_threshold),
        'hypoxic_mask': hypoxic_mask,
        'severe_mask': severe_mask,
        'coordinates': {
            'lon': oxygen.lon.values.tolist(),
            'lat': oxygen.lat.values.tolist()
        }
    }


def _detect_hypoxia_timeseries(
    oxygen: xr.DataArray,
    oxygen_threshold: float,
    severe_threshold: float,
    min_area_km2: float,
    min_duration_days: int
) -> Dict:
    """时间序列的低氧区检测"""
    # 识别持续低氧区域
    report_phase(
        phase="computing_hypoxia_threshold_mask",
        message="Computing hypoxia threshold mask",
        percent=0.1,
        compute_backend="dask" if is_dask_backed(oxygen) else "xarray",
    )
    hypoxic_mask = dataarray_to_numpy(
        oxygen < oxygen_threshold,
        label="hypoxia threshold mask",
        dtype=bool,
        start=0.12,
        end=0.45,
    )

    # 计算持续时间掩码
    report_phase(
        phase="computing_hypoxia_persistence",
        message="Computing hypoxia persistence mask",
        percent=0.46,
    )
    duration_mask = compute_duration_mask(
        hypoxic_mask,
        min_duration_days,
        event_label="hypoxia",
        start=0.46,
        end=0.54,
    )

    # 在每个时间步检测空间连通区域
    all_events = []
    event_mask = np.zeros_like(duration_mask, dtype=bool)
    lon = np.asarray(oxygen.lon.values)
    lat = np.asarray(oxygen.lat.values)
    time_values = oxygen.time.values if 'time' in oxygen.coords else None
    slices = time_batch_slices(oxygen)

    for batch_index, time_slice in enumerate(slices, start=1):
        batch_start = int(time_slice.start or 0)
        batch_stop = int(time_slice.stop or oxygen.sizes['time'])
        batch_label = f"{batch_start + 1}-{batch_stop}"
        report_phase(
            phase="extracting_hypoxia_events",
            message="Extracting hypoxia event properties",
            percent=0.55 + 0.4 * ((batch_index - 1) / max(1, len(slices))),
            completed_units=batch_index - 1,
            total_units=len(slices),
            unit_label="time batch",
            current_unit=batch_label,
            compute_backend="dask" if is_dask_backed(oxygen) else "xarray",
        )
        if not np.any(duration_mask[time_slice]):
            continue

        oxygen_batch = dataarray_to_numpy(
            oxygen.isel(time=time_slice),
            label=f"hypoxia oxygen batch {batch_index}/{len(slices)}",
            dtype=float,
            start=0.55 + 0.4 * ((batch_index - 1) / max(1, len(slices))),
            end=0.55 + 0.4 * (batch_index / max(1, len(slices))),
        )

        for local_t, t in enumerate(range(batch_start, batch_stop)):
            if not np.any(duration_mask[t]):
                continue

            labeled, n_labels = label(duration_mask[t])

            for i in range(1, n_labels + 1):
                mask = labeled == i
                area_km2 = estimate_area_km2(
                    mask, lon, lat
                )

                if area_km2 < min_area_km2:
                    continue

                event_mask[t][mask] = True

                # 提取事件属性
                event = _extract_hypoxia_properties(
                    mask, oxygen_batch[local_t],
                    lon, lat,
                    oxygen_threshold, severe_threshold,
                    time_index=t,
                    component_index=i,
                    timestamp=time_values[t] if time_values is not None else None,
                )

                event['area_km2'] = float(area_km2)
                all_events.append(event)

        report_phase(
            phase="extracting_hypoxia_events",
            message="Extracting hypoxia event properties",
            percent=0.55 + 0.4 * (batch_index / max(1, len(slices))),
            completed_units=batch_index,
            total_units=len(slices),
            unit_label="time batch",
            current_unit=batch_label,
            compute_backend="dask" if is_dask_backed(oxygen) else "xarray",
        )

    report_phase(
        phase="assembling_hypoxia_detection",
        message="Assembling hypoxia detection result",
        percent=0.96,
    )

    # 统计信息
    statistics = {
        'total_count': len(all_events),
        'detection_params': {
            'oxygen_threshold': oxygen_threshold,
            'severe_threshold': severe_threshold,
            'min_duration_days': min_duration_days,
            'min_area_km2': min_area_km2
        }
    }

    return {
        'event_type': 'hypoxia',
        'events': all_events,
        'statistics': statistics,
        'event_mask': event_mask,
        'threshold_field': float(oxygen_threshold),
        'coordinates': {
            'lon': oxygen.lon.values.tolist(),
            'lat': oxygen.lat.values.tolist()
        }
    }


def _extract_hypoxia_events(
    labels: np.ndarray,
    n_labels: int,
    oxygen_values: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    severity: str,
    threshold: float,
    min_area_km2: float
) -> List[Dict]:
    """提取低氧事件"""
    events = []

    for i in range(1, n_labels + 1):
        mask = labels == i

        # 计算面积
        area_km2 = estimate_area_km2(mask, lon, lat)

        if area_km2 < min_area_km2:
            continue

        # 提取属性
        affected_oxygen = oxygen_values[mask]

        # 中心位置（最低氧浓度）
        min_idx = np.argmin(affected_oxygen)
        y_indices, x_indices = np.where(mask)

        center_lat_idx = y_indices[min_idx]
        center_lon_idx = x_indices[min_idx]

        center_lon = float(lon[center_lon_idx])
        center_lat = float(lat[center_lat_idx])

        event = {
            'event_id': f'hypoxia_{severity}_{len(events) + 1}',
            'severity': severity,
            'center': {'lon': center_lon, 'lat': center_lat},
            'bbox': extract_bbox_from_mask(mask, lon, lat),
            'area_km2': float(area_km2),
            'mean_oxygen': float(np.mean(affected_oxygen)),
            'min_oxygen': float(np.min(affected_oxygen)),
            'threshold': threshold
        }

        events.append(event)

    return events


def _extract_hypoxia_properties(
    mask: np.ndarray,
    oxygen_values: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    oxygen_threshold: float,
    severe_threshold: float,
    time_index: int,
    component_index: int | None = None,
    timestamp=None,
) -> Dict:
    """提取低氧事件属性"""
    affected_oxygen = oxygen_values[mask]

    # 确定严重程度
    if np.min(affected_oxygen) < severe_threshold:
        severity = 'severe'
    else:
        severity = 'moderate'

    # 中心位置
    min_idx = np.argmin(affected_oxygen)
    y_indices, x_indices = np.where(mask)

    center_lat_idx = y_indices[min_idx]
    center_lon_idx = x_indices[min_idx]

    event_suffix = f'{time_index}_{component_index}' if component_index is not None else str(time_index)
    event = {
        'event_id': f'hypoxia_{event_suffix}',
        'time_index': time_index,
        'severity': severity,
        'bbox': extract_bbox_from_mask(mask, lon, lat),
        'center': {
            'lon': float(lon[center_lon_idx]),
            'lat': float(lat[center_lat_idx])
        },
        'mean_oxygen': float(np.mean(affected_oxygen)),
        'min_oxygen': float(np.min(affected_oxygen))
    }
    if timestamp is not None:
        event['timestamp'] = str(timestamp)
    return event

