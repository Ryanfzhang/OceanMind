"""
海洋事件检测工具 - 上升流检测

检测由深层冷水上涌引起的上升流事件
"""

import numpy as np
import xarray as xr
from typing import Dict, List
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


def detect_upwelling(
    temp: xr.DataArray,
    percentile_threshold: float = 10,
    min_duration_days: int = 5,
    min_area_km2: float = 1000,
    vertical_mode: str = "surface",
    depth_value: float | None = None,
    depth_range: tuple[float, float] | None = None,
    depth_aggregation: str = "mean",
    analysis_mask: xr.DataArray | None = None,
) -> Dict:
    """
    检测上升流事件

    基于低百分位阈值方法：识别温度持续异常偏低的区域

    上升流特征：
    - 表层水温异常偏低
    - 深层营养盐丰富的水上涌
    - 对渔业和生态系统重要

    Args:
        temp: 海表温度数据（需要time维度）
        percentile_threshold: 低百分位阈值（0-100），越小检测越强的上升流
        min_duration_days: 最小持续天数
        min_area_km2: 最小面积(km²)

    Returns:
        上升流事件字典

    Example:
        >>> sst = ds['temp']  # (time, lat, lon)
        >>> upwelling = detect_upwelling(sst, percentile_threshold=10, min_duration_days=5)
        >>> print(f"Found {upwelling['statistics']['total_count']} upwelling events")
    """
    temp = materialize_partitioned_xarray(temp)
    report_detection_input("upwelling", temp, percent=0.02)

    if 'time' not in temp.dims:
        raise ValueError("Temperature data must have time dimension")

    # 如果有深度维度，取表层
    temp = prepare_event_vertical_field(
        temp,
        default_mode="surface",
        vertical_mode=vertical_mode,
        depth_value=depth_value,
        depth_range=depth_range,
        depth_aggregation=depth_aggregation,
    )
    report_detection_input("upwelling vertical field", temp, percent=0.08)
    analysis_mask = materialize_partitioned_xarray(analysis_mask)
    if analysis_mask is not None:
        temp, analysis_mask = xr.align(temp, analysis_mask.astype(bool), join="inner")
        temp = temp.where(analysis_mask)

    # 计算低百分位阈值（上升流通常是低温）
    temp = ensure_time_reduction_chunks(temp, label="upwelling")
    threshold = temp.quantile(percentile_threshold / 100.0, dim='time')
    threshold_values = compute_event_array(
        threshold,
        label="upwelling percentile threshold",
        dtype=float,
        start=0.1,
        end=0.28,
    )
    threshold_np = xr.DataArray(threshold_values, coords=threshold.coords, dims=threshold.dims)

    # 识别低于阈值的区域
    cold_anomaly = compute_event_array(
        temp < threshold_np,
        label="upwelling cold anomaly mask",
        dtype=bool,
        start=0.3,
        end=0.48,
    )

    # 计算持续时间掩码
    duration_mask = compute_duration_mask(cold_anomaly, min_duration_days, event_label="upwelling", start=0.49, end=0.56)

    # 检测上升流事件
    upwelling_events = []
    event_mask = np.zeros_like(duration_mask, dtype=bool)
    lon = np.asarray(temp.lon.values)
    lat = np.asarray(temp.lat.values)
    time_values = temp.time.values if 'time' in temp.coords else None
    slices = time_batch_slices(temp)

    for batch_index, time_slice in enumerate(slices, start=1):
        batch_start = int(time_slice.start or 0)
        batch_stop = int(time_slice.stop or temp.sizes['time'])
        batch_label = f"{batch_start + 1}-{batch_stop}"
        report_phase(
            phase="extracting_event_properties",
            message="Extracting upwelling event properties",
            percent=0.58 + 0.35 * ((batch_index - 1) / max(1, len(slices))),
            completed_units=batch_index - 1,
            total_units=len(slices),
            unit_label="time batch",
            current_unit=batch_label,
            compute_backend="dask" if is_dask_backed(temp) else "xarray",
        )
        if not np.any(duration_mask[time_slice]):
            continue
        temp_batch = compute_event_array(
            temp.isel(time=time_slice),
            label=f"upwelling temperature batch {batch_index}/{len(slices)}",
            dtype=float,
            start=0.58 + 0.35 * ((batch_index - 1) / max(1, len(slices))),
            end=0.58 + 0.35 * (batch_index / max(1, len(slices))),
        )

        for local_t, t in enumerate(range(batch_start, batch_stop)):
            if not np.any(duration_mask[t]):
                continue

            # 标记连通区域
            labeled, n_labels = label(duration_mask[t])

            for i in range(1, n_labels + 1):
                mask = labeled == i

                # 计算面积
                area_km2 = estimate_area_km2(mask, lon, lat)

                if area_km2 < min_area_km2:
                    continue

                event_mask[t][mask] = True

                # 提取事件属性
                event = _extract_upwelling_properties(
                    mask, temp_batch[local_t], threshold_values,
                    lon, lat,
                    time_index=t,
                    timestamp=time_values[t] if time_values is not None else None,
                )

                event['area_km2'] = float(area_km2)
                upwelling_events.append(event)

    # 统计信息
    report_phase(phase="assembling_event_detection", message="Assembling upwelling detection result", percent=0.95)
    statistics = {
        'total_count': len(upwelling_events),
        'detection_params': {
            'percentile_threshold': percentile_threshold,
            'min_duration_days': min_duration_days,
            'min_area_km2': min_area_km2,
            'vertical_mode': vertical_mode,
            'depth_value': depth_value,
            'depth_range': list(depth_range) if depth_range is not None else None,
            'depth_aggregation': depth_aggregation,
            'analysis_mask': analysis_mask is not None,
        }
    }

    if upwelling_events:
        areas = [e['area_km2'] for e in upwelling_events]
        intensities = [e['max_intensity'] for e in upwelling_events]
        statistics['mean_area_km2'] = float(np.mean(areas))
        statistics['mean_intensity'] = float(np.mean(intensities))

    return {
        'event_type': 'upwelling',
        'events': upwelling_events,
        'statistics': statistics,
        'event_mask': event_mask,
        'threshold_field': threshold_values,
        'coordinates': {
            'lon': temp.lon.values.tolist(),
            'lat': temp.lat.values.tolist()
        }
    }


def _extract_upwelling_properties(
    mask: np.ndarray,
    temp_values: np.ndarray,
    threshold_values: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    time_index: int,
    timestamp=None,
) -> Dict:
    """提取上升流事件属性"""
    # 温度异常（负值，因为上升流是冷水）
    temp_anomaly = temp_values - threshold_values

    # 受影响区域的温度
    affected_temps = temp_values[mask]
    affected_anomalies = temp_anomaly[mask]

    # 中心位置（最大负异常，即最冷点）
    min_idx = np.argmin(affected_temps)
    y_indices, x_indices = np.where(mask)

    center_lat_idx = y_indices[min_idx]
    center_lon_idx = x_indices[min_idx]

    center_lon = float(lon[center_lon_idx])
    center_lat = float(lat[center_lat_idx])

    # 强度（温度低于阈值的程度）
    intensity = -affected_anomalies  # 转为正值表示强度

    event = {
        'event_id': f'upwelling_{time_index}',
        'time_index': time_index,
        'bbox': extract_bbox_from_mask(mask, lon, lat),
        'center': {'lon': center_lon, 'lat': center_lat},
        'mean_temp': float(np.mean(affected_temps)),
        'min_temp': float(np.min(affected_temps)),
        'mean_intensity': float(np.mean(intensity)),
        'max_intensity': float(np.max(intensity)),
        'n_pixels': int(np.sum(mask))
    }
    if timestamp is not None:
        event['timestamp'] = str(timestamp)
    return event
