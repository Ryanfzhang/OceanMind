"""
海洋事件检测工具 - 海洋热浪

检测海洋热浪事件
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Tuple
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


def detect_heatwaves(
    temp: xr.DataArray,
    percentile_threshold: float = 90,
    min_duration_days: int = 5,
    min_area_km2: float = 1000,
    vertical_mode: str = "surface",
    depth_value: float | None = None,
    depth_range: Tuple[float, float] | None = None,
    depth_aggregation: str = "mean",
    analysis_mask: xr.DataArray | None = None,
) -> Dict:
    """
    检测海洋热浪事件

    基于百分位阈值方法：识别温度持续超过百分位阈值的区域

    Args:
        temp: 温度数据（需要time维度）
        percentile_threshold: 百分位阈值（0-100）
        min_duration_days: 最小持续天数
        min_area_km2: 最小面积(km²)

    Returns:
        热浪事件字典

    Example:
        >>> temp = ds['temp']  # (time, lat, lon)
        >>> heatwaves = detect_heatwaves(temp, percentile_threshold=90, min_duration_days=5)
        >>> print(f"Found {heatwaves['statistics']['total_count']} heatwave events")
    """
    temp = materialize_partitioned_xarray(temp)
    report_detection_input("heatwave", temp, percent=0.02)

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
    report_detection_input("heatwave vertical field", temp, percent=0.08)
    analysis_mask = materialize_partitioned_xarray(analysis_mask)
    if analysis_mask is not None:
        temp, analysis_mask = xr.align(temp, analysis_mask.astype(bool), join="inner")
        temp = temp.where(analysis_mask)

    # 计算百分位阈值
    temp = ensure_time_reduction_chunks(temp, label="heatwave")
    threshold = temp.quantile(percentile_threshold / 100.0, dim='time')
    threshold_values = compute_event_array(
        threshold,
        label="heatwave percentile threshold",
        dtype=float,
        start=0.1,
        end=0.28,
    )
    threshold_np = xr.DataArray(threshold_values, coords=threshold.coords, dims=threshold.dims)

    # 识别超过阈值的区域
    exceedance = compute_event_array(
        temp > threshold_np,
        label="heatwave exceedance mask",
        dtype=bool,
        start=0.3,
        end=0.48,
    )

    # 计算每个网格点的持续天数
    duration_mask = compute_duration_mask(exceedance, min_duration_days, event_label="heatwave", start=0.49, end=0.56)

    # 识别热浪事件（空间连通区域）
    heatwave_events = []
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
            message="Extracting heatwave event properties",
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
            label=f"heatwave temperature batch {batch_index}/{len(slices)}",
            dtype=float,
            start=0.58 + 0.35 * ((batch_index - 1) / max(1, len(slices))),
            end=0.58 + 0.35 * (batch_index / max(1, len(slices))),
        )

        for local_t, t in enumerate(range(batch_start, batch_stop)):
            if not np.any(duration_mask[t]):
                continue

            # 标记连通区域
            labeled, n_labels = label(duration_mask[t])

            # 提取每个热浪事件
            for i in range(1, n_labels + 1):
                mask = labeled == i
                n_pixels = np.sum(mask)

                # 计算面积
                area_km2 = estimate_area_km2(mask, lon, lat)

                if area_km2 < min_area_km2:
                    continue

                event_mask[t][mask] = True

                # 提取事件属性
                event = _extract_heatwave_properties(
                    mask, temp_batch[local_t], threshold_values,
                    lon, lat,
                    time_index=t,
                    timestamp=time_values[t] if time_values is not None else None,
                )

                event['area_km2'] = float(area_km2)
                event['n_pixels'] = int(n_pixels)

                heatwave_events.append(event)

    # 合并时间上连续的热浪事件
    report_phase(phase="assembling_event_detection", message="Assembling heatwave detection result", percent=0.95)
    merged_events = _merge_temporal_events(heatwave_events, max_gap_days=2)

    # 统计信息
    statistics = {
        'total_count': len(merged_events),
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

    if merged_events:
        areas = [e['area_km2'] for e in merged_events]
        intensities = [e['max_intensity'] for e in merged_events]
        statistics['mean_area_km2'] = float(np.mean(areas))
        statistics['mean_intensity'] = float(np.mean(intensities))
        statistics['max_area_km2'] = float(np.max(areas))
        statistics['max_intensity'] = float(np.max(intensities))

    return {
        'event_type': 'heatwave',
        'events': merged_events,
        'statistics': statistics,
        'event_mask': event_mask,
        'threshold_field': threshold_values,
        'coordinates': {
            'lon': temp.lon.values.tolist(),
            'lat': temp.lat.values.tolist()
        }
    }


def _extract_heatwave_properties(
    mask: np.ndarray,
    temp_values: np.ndarray,
    threshold_values: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    time_index: int,
    timestamp=None,
) -> Dict:
    """提取热浪事件属性"""
    # 温度异常
    temp_anomaly = temp_values - threshold_values

    # 受影响区域的温度
    affected_temps = temp_values[mask]
    affected_anomalies = temp_anomaly[mask]

    # 中心位置（最大异常）
    max_idx = np.argmax(affected_anomalies)
    y_indices, x_indices = np.where(mask)

    center_lat_idx = y_indices[max_idx]
    center_lon_idx = x_indices[max_idx]

    center_lon = float(lon[center_lon_idx])
    center_lat = float(lat[center_lat_idx])

    event = {
        'event_id': f'heatwave_{time_index}',
        'time_index': time_index,
        'center': {'lon': center_lon, 'lat': center_lat},
        'bbox': extract_bbox_from_mask(mask, lon, lat),
        'mean_temp': float(np.mean(affected_temps)),
        'max_temp': float(np.max(affected_temps)),
        'mean_intensity': float(np.mean(affected_anomalies)),
        'max_intensity': float(np.max(affected_anomalies))
    }
    if timestamp is not None:
        event['timestamp'] = str(timestamp)
    return event


def _merge_temporal_events(events: List[Dict], max_gap_days: int = 2) -> List[Dict]:
    """
    合并时间上连续的事件

    如果两个事件在时间和空间上接近，将它们合并为一个事件

    Args:
        events: 事件列表
        max_gap_days: 最大时间间隔（天）

    Returns:
        合并后的事件列表
    """
    if not events:
        return []

    # 按时间排序
    sorted_events = sorted(events, key=lambda e: e['time_index'])

    merged = []
    current_group = [sorted_events[0]]

    for event in sorted_events[1:]:
        last_event = current_group[-1]

        # 检查时间连续性
        time_gap = event['time_index'] - last_event['time_index']

        # 检查空间邻近性
        dist = _haversine_distance(
            last_event['center']['lon'], last_event['center']['lat'],
            event['center']['lon'], event['center']['lat']
        )

        if time_gap <= max_gap_days and dist < 500:  # 500km
            current_group.append(event)
        else:
            # 保存当前组，开始新组
            merged.append(_combine_event_group(current_group))
            current_group = [event]

    # 保存最后一组
    merged.append(_combine_event_group(current_group))

    return merged


def _combine_event_group(events: List[Dict]) -> Dict:
    """合并事件组为单个事件"""
    if len(events) == 1:
        return events[0]

    combined = {
        'event_id': f"heatwave_{events[0]['time_index']}_{events[-1]['time_index']}",
        'time_index': events[0]['time_index'],
        'duration_days': len(events),
        'center': events[0]['center'],  # 使用第一个事件的中心
        'bbox': {
            'lon_min': float(min(e.get('bbox', {}).get('lon_min', e['center']['lon']) for e in events)),
            'lon_max': float(max(e.get('bbox', {}).get('lon_max', e['center']['lon']) for e in events)),
            'lat_min': float(min(e.get('bbox', {}).get('lat_min', e['center']['lat']) for e in events)),
            'lat_max': float(max(e.get('bbox', {}).get('lat_max', e['center']['lat']) for e in events)),
        },
        'mean_temp': float(np.mean([e['mean_temp'] for e in events])),
        'max_temp': float(np.max([e['max_temp'] for e in events])),
        'mean_intensity': float(np.mean([e['mean_intensity'] for e in events])),
        'max_intensity': float(np.max([e['max_intensity'] for e in events])),
        'area_km2': float(np.mean([e.get('area_km2', 0) for e in events])),
        'n_pixels': int(np.mean([e.get('n_pixels', 0) for e in events]))
    }
    if events[0].get('timestamp') is not None:
        combined['timestamp'] = events[0]['timestamp']
    if events[-1].get('timestamp') is not None:
        combined['end_timestamp'] = events[-1]['timestamp']
    return combined


def _haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """计算两点之间的Haversine距离(km)"""
    R = 6371.0  # 地球半径(km)

    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c
