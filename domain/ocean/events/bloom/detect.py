"""
海洋事件检测工具 - 藻华检测

检测叶绿素浓度异常升高的藻华事件
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Literal
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


def detect_algal_blooms(
    chlorophyll: xr.DataArray,
    threshold: float | None = None,
    percentile_threshold: float | None = None,
    min_duration_days: int = 5,
    min_area_km2: float = 500,
    bloom_type: Literal['auto', 'spring', 'harmful'] = 'auto',
    vertical_mode: str = "surface",
    depth_value: float | None = None,
    depth_range: tuple[float, float] | None = None,
    depth_aggregation: str = "mean",
    analysis_mask: xr.DataArray | None = None,
) -> Dict:
    """
    检测藻华事件

    基于叶绿素浓度阈值方法，识别持续高浓度叶绿素的区域

    藻华类型：
    - spring: 春季藻华（自然现象）
    - harmful: 有害藻华（富营养化驱动）
    - auto: 自动分类

    Args:
        chlorophyll: 叶绿素-a浓度数据（需要time维度）
        threshold: 绝对叶绿素阈值；未指定 threshold 和 percentile_threshold 时默认使用 1.0
        percentile_threshold: 百分位阈值（0-100），用于相对异常检测；显式指定时优先于默认绝对阈值
        min_duration_days: 最小持续天数
        min_area_km2: 最小面积(km²)
        bloom_type: 藻华类型分类

    Returns:
        藻华事件字典

    Example:
        >>> chl = ds['chlorophyll']  # (time, lat, lon)
        >>> blooms = detect_algal_blooms(chl, percentile_threshold=85, min_duration_days=5)
        >>> print(f"Found {blooms['statistics']['total_count']} algal bloom events")
    """
    chlorophyll = materialize_partitioned_xarray(chlorophyll)
    report_detection_input("algal bloom", chlorophyll, percent=0.02)

    if 'time' not in chlorophyll.dims:
        raise ValueError("Chlorophyll data must have time dimension")

    # 如果有深度维度，取表层
    chlorophyll = prepare_event_vertical_field(
        chlorophyll,
        default_mode="surface",
        vertical_mode=vertical_mode,
        depth_value=depth_value,
        depth_range=depth_range,
        depth_aggregation=depth_aggregation,
    )
    report_detection_input("algal bloom vertical field", chlorophyll, percent=0.08)
    analysis_mask = materialize_partitioned_xarray(analysis_mask)
    if analysis_mask is not None:
        chlorophyll, analysis_mask = xr.align(chlorophyll, analysis_mask.astype(bool), join="inner")
        chlorophyll = chlorophyll.where(analysis_mask)

    # 计算阈值。绝对阈值来自用户显式条件；百分位阈值用于相对异常检测。
    chlorophyll = ensure_time_reduction_chunks(chlorophyll, label="algal bloom")
    if threshold is not None or percentile_threshold is None:
        threshold_value = 1.0 if threshold is None else float(threshold)
        threshold = threshold_value
        threshold_field = xr.zeros_like(chlorophyll.isel(time=0, drop=True)) + threshold_value
        threshold_values = compute_event_array(
            threshold_field,
            label="algal bloom absolute threshold",
            dtype=float,
            start=0.1,
            end=0.28,
        )
        threshold_np = xr.DataArray(threshold_values, coords=threshold_field.coords, dims=threshold_field.dims)
    else:
        threshold_field = chlorophyll.quantile(float(percentile_threshold) / 100.0, dim='time')
        threshold_values = compute_event_array(
            threshold_field,
            label="algal bloom percentile threshold",
            dtype=float,
            start=0.1,
            end=0.28,
        )
        threshold_np = xr.DataArray(threshold_values, coords=threshold_field.coords, dims=threshold_field.dims)

    # 识别超过阈值的区域
    high_chl = compute_event_array(
        chlorophyll > threshold_np,
        label="algal bloom threshold mask",
        dtype=bool,
        start=0.3,
        end=0.48,
    )

    # 计算持续时间掩码
    duration_mask = compute_duration_mask(high_chl, min_duration_days, event_label="algal bloom", start=0.49, end=0.56)

    # 检测藻华事件
    bloom_events = []
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
            message="Extracting algal bloom event properties",
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
            label=f"algal bloom chlorophyll batch {batch_index}/{len(slices)}",
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
                event = _extract_bloom_properties(
                    mask, chlorophyll_batch[local_t], threshold_values,
                    lon, lat,
                    time_index=t,
                    bloom_type=bloom_type,
                    timestamp=time_values[t] if time_values is not None else None,
                )

                event['area_km2'] = float(area_km2)
                bloom_events.append(event)

    # 分类统计
    report_phase(phase="assembling_event_detection", message="Assembling algal bloom detection result", percent=0.95)
    statistics = _compute_bloom_statistics(bloom_events, bloom_type)
    statistics['detection_params'] = {
        'threshold': float(threshold) if threshold is not None else None,
        'percentile_threshold': percentile_threshold,
        'min_duration_days': min_duration_days,
        'min_area_km2': min_area_km2,
        'bloom_type': bloom_type,
        'vertical_mode': vertical_mode,
        'depth_value': depth_value,
        'depth_range': list(depth_range) if depth_range is not None else None,
        'depth_aggregation': depth_aggregation,
        'analysis_mask': analysis_mask is not None,
    }

    return {
        'event_type': 'algal_bloom',
        'events': bloom_events,
        'statistics': statistics,
        'event_mask': event_mask,
        'threshold_field': threshold_values,
        'coordinates': {
            'lon': chlorophyll.lon.values.tolist(),
            'lat': chlorophyll.lat.values.tolist()
        }
    }


def _extract_bloom_properties(
    mask: np.ndarray,
    chlorophyll_values: np.ndarray,
    threshold_values: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    time_index: int,
    bloom_type: str,
    timestamp=None,
) -> Dict:
    """提取藻华事件属性"""
    # 叶绿素值
    affected_chl = chlorophyll_values[mask]
    threshold_vals = threshold_values[mask]

    # 中心位置（最大叶绿素浓度）
    max_idx = np.argmax(affected_chl)
    y_indices, x_indices = np.where(mask)

    center_lat_idx = y_indices[max_idx]
    center_lon_idx = x_indices[max_idx]

    center_lon = float(lon[center_lon_idx])
    center_lat = float(lat[center_lat_idx])

    # 强度（相对于阈值的增量）
    intensity = affected_chl - threshold_vals

    # 藻华分类（如果是auto）
    if bloom_type == 'auto':
        # 简单规则：基于叶绿素浓度和时间
        mean_chl = np.mean(affected_chl)
        if mean_chl > 10:  # mg/m³
            classified_type = 'harmful'
        else:
            classified_type = 'spring'
    else:
        classified_type = bloom_type

    event = {
        'event_id': f'bloom_{time_index}',
        'time_index': time_index,
        'bloom_type': classified_type,
        'bbox': extract_bbox_from_mask(mask, lon, lat),
        'center': {'lon': center_lon, 'lat': center_lat},
        'mean_chlorophyll': float(np.mean(affected_chl)),
        'max_chlorophyll': float(np.max(affected_chl)),
        'mean_intensity': float(np.mean(intensity)),
        'max_intensity': float(np.max(intensity)),
        'n_pixels': int(np.sum(mask))
    }
    if timestamp is not None:
        event['timestamp'] = str(timestamp)
    return event


def _compute_bloom_statistics(events: List[Dict], bloom_type: str) -> Dict:
    """计算藻华统计信息"""
    if not events:
        return {
            'total_count': 0,
            'spring_count': 0,
            'harmful_count': 0
        }

    # 按类型分类
    spring_events = [e for e in events if e.get('bloom_type') == 'spring']
    harmful_events = [e for e in events if e.get('bloom_type') == 'harmful']

    statistics = {
        'total_count': len(events),
        'spring_count': len(spring_events),
        'harmful_count': len(harmful_events)
    }

    # 整体统计
    areas = [e.get('area_km2', 0) for e in events]
    intensities = [e['max_intensity'] for e in events]

    statistics['mean_area_km2'] = float(np.mean(areas))
    statistics['mean_intensity'] = float(np.mean(intensities))
    statistics['max_area_km2'] = float(np.max(areas))
    statistics['max_intensity'] = float(np.max(intensities))

    # 分类型统计
    if spring_events:
        spring_areas = [e.get('area_km2', 0) for e in spring_events]
        statistics['spring_mean_area_km2'] = float(np.mean(spring_areas))

    if harmful_events:
        harmful_areas = [e.get('area_km2', 0) for e in harmful_events]
        statistics['harmful_mean_area_km2'] = float(np.mean(harmful_areas))

    return statistics
