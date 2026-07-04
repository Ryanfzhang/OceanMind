"""
事件统计分析工具

对检测到的事件进行统计分析
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Literal, Tuple

from domain.ocean.data_access.partitioned import materialize_partitioned_xarray
from domain.ocean.dask_utils import dataarray_to_numpy, is_dask_backed, report_phase
from domain.ocean.events.detection_utils import ensure_time_reduction_chunks
from domain.ocean.events.vertical import prepare_event_vertical_field
from domain.ocean.result_payload import as_numeric_array


def compute_event_statistics(
    events: List[Dict],
    group_by: Literal['month', 'season', 'year'] = 'month'
) -> Dict:
    """
    计算事件统计信息

    按时间分组统计事件的数量、强度、面积等指标

    Args:
        events: 事件列表（来自detect_*函数的events字段）
        group_by: 分组方式
            - 'month': 按月统计
            - 'season': 按季节统计
            - 'year': 按年统计

    Returns:
        统计结果字典

    Example:
        >>> events = heatwaves['events']
        >>> stats = compute_event_statistics(events, group_by='month')
        >>> print(stats['monthly_counts'])
    """
    if not events:
        return {'total_count': 0, 'groups': {}}

    # 提取时间信息（如果有）
    if 'time_index' not in events[0]:
        # 如果没有时间信息，只计算总体统计
        return _compute_overall_statistics(events)

    # 转换为DataFrame方便分组
    df = pd.DataFrame(events)

    # 如果有时间戳，转换为datetime
    if 'timestamp' in df.columns:
        df['time'] = pd.to_datetime(df['timestamp'])
    elif 'time_index' in df.columns:
        # 使用time_index作为代理
        df['time'] = df['time_index']

    # 添加分组列
    if group_by == 'month':
        if 'time' in df.columns and isinstance(df['time'].iloc[0], pd.Timestamp):
            df['group'] = df['time'].dt.strftime('%b')
            group_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        else:
            df['group'] = df['time_index'] % 12
            group_labels = list(range(12))

    elif group_by == 'season':
        if 'time' in df.columns and isinstance(df['time'].iloc[0], pd.Timestamp):
            month = df['time'].dt.month
            df['group'] = month.map({
                12: 'DJF', 1: 'DJF', 2: 'DJF',
                3: 'MAM', 4: 'MAM', 5: 'MAM',
                6: 'JJA', 7: 'JJA', 8: 'JJA',
                9: 'SON', 10: 'SON', 11: 'SON'
            })
            group_labels = ['DJF', 'MAM', 'JJA', 'SON']
        else:
            df['group'] = (df['time_index'] // 3) % 4
            group_labels = ['Winter', 'Spring', 'Summer', 'Fall']

    elif group_by == 'year':
        if 'time' in df.columns and isinstance(df['time'].iloc[0], pd.Timestamp):
            df['group'] = df['time'].dt.year
            group_labels = sorted(df['group'].unique())
        else:
            df['group'] = df['time_index'] // 12
            group_labels = sorted(df['group'].unique())

    # 按组统计
    grouped = df.groupby('group')

    statistics = {
        'total_count': len(events),
        'group_by': group_by,
        'groups': {}
    }

    for group_label in group_labels:
        if group_label not in grouped.groups:
            statistics['groups'][str(group_label)] = {
                'count': 0,
                'mean_intensity': 0.0,
                'mean_area_km2': 0.0
            }
            continue

        group_data = grouped.get_group(group_label)

        # 计算统计指标
        count = len(group_data)

        # 提取强度指标（可能有不同的字段名）
        intensity_fields = ['max_intensity', 'intensity', 'max_gradient',
                           'max_vorticity', 'max_chlorophyll']
        intensities = []
        for field in intensity_fields:
            if field in group_data.columns:
                intensities = group_data[field].dropna().values
                break

        # 提取面积
        areas = group_data.get('area_km2', pd.Series()).dropna().values

        statistics['groups'][str(group_label)] = {
            'count': int(count),
            'mean_intensity': float(np.mean(intensities)) if len(intensities) > 0 else 0.0,
            'max_intensity': float(np.max(intensities)) if len(intensities) > 0 else 0.0,
            'mean_area_km2': float(np.mean(areas)) if len(areas) > 0 else 0.0,
            'total_area_km2': float(np.sum(areas)) if len(areas) > 0 else 0.0
        }

    return statistics


def _compute_overall_statistics(events: List[Dict]) -> Dict:
    """计算总体统计（无时间分组）"""
    # 提取所有可能的强度字段
    intensity_fields = ['max_intensity', 'intensity', 'max_gradient',
                       'max_vorticity', 'max_chlorophyll', 'min_oxygen']
    intensities = []
    for event in events:
        for field in intensity_fields:
            if field in event:
                intensities.append(event[field])
                break

    # 提取面积
    areas = [e.get('area_km2', 0) for e in events if 'area_km2' in e]

    statistics = {
        'total_count': len(events),
        'mean_intensity': float(np.mean(intensities)) if intensities else 0.0,
        'max_intensity': float(np.max(intensities)) if intensities else 0.0,
        'min_intensity': float(np.min(intensities)) if intensities else 0.0,
        'std_intensity': float(np.std(intensities)) if intensities else 0.0
    }

    if areas:
        statistics['mean_area_km2'] = float(np.mean(areas))
        statistics['max_area_km2'] = float(np.max(areas))
        statistics['total_area_km2'] = float(np.sum(areas))

    return statistics


def compute_event_spatial_distribution(events: List[Dict]) -> Dict:
    """
    计算事件的空间分布

    Args:
        events: 事件列表

    Returns:
        空间分布统计

    Example:
        >>> spatial_dist = compute_event_spatial_distribution(events)
        >>> print(f"Centroid: {spatial_dist['centroid']}")
    """
    if not events:
        return {'event_count': 0}

    # 提取中心位置
    centers = [e['center'] for e in events if 'center' in e]

    if not centers:
        return {'event_count': len(events), 'has_location': False}

    lons = [c['lon'] for c in centers]
    lats = [c['lat'] for c in centers]

    # 计算质心
    centroid_lon = float(np.mean(lons))
    centroid_lat = float(np.mean(lats))

    # 计算空间范围
    lon_range = [float(np.min(lons)), float(np.max(lons))]
    lat_range = [float(np.min(lats)), float(np.max(lats))]

    # 计算离散度
    lon_std = float(np.std(lons))
    lat_std = float(np.std(lats))

    return {
        'event_count': len(events),
        'has_location': True,
        'centroid': {'lon': centroid_lon, 'lat': centroid_lat},
        'lon_range': lon_range,
        'lat_range': lat_range,
        'spatial_dispersion': {
            'lon_std': lon_std,
            'lat_std': lat_std
        }
    }


def compare_event_periods(
    events1: List[Dict],
    events2: List[Dict],
    period1_label: str = 'Period 1',
    period2_label: str = 'Period 2'
) -> Dict:
    """
    比较两个时期的事件特征

    Args:
        events1: 第一时期的事件列表（由 detect_* 工具输出）
        events2: 第二时期的事件列表（由 detect_* 工具输出）
        period1_label: 第一时期标签
        period2_label: 第二时期标签

    Returns:
        比较结果

    Example:
        >>> comparison = compare_event_periods(
        ...     events1=period1_detection['events'],
        ...     events2=period2_detection['events'],
        ...     period1_label='2010-2015',
        ...     period2_label='2016-2020'
        ... )
    """
    # 计算各自的统计
    period1_stats = _compute_overall_statistics(events1)
    period2_stats = _compute_overall_statistics(events2)

    # 计算变化
    count_change = period2_stats['total_count'] - period1_stats['total_count']
    intensity_change = period2_stats.get('mean_intensity', 0) - period1_stats.get('mean_intensity', 0)

    return {
        'comparison': {
            period1_label: period1_stats,
            period2_label: period2_stats
        },
        'changes': {
            'count_change': int(count_change),
            'intensity_change': float(intensity_change),
            'count_change_percent': float(count_change / max(period1_stats['total_count'], 1) * 100)
        }
    }


def compute_event_frequency_map(
    events: Optional[List[Dict]] = None,
    event_detection: Optional[Dict] = None,
    data: Optional[xr.DataArray] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    resolution_deg: Optional[float] = None,
    normalize: bool = False
) -> Dict:
    """
    Compute a gridded event frequency map.

    When a full event-detection result is provided, the map is computed from
    the detector's event mask so every affected grid cell contributes. This is
    the preferred path for hotspot/frequency maps. The event-center histogram
    path is retained as a fallback for callers that only have an event list.

    Args:
        events: Optional event list from `detect_*()["events"]`.
        event_detection: Optional full event detection result with event_mask.
        data: Optional source field used to infer the ocean-valid domain mask.
        lon_range: Optional longitude range for the output grid.
        lat_range: Optional latitude range for the output grid.
        resolution_deg: Optional coarse grid spacing in degrees. If omitted,
            the source data lon/lat grid is used.
        normalize: If True, normalize counts by total event count.

    Returns:
        Map-ready spatial field result.
    """
    data = materialize_partitioned_xarray(data)
    events = list(events or (event_detection or {}).get("events") or [])
    if event_detection is not None and data is not None and event_detection.get("event_mask") is not None:
        return _compute_event_mask_frequency_map(
            event_detection=event_detection,
            data=data,
            normalize=normalize,
            event_count=len(events),
        )

    resolution_value = _normalize_frequency_resolution(resolution_deg)

    centers = [event["center"] for event in events if "center" in event]

    if centers:
        lons = np.asarray([center["lon"] for center in centers], dtype=float)
        lats = np.asarray([center["lat"] for center in centers], dtype=float)
    else:
        lons = np.asarray([], dtype=float)
        lats = np.asarray([], dtype=float)

    lon_bounds = lon_range or _infer_bounds(lons) or _infer_data_coord_bounds(data, "lon")
    lat_bounds = lat_range or _infer_bounds(lats) or _infer_data_coord_bounds(data, "lat")

    if lon_bounds is None or lat_bounds is None:
        return {
            "lon": [],
            "lat": [],
            "values": [],
            "metadata": {
                "event_count": len(events),
                "has_location": False,
                "resolution_deg": resolution_value,
                "grid_mode": "native" if resolution_value is None else "binned",
                "normalized": bool(normalize),
            },
        }

    native_grid = None if resolution_value is not None else _native_frequency_grid(data, lon_bounds, lat_bounds)
    if native_grid is not None:
        lon_centers, lat_centers, lon_edges, lat_edges = native_grid
        grid_mode = "native"
    else:
        effective_resolution = resolution_value if resolution_value is not None else 1.0
        lon_edges = _build_edges(lon_bounds, effective_resolution)
        lat_edges = _build_edges(lat_bounds, effective_resolution)
        lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        grid_mode = "binned"

    counts = np.zeros((len(lat_edges) - 1, len(lon_edges) - 1), dtype=float)

    if len(centers) > 0:
        counts, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])

    if normalize and counts.sum() > 0:
        counts = counts / counts.sum()

    ocean_mask = _build_frequency_ocean_mask(
        data=data,
        lat_edges=lat_edges,
        lon_edges=lon_edges,
    )
    if ocean_mask is not None:
        counts = counts.astype(float)
        counts[~ocean_mask] = np.nan

    spacing_metadata = _frequency_grid_spacing_metadata(lon_centers, lat_centers)

    return {
        "lon": lon_centers.tolist(),
        "lat": lat_centers.tolist(),
        "values": counts,
        "metadata": {
            "event_count": len(events),
            "has_location": bool(len(centers) > 0),
            "resolution_deg": resolution_value,
            "grid_mode": grid_mode,
            "frequency_source": "event_centers",
            "grid_spacing": spacing_metadata,
            "normalized": bool(normalize),
            "lon_range": [float(lon_edges[0]), float(lon_edges[-1])],
            "lat_range": [float(lat_edges[0]), float(lat_edges[-1])],
            "ocean_mask_applied": bool(ocean_mask is not None),
            "statistics": _compute_field_statistics(counts),
        },
    }


def _compute_event_mask_frequency_map(
    *,
    event_detection: Dict,
    data: xr.DataArray,
    normalize: bool,
    event_count: int,
) -> Dict:
    result = compute_event_summary_map(
        event_detection=event_detection,
        data=data,
        summary_mode="event_days",
    )
    values = np.asarray(result.get("values"), dtype=float)
    metadata = dict(result.get("metadata") or {})
    denominator = 1.0
    if normalize:
        field = _prepare_summary_field(data, event_detection)
        if "time" in field.dims:
            denominator = float(np.sum(_time_step_days(field["time"].values))) or 1.0
        values = values / denominator
        metadata["units"] = "fraction of analysis period"

    metadata.update(
        {
            "title": "Event frequency map",
            "variable": "event_frequency",
            "summary_mode": "event_frequency",
            "frequency_source": "event_mask",
            "event_count": int(event_count),
            "normalized": bool(normalize),
            "normalization_denominator_days": denominator if normalize else None,
            "statistics": _compute_field_statistics(values),
        }
    )
    return {
        **result,
        "values": values,
        "metadata": metadata,
    }


def compute_event_timeseries_count(
    events: List[Dict],
    weight_by: Literal['count', 'area_km2', 'intensity'] = 'count'
) -> Dict:
    """
    Count detected events per time step.

    Args:
        events: Event list from `detect_*()["events"]`.
        weight_by: Counting strategy. `count` adds one per event, `area_km2`
            sums event area, and `intensity` sums the first available intensity
            metric on each event.

    Returns:
        Timeseries-style count result.
    """
    if not events:
        return {
            "times": [],
            "values": [],
            "metadata": {
                "weight_by": weight_by,
                "event_count": 0,
            },
        }

    key = _detect_time_key(events)
    if key is None:
        total = float(sum(_event_weight(event, weight_by) for event in events))
        return {
            "times": ["all_events"],
            "values": [total],
            "metadata": {
                "weight_by": weight_by,
                "event_count": len(events),
                "time_key": None,
            },
        }

    frame = pd.DataFrame(events)
    frame["weight"] = frame.apply(lambda row: _event_weight(row.to_dict(), weight_by), axis=1)

    if key == "timestamp":
        frame["group"] = pd.to_datetime(frame["timestamp"])
        grouped = frame.groupby("group")["weight"].sum().sort_index()
        times = [ts.isoformat() for ts in grouped.index.to_pydatetime()]
    else:
        frame["group"] = frame[key]
        grouped = frame.groupby("group")["weight"].sum().sort_index()
        times = [str(value) for value in grouped.index.tolist()]

    return {
        "times": times,
        "values": [float(value) for value in grouped.values.tolist()],
        "metadata": {
            "weight_by": weight_by,
            "event_count": len(events),
            "time_key": key,
        },
    }


def compute_event_summary_map(
    event_detection: Dict,
    data: xr.DataArray,
    summary_mode: Literal["burden", "event_days"] = "burden",
) -> Dict:
    """
    Compute a summary map from a threshold-defined event detection result.

    The summary is computed from the persisted event mask returned by the
    detector so the output respects both persistence and minimum-area filters.
    """
    data = materialize_partitioned_xarray(data)

    if summary_mode not in {"burden", "event_days"}:
        raise ValueError("summary_mode must be 'burden' or 'event_days'")
    if "lat" not in data.coords or "lon" not in data.coords:
        raise ValueError("compute_event_summary_map requires lat/lon coordinates")

    event_type = str(event_detection.get("event_type") or "").strip()
    if not event_type:
        raise ValueError("event_detection must include event_type")

    field = _prepare_summary_field(data, event_detection)
    report_phase(
        phase="preparing_event_summary_map",
        message=f"Preparing {event_type} summary map",
        percent=0.05,
        compute_backend="dask" if is_dask_backed(field) else "xarray",
    )
    event_mask = _resolve_event_mask(event_detection, field)
    valid_mask = _horizontal_valid_mask_as_dataarray(field)

    if summary_mode == "event_days":
        contribution = event_mask.astype(float)
    else:
        threshold = _resolve_summary_threshold(event_detection, field)
        if event_type in {"algal_bloom", "heatwave", "eutrophication"}:
            severity = xr.where(field > threshold, field - threshold, 0.0)
        elif event_type in {"hypoxia", "upwelling"}:
            severity = xr.where(field < threshold, threshold - field, 0.0)
        else:
            raise ValueError(f"Unsupported event type for summary map: {event_type}")
        contribution = severity.where(event_mask, 0.0)

    if "time" in field.dims:
        step_days = _time_step_days(field["time"].values)
        summary = (contribution * step_days).sum(dim="time", skipna=True)
    else:
        summary = contribution

    summary = summary.where(valid_mask)
    values = dataarray_to_numpy(
        summary,
        label=f"{event_type} {summary_mode} summary map",
        dtype=float,
        start=0.2,
        end=0.95,
    )
    source_units = str(field.attrs.get("units", "") or "").strip()
    statistics = _compute_field_statistics(values)
    extrema_metadata = _field_extrema_metadata(
        values,
        lat=np.asarray(field["lat"].values, dtype=float),
        lon=np.asarray(field["lon"].values, dtype=float),
    )
    _validate_event_summary_consistency(
        event_detection=event_detection,
        summary_mode=summary_mode,
        statistics=statistics,
    )

    return {
        "lon": field["lon"].values.tolist(),
        "lat": field["lat"].values.tolist(),
        "values": values,
        "metadata": {
            "title": _event_summary_title(event_type, summary_mode),
            "event_type": event_type,
            "summary_mode": summary_mode,
            "variable": _event_summary_variable(event_type, summary_mode),
            "source_variable": str(field.name or ""),
            "units": _event_summary_units(source_units, summary_mode),
            "source_units": source_units,
            "time_range": _event_time_range(field),
            "statistics": statistics,
            **extrema_metadata,
        },
    }


def _infer_bounds(values: np.ndarray) -> Optional[Tuple[float, float]]:
    """Infer grid bounds from event center coordinates."""
    if values.size == 0:
        return None
    return (float(np.floor(values.min())), float(np.ceil(values.max())))


def _normalize_frequency_resolution(value: Optional[float]) -> Optional[float]:
    """Return a positive coarse-grid resolution, or None for native data grid."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "native", "source", "data"}:
        return None
    resolution = float(value)
    if resolution <= 0:
        raise ValueError("resolution_deg must be positive")
    return resolution


def _infer_data_coord_bounds(data: Optional[xr.DataArray], coord: str) -> Optional[Tuple[float, float]]:
    if data is None or coord not in data.coords:
        return None
    values = np.asarray(data[coord].values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return (float(values.min()), float(values.max()))


def _build_edges(bounds: Tuple[float, float], resolution_deg: float) -> np.ndarray:
    """Build closed histogram edges for a regular lon/lat grid."""
    start, end = bounds
    if end <= start:
        end = start + resolution_deg
    n_steps = max(int(np.ceil((end - start) / resolution_deg)), 1)
    return start + np.arange(n_steps + 1, dtype=float) * resolution_deg


def _native_frequency_grid(
    data: Optional[xr.DataArray],
    lon_bounds: Tuple[float, float],
    lat_bounds: Tuple[float, float],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if data is None or "lat" not in data.coords or "lon" not in data.coords:
        return None

    source_lon = np.asarray(data["lon"].values, dtype=float)
    source_lat = np.asarray(data["lat"].values, dtype=float)
    if source_lon.ndim != 1 or source_lat.ndim != 1:
        return None

    lon_centers = _coord_centers_in_bounds(source_lon, lon_bounds)
    lat_centers = _coord_centers_in_bounds(source_lat, lat_bounds)
    if lon_centers.size == 0 or lat_centers.size == 0:
        return None

    lon_edges = _edges_from_centers(lon_centers, fallback_values=source_lon)
    lat_edges = _edges_from_centers(lat_centers, fallback_values=source_lat)
    return lon_centers, lat_centers, lon_edges, lat_edges


def _coord_centers_in_bounds(values: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.asarray([], dtype=float)
    lower, upper = sorted((float(bounds[0]), float(bounds[1])))
    selected = finite[(finite >= lower) & (finite <= upper)]
    return np.unique(np.sort(selected.astype(float)))


def _edges_from_centers(centers: np.ndarray, *, fallback_values: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return np.asarray([], dtype=float)
    if centers.size == 1:
        spacing = _median_positive_spacing(fallback_values) or 1.0
        half = spacing / 2.0
        return np.asarray([centers[0] - half, centers[0] + half], dtype=float)

    diffs = np.diff(centers)
    first = centers[0] - diffs[0] / 2.0
    mids = centers[:-1] + diffs / 2.0
    last = centers[-1] + diffs[-1] / 2.0
    return np.concatenate(([first], mids, [last])).astype(float)


def _median_positive_spacing(values: np.ndarray) -> Optional[float]:
    finite = np.unique(np.sort(np.asarray(values, dtype=float)[np.isfinite(values)]))
    if finite.size < 2:
        return None
    diffs = np.diff(finite)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return None
    return float(np.median(positive))


def _frequency_grid_spacing_metadata(lon_centers: np.ndarray, lat_centers: np.ndarray) -> Dict[str, Optional[float]]:
    return {
        "lon_deg": _median_positive_spacing(lon_centers),
        "lat_deg": _median_positive_spacing(lat_centers),
    }


def _build_frequency_ocean_mask(
    data: Optional[xr.DataArray],
    *,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
) -> Optional[np.ndarray]:
    if data is None or "lat" not in data.coords or "lon" not in data.coords:
        return None

    horizontal_mask = _extract_horizontal_valid_mask(data)
    if horizontal_mask is None:
        return None

    source_lon = np.asarray(data["lon"].values, dtype=float)
    source_lat = np.asarray(data["lat"].values, dtype=float)
    if source_lon.ndim != 1 or source_lat.ndim != 1:
        return None

    lon_grid, lat_grid = np.meshgrid(source_lon, source_lat)
    valid_hist, _, _ = np.histogram2d(
        lat_grid[horizontal_mask],
        lon_grid[horizontal_mask],
        bins=[lat_edges, lon_edges],
    )
    return valid_hist > 0


def _extract_horizontal_valid_mask(data: xr.DataArray) -> Optional[np.ndarray]:
    valid = np.isfinite(data)
    reduce_dims = [dim for dim in data.dims if dim not in {"lat", "lon"}]
    for dim in reduce_dims:
        valid = valid.any(dim=dim)

    mask = dataarray_to_numpy(
        valid,
        label="event horizontal valid mask",
        dtype=bool,
        start=0.06,
        end=0.18,
    )
    if mask.ndim != 2:
        return None
    return mask


def _detect_time_key(events: List[Dict]) -> Optional[str]:
    """Detect the most useful time field available on the event list."""
    sample = events[0]
    if "timestamp" in sample:
        return "timestamp"
    if "time_index" in sample:
        return "time_index"
    return None


def _event_weight(
    event: Dict,
    weight_by: Literal['count', 'area_km2', 'intensity']
) -> float:
    """Return the scalar contribution of one event."""
    if weight_by == "count":
        return 1.0
    if weight_by == "area_km2":
        return float(event.get("area_km2", 0.0))
    return float(_extract_intensity_value(event))


def _extract_intensity_value(event: Dict) -> float:
    """Pick the first available intensity-like metric from an event record."""
    for field in (
        "max_intensity",
        "mean_intensity",
        "intensity",
        "max_gradient",
        "max_vorticity",
        "max_chlorophyll",
        "min_oxygen",
    ):
        if field in event:
            return float(event[field])
    return 0.0


def _prepare_summary_field(data: xr.DataArray, event_detection: Dict) -> xr.DataArray:
    params = _detection_params(event_detection)
    event_type = str(event_detection.get("event_type") or "").strip()
    default_mode = "bottom" if event_type == "hypoxia" else "surface"

    return prepare_event_vertical_field(
        data,
        default_mode=default_mode,
        vertical_mode=str(params.get("vertical_mode") or default_mode),
        depth_value=_normalize_optional_float(params.get("depth_value")),
        depth_range=_normalize_optional_range(params.get("depth_range")),
        depth_aggregation=str(params.get("depth_aggregation") or "mean"),
    )


def _resolve_event_mask(event_detection: Dict, field: xr.DataArray) -> xr.DataArray:
    raw_mask = event_detection.get("event_mask")
    if raw_mask is None:
        raise ValueError("event_detection must include event_mask for summary-map generation")

    mask = np.asarray(raw_mask, dtype=bool)
    if tuple(mask.shape) != tuple(field.shape):
        raise ValueError("event_mask shape does not match the prepared event field")

    return xr.DataArray(
        mask,
        coords={dim: field.coords[dim] for dim in field.dims},
        dims=field.dims,
    )


def _resolve_summary_threshold(event_detection: Dict, field: xr.DataArray) -> xr.DataArray | float:
    raw_threshold = event_detection.get("threshold_field")
    if raw_threshold is not None:
        threshold_array = np.asarray(raw_threshold, dtype=float)
        if threshold_array.ndim == 0:
            return float(threshold_array)
        expected_shape = tuple(field.sel(time=field.time[0]).shape) if "time" in field.dims else tuple(field.shape)
        if tuple(threshold_array.shape) != expected_shape:
            raise ValueError("threshold_field shape does not match the horizontal event field")
        coords = {dim: field.coords[dim] for dim in field.dims if dim != "time"}
        dims = tuple(dim for dim in field.dims if dim != "time")
        return xr.DataArray(threshold_array, coords=coords, dims=dims)

    params = _detection_params(event_detection)
    event_type = str(event_detection.get("event_type") or "").strip()

    if event_type == "hypoxia":
        threshold = params.get("oxygen_threshold")
        if threshold is None:
            raise ValueError("Hypoxia summary requires oxygen_threshold")
        return float(threshold)

    if "time" not in field.dims:
        raise ValueError("Percentile-based event summaries require a time dimension")

    if event_type == "eutrophication":
        percentile = params.get("chlorophyll_percentile", 90)
    else:
        percentile = params.get("percentile_threshold")

    if percentile is None:
        raise ValueError("Percentile-based event summary requires a percentile threshold")

    field = ensure_time_reduction_chunks(field, label=f"{event_type} summary")
    threshold = field.quantile(float(percentile) / 100.0, dim="time")
    if "quantile" in threshold.dims:
        threshold = threshold.squeeze("quantile", drop=True)
    return threshold


def _detection_params(event_detection: Dict) -> Dict:
    statistics = event_detection.get("statistics")
    if isinstance(statistics, dict):
        detection_params = statistics.get("detection_params")
        if isinstance(detection_params, dict):
            return detection_params
    return {}


def _horizontal_valid_mask_as_dataarray(data: xr.DataArray) -> xr.DataArray:
    mask = _extract_horizontal_valid_mask(data)
    if mask is None:
        raise ValueError("Unable to infer horizontal valid mask from event field")
    return xr.DataArray(
        mask,
        coords={"lat": data["lat"], "lon": data["lon"]},
        dims=("lat", "lon"),
    )


def _time_step_days(time_values: np.ndarray) -> xr.DataArray:
    timestamps = pd.to_datetime(time_values)
    if len(timestamps) == 0:
        return xr.DataArray([], dims=("time",))
    if len(timestamps) == 1:
        step_days = np.array([1.0], dtype=float)
    else:
        deltas = np.diff(timestamps.to_numpy()).astype("timedelta64[s]").astype(float) / 86400.0
        valid = deltas[np.isfinite(deltas) & (deltas > 0)]
        fallback = float(valid[-1]) if valid.size else 1.0
        step_days = np.empty(len(timestamps), dtype=float)
        step_days[:-1] = np.where(np.isfinite(deltas) & (deltas > 0), deltas, fallback)
        step_days[-1] = fallback
    return xr.DataArray(step_days, coords={"time": time_values}, dims=("time",))


def _event_summary_title(event_type: str, summary_mode: str) -> str:
    titles = {
        "algal_bloom": {
            "burden": "Bloom Chlorophyll Burden",
            "event_days": "Bloom Event Days",
        },
        "heatwave": {
            "burden": "Marine Heatwave Burden",
            "event_days": "Marine Heatwave Days",
        },
        "hypoxia": {
            "burden": "Hypoxia Oxygen Deficit Burden",
            "event_days": "Hypoxic Days",
        },
        "upwelling": {
            "burden": "Upwelling Cold-Anomaly Burden",
            "event_days": "Upwelling Days",
        },
        "eutrophication": {
            "burden": "Eutrophication Chlorophyll Burden",
            "event_days": "Eutrophic Days",
        },
    }
    return titles.get(event_type, {}).get(summary_mode, f"{event_type.title()} {summary_mode.replace('_', ' ').title()}")


def _event_summary_variable(event_type: str, summary_mode: str) -> str:
    if summary_mode == "event_days":
        return "Event Days"
    variables = {
        "algal_bloom": "Chlorophyll Burden",
        "heatwave": "Heatwave Burden",
        "hypoxia": "Oxygen Deficit Burden",
        "upwelling": "Cold-Anomaly Burden",
        "eutrophication": "Chlorophyll Burden",
    }
    return variables.get(event_type, "Event Burden")


def _event_summary_units(source_units: str, summary_mode: str) -> str:
    if summary_mode == "event_days":
        return "days"
    if not source_units:
        return "native-units·days"
    return f"{source_units}·days"


def _event_time_range(field: xr.DataArray) -> Optional[List[str]]:
    if "time" not in field.coords or field.sizes.get("time", 0) == 0:
        return None
    return [str(field["time"].values[0]), str(field["time"].values[-1])]


def _compute_field_statistics(values: np.ndarray) -> Dict[str, object]:
    values = as_numeric_array(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "total": 0.0,
            "n_nonzero": 0,
            "nonzero_fraction": 0.0,
        }
    n_nonzero = int(np.count_nonzero(np.abs(finite) > 0.0))
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
        "total": float(np.sum(finite)),
        "n_nonzero": n_nonzero,
        "nonzero_fraction": float(n_nonzero / finite.size),
    }


def _field_extrema_metadata(values: np.ndarray, *, lat: np.ndarray, lon: np.ndarray) -> Dict[str, object]:
    values = as_numeric_array(values)
    if values.ndim != 2 or lat.ndim != 1 or lon.ndim != 1:
        return {}
    if values.shape != (lat.size, lon.size) or not np.any(np.isfinite(values)):
        return {}
    if not np.any(np.abs(values[np.isfinite(values)]) > 0.0):
        return {}

    max_index = tuple(int(idx) for idx in np.unravel_index(int(np.nanargmax(values)), values.shape))
    min_index = tuple(int(idx) for idx in np.unravel_index(int(np.nanargmin(values)), values.shape))
    max_location = {
        "lon": float(lon[max_index[1]]),
        "lat": float(lat[max_index[0]]),
        "value": float(values[max_index]),
    }
    min_location = {
        "lon": float(lon[min_index[1]]),
        "lat": float(lat[min_index[0]]),
        "value": float(values[min_index]),
    }
    metadata: Dict[str, object] = {
        "max_location": max_location,
        "min_location": min_location,
    }
    top_hotspots = _top_spatial_hotspots(values, lat=lat, lon=lon, limit=4)
    if top_hotspots:
        metadata["top_hotspots"] = top_hotspots
    hotspot_label = _label_ocean_region(max_location["lon"], max_location["lat"])
    coldspot_label = _label_ocean_region(min_location["lon"], min_location["lat"])
    if hotspot_label:
        metadata["hotspot_region_label"] = hotspot_label
    if coldspot_label:
        metadata["coldspot_region_label"] = coldspot_label
    return metadata


def _top_spatial_hotspots(
    values: np.ndarray,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    limit: int,
) -> List[Dict[str, object]]:
    values = as_numeric_array(values)
    if values.ndim != 2 or lat.ndim != 1 or lon.ndim != 1:
        return []
    if values.shape != (lat.size, lon.size):
        return []

    finite_mask = np.isfinite(values) & (np.abs(values) > 0.0)
    if not np.any(finite_mask):
        return []

    candidate_indices = np.argwhere(finite_mask)
    candidate_values = values[finite_mask]
    order = np.argsort(candidate_values)[::-1]
    min_separation_deg = 1.0
    hotspots: List[Dict[str, object]] = []

    for ordered_index in order:
        lat_idx, lon_idx = (int(idx) for idx in candidate_indices[int(ordered_index)])
        hotspot_lon = float(lon[lon_idx])
        hotspot_lat = float(lat[lat_idx])
        if any(
            float(np.hypot(hotspot_lon - float(existing["lon"]), hotspot_lat - float(existing["lat"])))
            < min_separation_deg
            for existing in hotspots
        ):
            continue
        label = _label_ocean_region(hotspot_lon, hotspot_lat)
        hotspot: Dict[str, object] = {
            "rank": len(hotspots) + 1,
            "lon": hotspot_lon,
            "lat": hotspot_lat,
            "value": float(values[lat_idx, lon_idx]),
        }
        if label:
            hotspot["label"] = label
        hotspots.append(hotspot)
        if len(hotspots) >= limit:
            break
    return hotspots


def _validate_event_summary_consistency(
    *,
    event_detection: Dict,
    summary_mode: str,
    statistics: Dict[str, float],
) -> None:
    event_count = _event_detection_count(event_detection)
    raw_mask = event_detection.get("event_mask")
    mask_nonzero = False
    if raw_mask is not None:
        try:
            mask_nonzero = bool(np.count_nonzero(np.asarray(raw_mask, dtype=bool)))
        except Exception:
            mask_nonzero = False
    has_detected_event = event_count > 0 or mask_nonzero
    if not has_detected_event:
        return
    n_nonzero = int(statistics.get("n_nonzero") or 0)
    total = float(statistics.get("total") or 0.0)
    if summary_mode == "event_days" and n_nonzero == 0:
        raise ValueError(
            "Event summary map is empty despite nonzero event detection. "
            "Check event_mask/result references and summary_mode wiring."
        )
    if summary_mode == "burden" and n_nonzero == 0 and total == 0.0:
        statistics["consistency_warning"] = (
            "Detected events exist, but burden map is zero. This can occur with zero severity, "
            "but usually indicates a threshold or result-reference mismatch."
        )


def _event_detection_count(event_detection: Dict) -> int:
    for key in ("event_count", "total_count", "n_events"):
        value = event_detection.get(key)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float) and np.isfinite(value):
            return int(value)
    statistics = event_detection.get("statistics")
    if isinstance(statistics, dict):
        for key in ("total_count", "event_count", "n_events"):
            value = statistics.get(key)
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, float) and np.isfinite(value):
                return int(value)
    events = event_detection.get("events")
    if isinstance(events, list):
        return len(events)
    return 0


def _label_ocean_region(lon: float, lat: float) -> Optional[str]:
    if not np.isfinite(lon) or not np.isfinite(lat):
        return None
    if 112.0 <= lon <= 116.5 and 20.0 <= lat <= 24.0:
        return "Pearl River Estuary / northern South China Sea shelf"
    if 105.0 <= lon <= 110.5 and 17.0 <= lat <= 22.0:
        return "Gulf of Tonkin / western northern South China Sea"
    if 119.0 <= lon <= 123.5 and 22.0 <= lat <= 26.5:
        return "Taiwan Strait / Luzon Strait approach"
    if 108.0 <= lon <= 121.5 and 5.0 <= lat <= 23.5:
        return "South China Sea shelf/basin"
    if 118.0 <= lon <= 124.5 and 24.0 <= lat <= 32.5:
        return "East China Sea shelf"
    if 119.0 <= lon <= 124.5 and 34.0 <= lat <= 40.5:
        return "Yellow Sea / Bohai Sea shelf"
    return None


def _normalize_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _normalize_optional_range(value: object) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError("depth_range must be a two-element list or tuple")
