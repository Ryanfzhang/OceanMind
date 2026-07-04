"""
时间序列分析工具

提供时间序列提取、统计、变换等功能
"""

import xarray as xr
import numpy as np
from typing import Dict, Tuple, Optional, List, Literal

from domain.ocean.data_access.partitioned import (
    find_partitioned_values,
    materialize_partitioned_xarray,
)
from domain.ocean.data_access.load import get_depth_dim, _normalize_depth_range, resolve_pending_bottom_selection
from domain.ocean.dask_utils import dataarray_to_numpy, is_dask_backed, report_phase


def extract_regional_mean(
    data: xr.DataArray,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    depth_range: Optional[Tuple[float, float]] = None,
    depth_aggregation: Literal['mean', 'max', 'min', 'surface'] = 'mean',
) -> Dict:
    """
    提取区域平均时间序列

    Args:
        data: 输入数据（必须包含time维度）
        lon_range: 经度范围 (min, max)
        lat_range: 纬度范围 (min, max)
        depth_range: 可选深度范围
        depth_aggregation: 深度聚合方式；默认对 depth 维取平均

    Returns:
        字典，包含：
        - times: 时间列表
        - values: 数值列表
        - metadata: 元数据（统计信息、区域等）

    Example:
        >>> ts = extract_regional_mean(
        ...     data,
        ...     lon_range=(110, 120),
        ...     lat_range=(18, 23)
        ... )
        >>> print(ts['metadata']['statistics'])
    """
    if find_partitioned_values((data,)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="extract_regional_mean",
            tool_func=extract_regional_mean,
            params={
                "data": data,
                "lon_range": lon_range,
                "lat_range": lat_range,
                "depth_range": depth_range,
                "depth_aggregation": depth_aggregation,
            },
        )
    data = materialize_partitioned_xarray(data)
    data = resolve_pending_bottom_selection(
        data,
        label="regional mean bottom layer",
        start=0.05,
        end=0.15,
    )

    # 空间选择
    subset = data.sel(
        lon=slice(*lon_range),
        lat=slice(*lat_range)
    )
    subset = _aggregate_depth(subset, depth_range, depth_aggregation)

    # 计算区域平均
    report_phase(
        phase="preparing_timeseries",
        message="Preparing regional mean time series",
        percent=0.05,
        compute_backend="dask" if is_dask_backed(subset) else "xarray",
    )
    timeseries = subset.mean(dim=['lon', 'lat'])
    timeseries = _normalize_timeseries_array(timeseries)

    # 提取数据
    times = [str(t) for t in timeseries.time.values]
    value_array = dataarray_to_numpy(
        timeseries,
        label="regional mean time series",
        dtype=float,
        start=0.2,
        end=0.9,
    )
    values = value_array.tolist()
    report_phase(
        phase="timeseries_complete",
        message="Regional mean time series complete",
        percent=1.0,
        compute_backend="numpy",
    )

    # 计算统计信息
    statistics = {
        'mean': float(np.mean(value_array)),
        'std': float(np.std(value_array)),
        'min': float(np.min(value_array)),
        'max': float(np.max(value_array)),
        'median': float(np.median(value_array)),
        'n_points': len(values)
    }

    return {
        'times': times,
        'values': values,
        'metadata': {
            'variable': data.name or 'unknown',
            'unit': data.attrs.get('units', ''),
            'region': {
                'lon_range': lon_range,
                'lat_range': lat_range
            },
            'depth_range': list(depth_range) if depth_range is not None else None,
            'depth_aggregation': depth_aggregation if get_depth_dim(data) is not None else None,
            'statistics': statistics
        }
    }


def extract_point_timeseries(
    data: xr.DataArray,
    lon: float,
    lat: float,
    method: str = 'nearest',
    depth_range: Optional[Tuple[float, float]] = None,
    depth_aggregation: Literal['mean', 'max', 'min', 'surface'] = 'mean',
) -> Dict:
    """
    提取点位时间序列

    Args:
        data: 输入数据
        lon: 经度
        lat: 纬度
        method: 插值方法（'nearest'或'linear'）
        depth_range: 可选深度范围
        depth_aggregation: 深度聚合方式；默认对 depth 维取平均

    Returns:
        时间序列字典

    Example:
        >>> ts = extract_point_timeseries(data, lon=115.5, lat=20.5)
    """
    if find_partitioned_values((data,)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="extract_point_timeseries",
            tool_func=extract_point_timeseries,
            params={
                "data": data,
                "lon": lon,
                "lat": lat,
                "method": method,
                "depth_range": depth_range,
                "depth_aggregation": depth_aggregation,
            },
        )
    data = materialize_partitioned_xarray(data)
    data = resolve_pending_bottom_selection(
        data,
        label="point time series bottom layer",
        start=0.05,
        end=0.15,
    )

    # 选择点位
    point_data = data.sel(lon=lon, lat=lat, method=method)
    selected_location = {
        'lon': _coord_scalar(point_data, 'lon'),
        'lat': _coord_scalar(point_data, 'lat'),
    }
    point_data = _aggregate_depth(point_data, depth_range, depth_aggregation)

    point_data = _normalize_timeseries_array(point_data)

    report_phase(
        phase="preparing_timeseries",
        message="Preparing point time series",
        percent=0.05,
        compute_backend="dask" if is_dask_backed(point_data) else "xarray",
    )
    times = [str(t) for t in point_data.time.values]
    values = dataarray_to_numpy(
        point_data,
        label="point time series",
        dtype=float,
        start=0.2,
        end=0.9,
    ).tolist()
    report_phase(
        phase="timeseries_complete",
        message="Point time series complete",
        percent=1.0,
        compute_backend="numpy",
    )

    return {
        'times': times,
        'values': values,
        'metadata': {
            'variable': data.name or 'unknown',
            'unit': data.attrs.get('units', ''),
            'location': selected_location,
            'requested_location': {'lon': lon, 'lat': lat},
            'selected_location': selected_location,
            'method': method,
            'depth_range': list(depth_range) if depth_range is not None else None,
            'depth_aggregation': depth_aggregation if get_depth_dim(data) is not None else None,
        }
    }


def compute_mixed_layer_mean(
    data: xr.DataArray,
    mixed_layer_depth: xr.DataArray,
) -> xr.DataArray:
    """
    对混合层以上做垂向平均。

    对输入场的每个时间/空间点，使用对应的混合层深度在表层到 MLD 之间
    做平均，返回去掉 depth 维后的场。

    Args:
        data: 待平均的 3D/4D 数据，需要 depth 维
        mixed_layer_depth: 混合层深度场，通常来自 identify_mixed_layer_depth

    Returns:
        混合层平均后的 DataArray
    """
    if find_partitioned_values((data, mixed_layer_depth)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_mixed_layer_mean",
            tool_func=compute_mixed_layer_mean,
            params={"data": data, "mixed_layer_depth": mixed_layer_depth},
        )
    data = materialize_partitioned_xarray(data)
    mixed_layer_depth = materialize_partitioned_xarray(mixed_layer_depth)

    mixed_layer_mean = compute_layer_mean(
        data=data,
        lower_bound_field=mixed_layer_depth,
    )
    mixed_layer_mean.name = f"{data.name or 'field'}_mixed_layer_mean"
    mixed_layer_mean.attrs = {
        **mixed_layer_mean.attrs,
        "aggregation": "mixed_layer_mean",
        "mixed_layer_depth_source": mixed_layer_depth.name or "mixed_layer_depth",
    }
    return mixed_layer_mean


def compute_layer_mean(
    data: xr.DataArray,
    upper_bound_value: Optional[float] = None,
    lower_bound_value: Optional[float] = None,
    upper_bound_field: Optional[xr.DataArray] = None,
    lower_bound_field: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    在给定上下边界之间做垂向平均。

    Args:
        data: 待平均的 3D/4D 数据，需要 depth 维
        upper_bound_value: 固定上边界深度；不传时默认为海表 0 m
        lower_bound_value: 固定下边界深度
        upper_bound_field: 动态上边界深度场
        lower_bound_field: 动态下边界深度场

    Returns:
        去掉 depth 维后的层平均 DataArray
    """
    if find_partitioned_values((data, upper_bound_field, lower_bound_field)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_layer_mean",
            tool_func=compute_layer_mean,
            params={
                "data": data,
                "upper_bound_value": upper_bound_value,
                "lower_bound_value": lower_bound_value,
                "upper_bound_field": upper_bound_field,
                "lower_bound_field": lower_bound_field,
            },
        )
    data = materialize_partitioned_xarray(data)
    upper_bound_field = materialize_partitioned_xarray(upper_bound_field)
    lower_bound_field = materialize_partitioned_xarray(lower_bound_field)

    depth_dim = get_depth_dim(data)
    if depth_dim is None:
        raise ValueError("compute_layer_mean requires a depth dimension")

    if upper_bound_value is not None and upper_bound_field is not None:
        raise ValueError("Provide either upper_bound_value or upper_bound_field, not both")
    if lower_bound_value is not None and lower_bound_field is not None:
        raise ValueError("Provide either lower_bound_value or lower_bound_field, not both")
    if lower_bound_value is None and lower_bound_field is None:
        raise ValueError("compute_layer_mean requires a lower boundary")

    upper_bound, upper_source = _resolve_layer_bound(
        data=data,
        bound_value=upper_bound_value,
        bound_field=upper_bound_field,
        default_value=0.0,
        label="upper",
    )
    lower_bound, lower_source = _resolve_layer_bound(
        data=data,
        bound_value=lower_bound_value,
        bound_field=lower_bound_field,
        default_value=None,
        label="lower",
    )

    depth_coord = xr.DataArray(
        np.abs(data[depth_dim].values),
        coords={depth_dim: data[depth_dim].values},
        dims=(depth_dim,),
    )
    depth_broadcast, upper_broadcast = xr.broadcast(depth_coord, upper_bound)
    _, lower_broadcast = xr.broadcast(depth_coord, lower_bound)

    lower_mask = xr.apply_ufunc(
        np.minimum,
        upper_broadcast,
        lower_broadcast,
        dask="parallelized",
        output_dtypes=[float],
    )
    upper_mask = xr.apply_ufunc(
        np.maximum,
        upper_broadcast,
        lower_broadcast,
        dask="parallelized",
        output_dtypes=[float],
    )
    mask = (depth_broadcast >= lower_mask) & (depth_broadcast <= upper_mask)

    layer_mean = data.where(mask).mean(dim=depth_dim, skipna=True)
    layer_mean.name = f"{data.name or 'field'}_layer_mean"
    layer_mean.attrs = {
        **data.attrs,
        "long_name": f"Layer mean of {data.name or 'field'}",
        "aggregation": "layer_mean",
        "upper_bound_source": upper_source,
        "lower_bound_source": lower_source,
    }
    return layer_mean


def _aggregate_depth(
    data: xr.DataArray,
    depth_range: Optional[Tuple[float, float]],
    depth_aggregation: str,
) -> xr.DataArray:
    depth_dim = get_depth_dim(data)
    if depth_dim is None:
        return data

    field = data
    if depth_range is not None:
        normalized_range = _normalize_depth_range(np.asarray(field[depth_dim].values, dtype=float), depth_range)
        field = field.sel({depth_dim: _build_coord_slice(field[depth_dim].values, normalized_range)})

    if depth_aggregation == 'surface':
        return field.isel({depth_dim: 0})
    if depth_aggregation == 'mean':
        return field.mean(dim=depth_dim, skipna=True)
    if depth_aggregation == 'max':
        return field.max(dim=depth_dim, skipna=True)
    if depth_aggregation == 'min':
        return field.min(dim=depth_dim, skipna=True)

    raise ValueError(f"Unknown depth_aggregation: {depth_aggregation}")


def _normalize_timeseries_array(data: xr.DataArray) -> xr.DataArray:
    """
    Ensure a time-series result is truly one-dimensional over time.

    Singleton non-time dimensions are squeezed automatically. If any non-time
    dimension still has more than one element, the caller must constrain the
    data first, for example by selecting a depth level before computing the
    time series.
    """
    for dim in list(data.dims):
        if dim == 'time':
            continue
        if data.sizes[dim] == 1:
            data = data.squeeze(dim, drop=True)

    remaining = [dim for dim in data.dims if dim != 'time']
    if remaining:
        raise ValueError(
            f"Time-series extraction requires only a time dimension after selection; remaining dims: {remaining}"
        )

    return data


def _coord_scalar(data: xr.DataArray, coord_name: str) -> Optional[float]:
    if coord_name not in data.coords:
        return None
    values = np.asarray(data[coord_name].values, dtype=float)
    if values.size == 0:
        return None
    return float(values.reshape(-1)[0])


def _resolve_layer_bound(
    data: xr.DataArray,
    bound_value: Optional[float],
    bound_field: Optional[xr.DataArray],
    default_value: Optional[float],
    label: str,
) -> Tuple[xr.DataArray, str]:
    if bound_field is not None:
        if not isinstance(bound_field, xr.DataArray):
            raise ValueError(f"{label}_bound_field must be an xarray DataArray")
        depth_dim = get_depth_dim(bound_field)
        if depth_dim is not None:
            raise ValueError(f"{label}_bound_field must not contain a depth dimension")
        field = xr.apply_ufunc(
            np.abs,
            bound_field,
            dask="parallelized",
            output_dtypes=[float],
        )
        source = f"field:{bound_field.name or label + '_bound'}"
        return field, source

    if bound_value is None:
        if default_value is None:
            raise ValueError(f"{label}_bound is required")
        bound_value = default_value
        source = "surface" if float(default_value) == 0.0 else f"fixed_depth:{float(default_value)}"
    else:
        source = f"fixed_depth:{float(bound_value)}"

    scalar = xr.DataArray(np.abs(float(bound_value)))
    return scalar, source


def _build_coord_slice(values, coord_range: Tuple[float, float]) -> slice:
    start, end = coord_range
    values = np.asarray(values)
    ascending = values[0] <= values[-1]
    if ascending:
        return slice(min(start, end), max(start, end))
    return slice(max(start, end), min(start, end))


def compute_climatology(
    timeseries: Dict,
    period: str = 'monthly'
) -> Dict:
    """
    计算气候态

    Args:
        timeseries: extract_regional_mean或extract_point_timeseries的输出
        period: 'monthly'（月气候态）或'seasonal'（季节气候态）

    Returns:
        气候态数据

    Example:
        >>> clim = compute_climatology(timeseries, period='monthly')
        >>> print(clim['values'])  # 12个月的平均值
    """
    import pandas as pd

    # 转换为DataFrame
    df = pd.DataFrame({
        'time': pd.to_datetime(timeseries['times']),
        'value': timeseries['values']
    })

    # 计算气候态
    if period == 'monthly':
        clim = df.groupby(df['time'].dt.month)['value'].mean()
        labels = clim.index.tolist()
    elif period == 'seasonal':
        clim = df.groupby(df['time'].dt.quarter)['value'].mean()
        labels = clim.index.tolist()
    else:
        raise ValueError(f"Unknown period: {period}")

    return {
        'period': period,
        'values': clim.tolist(),
        'labels': labels,
        'metadata': {
            'source_variable': timeseries['metadata'].get('variable'),
            'n_years': len(df['time'].dt.year.unique())
        }
    }


def compute_anomaly(
    timeseries: Dict,
    climatology: Dict
) -> Dict:
    """
    计算异常（去除气候态）

    Args:
        timeseries: 原始时间序列
        climatology: 气候态（compute_climatology的输出）

    Returns:
        异常时间序列

    Example:
        >>> anomaly = compute_anomaly(timeseries, climatology)
    """
    import pandas as pd

    # 创建DataFrame
    df = pd.DataFrame({
        'time': pd.to_datetime(timeseries['times']),
        'value': timeseries['values']
    })

    # 根据气候态类型去除季节循环
    if climatology['period'] == 'monthly':
        df['month'] = df['time'].dt.month
        clim_dict = dict(zip(climatology['labels'], climatology['values']))
        df['climatology'] = df['month'].map(clim_dict)

    elif climatology['period'] == 'seasonal':
        df['quarter'] = df['time'].dt.quarter
        clim_dict = dict(zip(climatology['labels'], climatology['values']))
        df['climatology'] = df['quarter'].map(clim_dict)

    # 计算异常
    df['anomaly'] = df['value'] - df['climatology']

    return {
        'times': timeseries['times'],
        'values': df['anomaly'].tolist(),
        'metadata': {
            **timeseries['metadata'],
            'is_anomaly': True,
            'climatology_period': climatology['period']
        }
    }


def compute_trend(
    timeseries: Dict,
    method: str = 'linear',
    confidence_level: float = 0.95
) -> Dict:
    """
    计算时间序列趋势

    Args:
        timeseries: 时间序列（通常是异常序列）
        method: 'linear'（线性回归）或'mann_kendall'（Mann-Kendall检验）
        confidence_level: 置信水平

    Returns:
        趋势分析结果

    Example:
        >>> trend = compute_trend(anomaly, method='linear')
        >>> print(f"Trend: {trend['slope']:.4f} per year")
    """
    import pandas as pd
    from scipy import stats

    df = pd.DataFrame({
        'time': pd.to_datetime(timeseries['times'], errors='coerce'),
        'value': pd.to_numeric(pd.Series(timeseries['values']), errors='coerce')
    })
    source_metadata = timeseries.get('metadata', {}) if isinstance(timeseries.get('metadata'), dict) else {}
    source_times = [str(t) for t in timeseries.get('times', [])]
    source_values = [
        float(v) if np.isfinite(float(v)) else np.nan
        for v in np.asarray(timeseries.get('values', []), dtype=float).reshape(-1)
    ]

    # 转换为数值时间（年）
    df['year'] = df['time'].dt.year + (df['time'].dt.dayofyear - 1) / 365.25

    if method == 'linear':
        # 线性回归只使用有限值；缺测点保留在源序列里，但不参与拟合。
        x = df['year'].to_numpy(dtype=float)
        y = df['value'].to_numpy(dtype=float)
        valid_mask = np.isfinite(x) & np.isfinite(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        n_valid = int(x_valid.size)

        if n_valid >= 2 and np.unique(x_valid).size >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
            trend_line = (slope * x + intercept).tolist()
        else:
            slope = intercept = r_value = p_value = std_err = np.nan
            trend_line = [np.nan for _ in x]

        # 判断显著性
        is_significant = bool(np.isfinite(p_value) and p_value < (1 - confidence_level))

        return {
            'method': 'linear',
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'is_significant': is_significant,
            'confidence_level': confidence_level,
            'trend_line': trend_line,
            'times': source_times,
            'values': source_values,
            'n_points': len(x),
            'n_valid_points': n_valid,
            'n_missing_points': int(len(x) - n_valid),
            'metadata': {
                **source_metadata,
                'source': 'compute_trend',
                'source_variable': source_metadata.get('variable') or source_metadata.get('source_variable'),
                'time_range': [source_times[0], source_times[-1]] if source_times else None,
            }
        }

    else:
        raise ValueError(f"Unknown method: {method}")


def resample_timeseries(
    timeseries: Dict,
    freq: str = 'MS',
    method: str = 'mean'
) -> Dict:
    """
    重采样时间序列

    Args:
        timeseries: 输入时间序列
        freq: 目标频率（'MS'=月初,'D'=日,'Y'=年等）
        method: 聚合方法（'mean','sum','min','max'）

    Returns:
        重采样后的时间序列

    Example:
        >>> monthly = resample_timeseries(daily_ts, freq='MS', method='mean')
    """
    import pandas as pd

    df = pd.DataFrame({
        'time': pd.to_datetime(timeseries['times']),
        'value': timeseries['values']
    })

    df = df.set_index('time')

    # 重采样
    if method == 'mean':
        resampled = df.resample(freq).mean()
    elif method == 'sum':
        resampled = df.resample(freq).sum()
    elif method == 'min':
        resampled = df.resample(freq).min()
    elif method == 'max':
        resampled = df.resample(freq).max()
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        'times': [str(t) for t in resampled.index],
        'values': resampled['value'].tolist(),
        'metadata': {
            **timeseries['metadata'],
            'resampled': True,
            'frequency': freq,
            'method': method
        }
    }
