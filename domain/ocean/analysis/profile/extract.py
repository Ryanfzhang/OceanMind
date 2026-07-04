"""
垂直剖面分析工具

提取和分析垂直剖面数据
"""

import numpy as np
import xarray as xr
from typing import Dict, Tuple, Optional, Literal

from domain.ocean.data_access.partitioned import materialize_partitioned_xarray
from domain.ocean.dask_utils import dataarray_to_numpy, is_dask_backed, report_phase


def _ensure_gradient_depth_chunks(data: xr.DataArray, depth_dim: str = "depth") -> xr.DataArray:
    """Dask gradient needs every differentiated chunk to contain enough points."""
    chunks = getattr(data, "chunks", None)
    if not chunks or depth_dim not in data.dims:
        return data

    axis = data.get_axis_num(depth_dim)
    depth_chunks = chunks[axis]
    if depth_chunks and min(depth_chunks) <= 2:
        return data.chunk({depth_dim: -1})
    return data


def extract_vertical_profile(
    data: xr.DataArray,
    lon: float,
    lat: float,
    method: str = 'nearest'
) -> Dict:
    """
    提取垂直剖面

    在指定位置提取深度-变量剖面

    Args:
        data: 输入数据（需要depth维度）
        lon: 经度
        lat: 纬度
        method: 插值方法 ('nearest', 'linear')

    Returns:
        垂直剖面字典

    Example:
        >>> temp = ds['temp']
        >>> profile = extract_vertical_profile(temp, lon=125.0, lat=30.0)
        >>> print(f"Profile depth range: {profile['depth_range']}")
    """
    data = materialize_partitioned_xarray(data)

    if 'depth' not in data.dims:
        raise ValueError("Data must have depth dimension")

    # 选择位置
    if method == 'nearest':
        point_data = data.sel(lon=lon, lat=lat, method='nearest')
    else:
        point_data = data.interp(lon=lon, lat=lat, method=method)

    # 获取实际位置
    actual_lon = float(point_data.lon.values)
    actual_lat = float(point_data.lat.values)

    # 如果有时间维度，取第一个时间步
    if 'time' in point_data.dims:
        point_data = point_data.isel(time=0)

    # 提取深度和值
    depth_values = point_data.depth.values
    report_phase(
        phase="extracting_vertical_profile",
        message="Extracting vertical profile",
        percent=0.1,
        compute_backend="dask" if is_dask_backed(point_data) else "xarray",
    )
    profile_values = dataarray_to_numpy(
        point_data,
        label="vertical profile",
        dtype=float,
        start=0.2,
        end=0.9,
    )

    # 过滤有效值，同时去掉 9999 这类哑元深度
    valid_mask = (
        ~np.isnan(profile_values)
        & np.isfinite(depth_values)
        & (np.abs(depth_values) < 9000)
    )
    valid_depths = depth_values[valid_mask]
    valid_values = profile_values[valid_mask]

    if len(valid_depths) == 0:
        raise ValueError(f"No valid data at ({lon}, {lat}). Point may be on land.")

    # 统计信息
    statistics = {
        'min': float(np.min(valid_values)),
        'max': float(np.max(valid_values)),
        'mean': float(np.mean(valid_values)),
        'std': float(np.std(valid_values))
    }
    report_phase(
        phase="vertical_profile_complete",
        message="Vertical profile complete",
        percent=1.0,
        compute_backend="numpy",
    )

    return {
        'depth': valid_depths.tolist(),
        'values': valid_values.tolist(),
        'variable': data.name or 'unknown',
        'point': {'lon': actual_lon, 'lat': actual_lat},
        'requested_point': {'lon': lon, 'lat': lat},
        'n_levels': len(valid_depths),
        'depth_range': [float(valid_depths.min()), float(valid_depths.max())],
        'statistics': statistics
    }


def identify_mixed_layer_depth(
    density: xr.DataArray,
    threshold: float = 0.03
) -> xr.DataArray:
    """
    识别混合层深度(MLD)

    基于密度差阈值方法：找到第一个与表层密度差超过阈值的深度

    Args:
        density: 密度场（需要depth维度）
        threshold: 密度差阈值（kg/m³），默认0.03

    Returns:
        混合层深度场

    Example:
        >>> density = ds['density']
        >>> mld = identify_mixed_layer_depth(density, threshold=0.03)
        >>> print(f"Mean MLD: {mld.mean().values:.1f} m")
    """
    density = materialize_partitioned_xarray(density)

    if 'depth' not in density.dims:
        raise ValueError("Density data must have depth dimension")

    density = _ensure_gradient_depth_chunks(density)
    depth_values = density.depth.values

    # 计算与表层的密度差
    surface_density = density.isel(depth=0)
    density_diff = np.abs(density - surface_density)

    # 找到第一个超过阈值的深度
    def find_mld(diff_profile):
        """找到第一个超过阈值的深度"""
        exceeded = diff_profile > threshold
        if np.any(exceeded):
            idx = np.where(exceeded)[0][0]
            return depth_values[idx]
        return np.nan

    # 沿深度维度应用
    mld = xr.apply_ufunc(
        find_mld,
        density_diff,
        input_core_dims=[['depth']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={'allow_rechunk': True},
    )
    mld.name = "mixed_layer_depth"

    mld.attrs = {
        'long_name': 'Mixed Layer Depth',
        'units': 'm',
        'method': 'density_threshold',
        'threshold': f'{threshold} kg/m³',
        'feature': 'mixed_layer',
    }
    report_phase(
        phase="vertical_structure_ready",
        message="Prepared mixed-layer depth lazy field",
        percent=1.0,
        compute_backend="dask" if is_dask_backed(mld) else "xarray",
    )

    return mld


def identify_thermocline_depth(temp: xr.DataArray) -> xr.DataArray:
    """
    识别温跃层深度

    找到温度垂直梯度最大的深度

    Args:
        temp: 温度场（需要depth维度）

    Returns:
        温跃层深度场

    Example:
        >>> temp = ds['temp']
        >>> thermocline = identify_thermocline_depth(temp)
        >>> print(f"Mean thermocline depth: {thermocline.mean().values:.1f} m")
    """
    temp = materialize_partitioned_xarray(temp)

    if 'depth' not in temp.dims:
        raise ValueError("Temperature data must have depth dimension")

    temp = _ensure_gradient_depth_chunks(temp)

    # 计算垂直梯度
    dTdz = temp.differentiate('depth')
    abs_dTdz = np.abs(dTdz)

    # 找到最大梯度的深度
    depth_values = temp.depth.values

    def find_max_gradient_depth(gradient_profile):
        """找到最大梯度的深度"""
        if np.all(np.isnan(gradient_profile)):
            return np.nan
        idx = np.nanargmax(gradient_profile)
        return depth_values[idx]

    thermocline_depth = xr.apply_ufunc(
        find_max_gradient_depth,
        abs_dTdz,
        input_core_dims=[['depth']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={'allow_rechunk': True},
    )
    thermocline_depth.name = "thermocline_depth"

    thermocline_depth.attrs = {
        'long_name': 'Thermocline Depth',
        'units': 'm',
        'method': 'max_temperature_gradient',
        'feature': 'thermocline',
    }
    report_phase(
        phase="vertical_structure_ready",
        message="Prepared thermocline depth lazy field",
        percent=1.0,
        compute_backend="dask" if is_dask_backed(thermocline_depth) else "xarray",
    )

    return thermocline_depth


def identify_pycnocline_depth(density: xr.DataArray) -> xr.DataArray:
    """
    识别密度跃层深度

    找到密度垂直梯度最大的深度

    Args:
        density: 密度场（需要depth维度）

    Returns:
        密度跃层深度场

    Example:
        >>> density = ds['density']
        >>> pycnocline = identify_pycnocline_depth(density)
        >>> print(f"Mean pycnocline depth: {pycnocline.mean().values:.1f} m")
    """
    density = materialize_partitioned_xarray(density)

    if 'depth' not in density.dims:
        raise ValueError("Density data must have depth dimension")

    density = _ensure_gradient_depth_chunks(density)

    # 计算垂直梯度
    drhodz = density.differentiate('depth')
    abs_drhodz = np.abs(drhodz)

    # 找到最大梯度的深度
    depth_values = density.depth.values

    def find_max_gradient_depth(gradient_profile):
        """找到最大梯度的深度"""
        if np.all(np.isnan(gradient_profile)):
            return np.nan
        idx = np.nanargmax(gradient_profile)
        return depth_values[idx]

    pycnocline_depth = xr.apply_ufunc(
        find_max_gradient_depth,
        abs_drhodz,
        input_core_dims=[['depth']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={'allow_rechunk': True},
    )
    pycnocline_depth.name = "pycnocline_depth"

    pycnocline_depth.attrs = {
        'long_name': 'Pycnocline Depth',
        'units': 'm',
        'method': 'max_density_gradient',
        'feature': 'pycnocline',
    }
    report_phase(
        phase="vertical_structure_ready",
        message="Prepared pycnocline depth lazy field",
        percent=1.0,
        compute_backend="dask" if is_dask_backed(pycnocline_depth) else "xarray",
    )

    return pycnocline_depth


def analyze_vertical_structure(
    data: xr.Dataset,
    structure_type: Literal['mld', 'thermocline', 'pycnocline'],
    variable: Optional[str] = None,
    threshold: float = 0.03
) -> xr.DataArray:
    """
    分析垂直结构

    统一接口用于识别各种垂直结构特征

    Args:
        data: 输入数据集
        structure_type: 结构类型 ('mld', 'thermocline', 'pycnocline')
        variable: 变量名（mld和pycnocline需要density，thermocline需要temp）
        threshold: MLD阈值（仅用于mld）

    Returns:
        结构深度场

    Example:
        >>> ds = xr.open_dataset('ocean.nc')
        >>> mld = analyze_vertical_structure(ds, 'mld', variable='density')
        >>> thermocline = analyze_vertical_structure(ds, 'thermocline', variable='temp')
    """
    data = materialize_partitioned_xarray(data)

    if structure_type == 'mld':
        var = variable or 'density'
        if var not in data:
            raise ValueError(f"Variable '{var}' not found. MLD requires density.")
        return identify_mixed_layer_depth(data[var], threshold)

    elif structure_type == 'thermocline':
        var = variable or 'temp'
        if var not in data:
            raise ValueError(f"Variable '{var}' not found. Thermocline requires temperature.")
        return identify_thermocline_depth(data[var])

    elif structure_type == 'pycnocline':
        var = variable or 'density'
        if var not in data:
            raise ValueError(f"Variable '{var}' not found. Pycnocline requires density.")
        return identify_pycnocline_depth(data[var])

    else:
        raise ValueError(f"Unknown structure_type: {structure_type}")
