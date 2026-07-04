"""
海洋事件检测工具 - 涡旋检测

使用Okubo-Weiss方法检测海洋涡旋
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Tuple
from scipy.ndimage import label, find_objects

from domain.ocean.data_access.partitioned import materialize_partitioned_xarray
from domain.ocean.dask_utils import compute_together_with_progress, is_dask_backed, report_phase
from domain.ocean.events.detection_utils import estimate_grid_spacing_km, report_detection_input


def detect_eddies(
    u: xr.DataArray,
    v: xr.DataArray,
    ow_threshold: float = -2e-12,
    min_radius_km: float = 30,
    max_radius_km: float = 300,
    min_pixels: int = 10
) -> Dict:
    """
    使用Okubo-Weiss方法检测涡旋

    识别气旋式和反气旋式涡旋

    Args:
        u: 东向流速
        v: 北向流速
        ow_threshold: Okubo-Weiss参数阈值（负值，越负越严格）
        min_radius_km: 最小涡旋半径(km)
        max_radius_km: 最大涡旋半径(km)
        min_pixels: 最小像素数

    Returns:
        涡旋事件字典

    Example:
        >>> u = ds['u'].isel(depth=0, time=0)
        >>> v = ds['v'].isel(depth=0, time=0)
        >>> eddies = detect_eddies(u, v)
        >>> print(f"Found {eddies['statistics']['total_count']} eddies")
    """
    u = materialize_partitioned_xarray(u)
    v = materialize_partitioned_xarray(v)
    report_detection_input("eddy", u, percent=0.02)

    # 如果有时间和深度维度，取第一个时间步和表层
    if 'time' in u.dims:
        u = u.isel(time=0)
        v = v.isel(time=0)

    depth_dim = None
    if 'depth' in u.dims:
        depth_dim = 'depth'
    elif 'z' in u.dims:
        depth_dim = 'z'

    if depth_dim is not None:
        u = u.isel({depth_dim: 0})
        v = v.isel({depth_dim: 0})
    report_detection_input("eddy horizontal field", u, percent=0.08)

    # 计算Okubo-Weiss参数
    ow = _compute_okubo_weiss(u, v)

    # 计算涡度
    vorticity = _compute_vorticity(u, v)
    ow, vorticity = compute_together_with_progress(
        (ow, vorticity),
        label="eddy Okubo-Weiss and vorticity fields",
        start=0.12,
        end=0.65,
    )
    ow_values = np.asarray(ow.values, dtype=float)
    vorticity_values = np.asarray(vorticity.values, dtype=float)

    # 检测涡旋区域
    eddy_mask = ow_values < ow_threshold

    # 分离气旋和反气旋涡旋
    cyclonic_mask = eddy_mask & (vorticity_values > 0)
    anticyclonic_mask = eddy_mask & (vorticity_values < 0)

    # 标记连通区域
    report_phase(phase="labeling_event_components", message="Labeling eddy components", percent=0.7)
    cyclonic_labels, n_cyclonic = label(cyclonic_mask)
    anticyclonic_labels, n_anticyclonic = label(anticyclonic_mask)

    # 提取涡旋
    cyclonic_eddies = _extract_eddies(
        cyclonic_labels, n_cyclonic,
        np.asarray(u.lon.values), np.asarray(u.lat.values),
        ow_values, vorticity_values,
        eddy_type='cyclonic', min_pixels=min_pixels,
        min_radius_km=min_radius_km, max_radius_km=max_radius_km
    )

    anticyclonic_eddies = _extract_eddies(
        anticyclonic_labels, n_anticyclonic,
        np.asarray(u.lon.values), np.asarray(u.lat.values),
        ow_values, vorticity_values,
        eddy_type='anticyclonic', min_pixels=min_pixels,
        min_radius_km=min_radius_km, max_radius_km=max_radius_km
    )

    all_eddies = cyclonic_eddies + anticyclonic_eddies

    # 统计信息
    statistics = {
        'total_count': len(all_eddies),
        'cyclonic_count': len(cyclonic_eddies),
        'anticyclonic_count': len(anticyclonic_eddies),
        'detection_params': {
            'ow_threshold': ow_threshold,
            'min_radius_km': min_radius_km,
            'max_radius_km': max_radius_km
        }
    }
    report_phase(phase="event_detection_complete", message="Eddy detection complete", percent=1.0)

    return {
        'event_type': 'eddy',
        'events': all_eddies,
        'statistics': statistics,
        'mask': eddy_mask,
        'ow_field': ow_values,
        'vorticity_field': vorticity_values,
        'coordinates': {
            'lon': u.lon.values.tolist(),
            'lat': u.lat.values.tolist()
        }
    }


def _compute_okubo_weiss(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """计算Okubo-Weiss参数: OW = Sn² + Ss² - ζ²"""
    _validate_gradient_grid(u)
    _validate_gradient_grid(v)
    u = _ensure_gradient_chunks(_ensure_gradient_chunks(u, 'lon'), 'lat')
    v = _ensure_gradient_chunks(_ensure_gradient_chunks(v, 'lon'), 'lat')

    # 计算应变和涡度
    dudx = u.differentiate('lon')
    dudy = u.differentiate('lat')
    dvdx = v.differentiate('lon')
    dvdy = v.differentiate('lat')

    # 正应变 (normal strain)
    Sn = dudx - dvdy

    # 剪切应变 (shear strain)
    Ss = dvdx + dudy

    # 相对涡度 (relative vorticity)
    zeta = dvdx - dudy

    # Okubo-Weiss参数
    ow = Sn**2 + Ss**2 - zeta**2

    return ow


def _compute_vorticity(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """计算相对涡度"""
    _validate_gradient_grid(u)
    _validate_gradient_grid(v)
    u = _ensure_gradient_chunks(u, 'lat')
    v = _ensure_gradient_chunks(v, 'lon')
    dvdx = v.differentiate('lon')
    dudy = u.differentiate('lat')
    return dvdx - dudy


def _validate_gradient_grid(data: xr.DataArray) -> None:
    if 'lat' not in data.dims or 'lon' not in data.dims:
        raise ValueError("Eddy detection requires lat and lon dimensions")
    if data.sizes.get('lat', 0) < 3 or data.sizes.get('lon', 0) < 3:
        raise ValueError("Eddy detection needs at least 3 grid points along both lat and lon")


def _ensure_gradient_chunks(data: xr.DataArray, dim: str) -> xr.DataArray:
    chunks = getattr(data.data, "chunks", None)
    if not chunks or dim not in data.dims:
        return data
    axis = data.get_axis_num(dim)
    dim_chunks = chunks[axis]
    if dim_chunks and min(int(value) for value in dim_chunks) <= 2:
        report_phase(
            phase="preparing_gradient_chunks",
            message=f"Rechunking {dim} for eddy detection gradient",
            percent=0.1,
            compute_backend="dask" if is_dask_backed(data) else "xarray",
        )
        return data.chunk({dim: -1})
    return data


def _extract_eddies(
    labels: np.ndarray,
    n_labels: int,
    lon: np.ndarray,
    lat: np.ndarray,
    ow_values: np.ndarray,
    vorticity_values: np.ndarray,
    eddy_type: str,
    min_pixels: int,
    min_radius_km: float,
    max_radius_km: float
) -> List[Dict]:
    """从标记的区域中提取涡旋"""
    eddies = []

    # 计算网格分辨率
    dx_km = estimate_grid_spacing_km(lon, lat)

    for i in range(1, n_labels + 1):
        mask = labels == i
        n_pixels = np.sum(mask)

        if n_pixels < min_pixels:
            continue

        # 计算涡旋属性
        y_indices, x_indices = np.where(mask)

        # 中心位置（加权平均）
        masked_ow_values = np.abs(ow_values[mask])
        weights = masked_ow_values / np.sum(masked_ow_values)

        center_lat_idx = int(np.average(y_indices, weights=weights))
        center_lon_idx = int(np.average(x_indices, weights=weights))

        center_lon = float(lon[center_lon_idx])
        center_lat = float(lat[center_lat_idx])

        # 估计半径（等效圆半径）
        area_km2 = n_pixels * (dx_km ** 2)
        radius_km = np.sqrt(area_km2 / np.pi)

        # 过滤半径
        if radius_km < min_radius_km or radius_km > max_radius_km:
            continue

        # 强度（OW参数的平均值）
        intensity = float(np.mean(ow_values[mask]))

        # 最大涡度
        max_vorticity = float(np.max(np.abs(vorticity_values[mask])))

        eddy = {
            'eddy_id': f'{eddy_type}_{len(eddies) + 1}',
            'type': eddy_type,
            'center': {'lon': center_lon, 'lat': center_lat},
            'radius_km': float(radius_km),
            'area_km2': float(area_km2),
            'intensity': intensity,
            'max_vorticity': max_vorticity,
            'n_pixels': int(n_pixels)
        }

        eddies.append(eddy)

    return eddies

