"""
海洋事件检测工具 - 锋面检测

基于水平梯度分析检测海洋锋面
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Optional
from scipy.ndimage import label, gaussian_filter

from domain.ocean.data_access.partitioned import materialize_partitioned_xarray
from domain.ocean.dask_utils import report_phase
from domain.ocean.events.detection_utils import (
    compute_event_array,
    estimate_grid_spacing_km,
    extract_bbox_from_indices,
    report_detection_input,
)


def detect_fronts(
    data: xr.DataArray,
    variable: str = 'temp',
    gradient_threshold: Optional[float] = None,
    min_length_km: float = 50,
    min_pixels: int = 10,
    smoothing_sigma: float = 1.0
) -> Dict:
    """
    检测海洋锋面

    基于温度或盐度的水平梯度分析，识别物理性质急剧变化的区域

    锋面特征：
    - 水平梯度异常大
    - 标志不同水团的边界
    - 对海洋环流和生态系统重要

    Args:
        data: 温度或盐度数据
        variable: 变量名称（'temp' 或 'salt'）
        gradient_threshold: 梯度阈值（单位/km），None表示自动计算
        min_length_km: 最小锋面长度(km)
        min_pixels: 最小像素数
        smoothing_sigma: 高斯平滑参数

    Returns:
        锋面事件字典

    Example:
        >>> temp = ds['temp'].isel(depth=0, time=0)
        >>> fronts = detect_fronts(temp, variable='temp')
        >>> print(f"Found {fronts['statistics']['total_count']} fronts")
    """
    data = materialize_partitioned_xarray(data)
    report_detection_input("front", data, percent=0.02)

    # 如果有时间和深度维度，取第一个时间步和表层
    if 'time' in data.dims:
        data = data.isel(time=0)
    if 'depth' in data.dims:
        data = data.isel(depth=0)
    report_detection_input("front horizontal field", data, percent=0.08)

    field_values = compute_event_array(
        data,
        label="front detection field",
        dtype=float,
        start=0.1,
        end=0.45,
    )

    # 平滑数据以减少噪声
    if smoothing_sigma > 0:
        smoothed = gaussian_filter(field_values, sigma=smoothing_sigma)
    else:
        smoothed = field_values

    # 计算水平梯度
    report_phase(phase="computing_event_diagnostic", message="Computing front gradient", percent=0.5)
    gradient_magnitude = _compute_gradient_magnitude(
        smoothed, data.lon.values, data.lat.values
    )

    # 确定阈值
    if gradient_threshold is None:
        # 自动计算：使用梯度的90百分位
        gradient_threshold = float(np.percentile(gradient_magnitude[~np.isnan(gradient_magnitude)], 90))

    # 识别强梯度区域（锋面）
    front_mask = gradient_magnitude > gradient_threshold

    # 标记连通区域
    report_phase(phase="labeling_event_components", message="Labeling front components", percent=0.7)
    labeled, n_labels = label(front_mask)

    # 估计网格分辨率
    grid_spacing_km = estimate_grid_spacing_km(data.lon.values, data.lat.values)

    # 提取锋面
    fronts = _extract_fronts(
        labeled, n_labels, field_values, data.lon.values, data.lat.values, gradient_magnitude,
        variable, min_pixels, min_length_km,
        grid_spacing_km
    )

    # 统计信息
    statistics = {
        'total_count': len(fronts),
        'detection_params': {
            'variable': variable,
            'gradient_threshold': gradient_threshold,
            'min_length_km': min_length_km,
            'smoothing_sigma': smoothing_sigma
        }
    }

    if fronts:
        lengths = [f['length_km'] for f in fronts]
        intensities = [f['max_gradient'] for f in fronts]
        statistics['mean_length_km'] = float(np.mean(lengths))
        statistics['mean_intensity'] = float(np.mean(intensities))
        statistics['max_length_km'] = float(np.max(lengths))
        statistics['max_intensity'] = float(np.max(intensities))

    return {
        'event_type': 'front',
        'events': fronts,
        'statistics': statistics,
        'gradient_field': gradient_magnitude,
        'front_mask': front_mask,
        'coordinates': {
            'lon': data.lon.values.tolist(),
            'lat': data.lat.values.tolist()
        }
    }


def _compute_gradient_magnitude(
    field: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray
) -> np.ndarray:
    """计算水平梯度幅度"""
    # 计算网格间距
    R = 6371.0  # km
    mean_lat = np.mean(lat)

    dlon = np.mean(np.diff(lon))
    dlat = np.mean(np.diff(lat))

    dx = R * np.deg2rad(dlon) * np.cos(np.deg2rad(mean_lat))
    dy = R * np.deg2rad(dlat)

    # 计算梯度（单位/km）
    grad_y, grad_x = np.gradient(field)
    grad_x = grad_x / dx
    grad_y = grad_y / dy

    # 梯度幅度
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return gradient_magnitude


def _extract_fronts(
    labels: np.ndarray,
    n_labels: int,
    data_values: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    gradient: np.ndarray,
    variable: str,
    min_pixels: int,
    min_length_km: float,
    grid_spacing_km: float
) -> List[Dict]:
    """提取锋面"""
    fronts = []

    for i in range(1, n_labels + 1):
        mask = labels == i
        n_pixels = np.sum(mask)

        if n_pixels < min_pixels:
            continue

        # 估计锋面长度（像素数 * 网格间距）
        length_km = n_pixels * grid_spacing_km

        if length_km < min_length_km:
            continue

        # 提取锋面属性
        y_indices, x_indices = np.where(mask)

        # 中心位置（最大梯度点）
        front_gradients = gradient[mask]
        max_gradient_idx = np.argmax(front_gradients)

        center_lat_idx = y_indices[max_gradient_idx]
        center_lon_idx = x_indices[max_gradient_idx]

        center_lon = float(lon[center_lon_idx])
        center_lat = float(lat[center_lat_idx])

        # 锋面方向（基于形状的主轴）
        orientation = _compute_orientation(y_indices, x_indices)

        # 温度/盐度差异
        front_values = data_values[mask]
        value_range = float(np.max(front_values) - np.min(front_values))

        front = {
            'front_id': f'front_{len(fronts) + 1}',
            'variable': variable,
            'center': {'lon': center_lon, 'lat': center_lat},
            'bbox': extract_bbox_from_indices(y_indices, x_indices, lon, lat),
            'length_km': float(length_km),
            'n_pixels': int(n_pixels),
            'max_gradient': float(np.max(front_gradients)),
            'mean_gradient': float(np.mean(front_gradients)),
            'value_range': value_range,
            'orientation_deg': float(orientation)
        }

        fronts.append(front)

    return fronts


def _compute_orientation(y_indices: np.ndarray, x_indices: np.ndarray) -> float:
    """
    计算锋面方向（度数）

    基于协方差矩阵的主成分分析
    """
    # 中心化
    y_center = y_indices - np.mean(y_indices)
    x_center = x_indices - np.mean(x_indices)

    # 协方差矩阵
    cov = np.cov(x_center, y_center)

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # 主轴对应最大特征值
    main_axis_idx = np.argmax(eigenvalues)
    main_axis = eigenvectors[:, main_axis_idx]

    # 计算角度（相对于东向）
    angle = np.arctan2(main_axis[1], main_axis[0])
    angle_deg = np.rad2deg(angle)

    # 归一化到[0, 180)
    if angle_deg < 0:
        angle_deg += 180

    return angle_deg

