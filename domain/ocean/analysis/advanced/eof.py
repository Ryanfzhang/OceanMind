"""
EOF（经验正交函数）分析工具

识别海洋数据的主要空间-时间变化模式
也称为主成分分析(PCA)
"""

import numpy as np
import xarray as xr
from typing import Dict, Tuple, Optional, Literal

from domain.ocean.data_access.partitioned import materialize_partitioned_xarray
from domain.ocean.dask_utils import dataarray_to_numpy, is_dask_backed, report_phase


def perform_eof_analysis(
    data: xr.DataArray,
    n_modes: int = 3,
    preprocessing: Literal['anomaly', 'standardized', 'none'] = 'anomaly',
    weight_by_latitude: bool = True
) -> Dict:
    """
    执行EOF分析

    将时空数据分解为主要变化模式

    应用场景：
    - ENSO模式识别
    - 气候变化模式检测
    - 数据降噪和压缩

    Args:
        data: 输入数据（需要time维度）
        n_modes: 提取的模态数量
        preprocessing: 预处理方法
            - 'anomaly': 去除均值
            - 'standardized': 标准化(z-score)
            - 'none': 无预处理
        weight_by_latitude: 是否应用纬度权重（面积校正）

    Returns:
        EOF结果字典，包含空间模态和时间序列

    Example:
        >>> sst = ds['temp']  # (time, lat, lon)
        >>> eof_result = perform_eof_analysis(sst, n_modes=3)
        >>> print(f"Mode 1 explains {eof_result['modes'][0]['variance_explained']:.1f}% variance")
        >>> eof1 = eof_result['modes'][0]['spatial_pattern']
        >>> pc1 = eof_result['modes'][0]['time_series']
    """
    data = materialize_partitioned_xarray(data)

    if 'time' not in data.dims:
        raise ValueError("Data must have time dimension for EOF analysis")
    if 'lat' not in data.dims or 'lon' not in data.dims:
        raise ValueError("EOF analysis requires lat and lon dimensions")

    # 如果有深度维度，取表层
    depth_dim = 'depth' if 'depth' in data.dims else 'z' if 'z' in data.dims else None
    if depth_dim is not None:
        report_phase(
            phase="preparing_eof_input",
            message="Selecting surface layer for EOF analysis",
            percent=0.02,
            compute_backend="dask" if is_dask_backed(data) else "xarray",
        )
        data = data.isel({depth_dim: 0})

    # 获取维度
    data = data.transpose('time', 'lat', 'lon')
    n_times = len(data.time)
    n_lat = len(data.lat)
    n_lon = len(data.lon)

    # 重塑为(time, space)
    report_phase(
        phase="preparing_eof_matrix",
        message="Preparing EOF input matrix",
        percent=0.05,
        compute_backend="dask" if is_dask_backed(data) else "xarray",
    )
    data_values = dataarray_to_numpy(
        data,
        label="EOF input matrix",
        dtype=float,
        start=0.05,
        end=0.45,
    )
    data_2d = data_values.reshape(n_times, n_lat * n_lon)

    # 处理缺失值
    valid_mask = ~np.isnan(data_2d).any(axis=0)
    n_valid = int(np.sum(valid_mask))

    if n_valid == 0:
        raise ValueError("No valid data points (all NaN)")

    data_valid = data_2d[:, valid_mask]

    # 预处理
    report_phase(
        phase="preprocessing_eof_matrix",
        message="Preprocessing EOF matrix",
        percent=0.5,
        compute_backend="numpy",
    )
    if preprocessing == 'anomaly':
        # 去除气候态均值
        climatology = np.mean(data_valid, axis=0, keepdims=True)
        data_processed = data_valid - climatology
    elif preprocessing == 'standardized':
        # 标准化
        mean = np.mean(data_valid, axis=0, keepdims=True)
        std = np.std(data_valid, axis=0, keepdims=True)
        std[std == 0] = 1
        data_processed = (data_valid - mean) / std
    else:
        data_processed = data_valid.copy()

    # 纬度权重（面积校正）
    if weight_by_latitude:
        lat_grid, lon_grid = np.meshgrid(data.lat.values, data.lon.values, indexing='ij')
        lat_flat = lat_grid.flatten()[valid_mask]

        # 权重为sqrt(cos(纬度))
        weights = np.sqrt(np.cos(np.deg2rad(lat_flat)))
        weights = weights / np.mean(weights)  # 归一化

        data_weighted = data_processed * weights[np.newaxis, :]
    else:
        data_weighted = data_processed
        weights = None

    # 中心化
    data_centered = data_weighted - np.mean(data_weighted, axis=0, keepdims=True)

    # SVD分解
    report_phase(
        phase="solving_eof",
        message="Solving EOF decomposition",
        percent=0.62,
        compute_backend="numpy",
    )
    if n_times < n_valid:
        # 时间维度较小 - 使用时间协方差矩阵
        cov_matrix = (data_centered @ data_centered.T) / (n_times - 1)
        eigenvalues, eigenvectors_time = np.linalg.eigh(cov_matrix)

        # 按特征值降序排序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = _clip_eigenvalues(eigenvalues[idx])
        eigenvectors_time = eigenvectors_time[:, idx]
        mode_scale = _mode_scale(eigenvalues, n_times - 1)

        # 计算空间模态(EOFs)
        eofs_weighted = (data_centered.T @ eigenvectors_time) / mode_scale

        # 主成分(时间序列)
        pcs = eigenvectors_time * mode_scale
    else:
        # 空间维度较小 - 使用空间协方差矩阵
        cov_matrix = (data_centered.T @ data_centered) / (n_times - 1)
        eigenvalues, eofs_weighted = np.linalg.eigh(cov_matrix)

        # 降序排序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = _clip_eigenvalues(eigenvalues[idx])
        eofs_weighted = eofs_weighted[:, idx]

        # 主成分
        pcs = data_centered @ eofs_weighted / _safe_sqrt(eigenvalues)

    # 保留请求的模态数
    n_modes = min(n_modes, len(eigenvalues))
    eigenvalues = eigenvalues[:n_modes]
    eofs_weighted = eofs_weighted[:, :n_modes]
    pcs = pcs[:, :n_modes]

    # 计算方差解释率
    total_variance = np.sum(eigenvalues)
    if not np.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError("EOF analysis found no finite variance in the selected data")
    variance_explained = (eigenvalues / total_variance) * 100
    cumulative_variance = np.cumsum(variance_explained)

    # 去除权重
    if weight_by_latitude:
        eofs_unweighted = eofs_weighted / weights[:, np.newaxis]
    else:
        eofs_unweighted = eofs_weighted

    # 构建模态
    report_phase(
        phase="assembling_eof_result",
        message="Assembling EOF modes",
        percent=0.9,
        compute_backend="numpy",
    )
    modes = []
    for i in range(n_modes):
        # 重构空间模态（包括NaN位置）
        eof_pattern_flat = np.full(n_lat * n_lon, np.nan)
        eof_pattern_flat[valid_mask] = eofs_unweighted[:, i]
        eof_pattern = eof_pattern_flat.reshape(n_lat, n_lon)

        # 符号约定：使空间模态主要为正值
        if np.sum(eof_pattern[~np.isnan(eof_pattern)] > 0) < np.sum(eof_pattern[~np.isnan(eof_pattern)] < 0):
            eof_pattern = -eof_pattern
            pcs[:, i] = -pcs[:, i]

        # 创建DataArray
        spatial_pattern = xr.DataArray(
            eof_pattern,
            coords={'lat': data.lat, 'lon': data.lon},
            dims=['lat', 'lon'],
            attrs={
                'long_name': f'EOF Mode {i+1}',
                'variance_explained': f'{variance_explained[i]:.2f}%'
            }
        )

        time_series = xr.DataArray(
            pcs[:, i],
            coords={'time': data.time},
            dims=['time'],
            attrs={
                'long_name': f'Principal Component {i+1}',
                'variance_explained': f'{variance_explained[i]:.2f}%'
            }
        )

        mode = {
            'mode_number': i + 1,
            'variance_explained': float(variance_explained[i]),
            'spatial_pattern': spatial_pattern,
            'time_series': time_series,
            'eigenvalue': float(eigenvalues[i])
        }

        modes.append(mode)

    report_phase(
        phase="eof_complete",
        message="EOF analysis complete",
        percent=1.0,
        compute_backend="numpy",
    )

    return {
        'n_modes': n_modes,
        'modes': modes,
        'cumulative_variance': cumulative_variance.tolist(),
        'total_variance': float(total_variance),
        'metadata': {
            'variable': data.name or 'unknown',
            'preprocessing': preprocessing,
            'weight_by_latitude': weight_by_latitude,
            'n_times': n_times,
            'n_spatial_points': n_valid,
            'data_shape': list(data.shape)
        }
    }


def _clip_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)


def _safe_sqrt(values: np.ndarray) -> np.ndarray:
    eps = np.finfo(float).eps
    return np.sqrt(np.maximum(np.asarray(values, dtype=float), eps))


def _mode_scale(eigenvalues: np.ndarray, degrees: int) -> np.ndarray:
    return _safe_sqrt(eigenvalues * max(1, int(degrees)))


def reconstruct_from_eof(
    eof_result: Dict,
    mode_indices: Optional[list] = None
) -> xr.DataArray:
    """
    从EOF模态重构数据

    Args:
        eof_result: EOF分析结果
        mode_indices: 使用的模态索引列表（None表示使用全部）

    Returns:
        重构的数据

    Example:
        >>> # 只使用前2个模态重构
        >>> reconstructed = reconstruct_from_eof(eof_result, mode_indices=[0, 1])
    """
    if mode_indices is None:
        mode_indices = list(range(eof_result['n_modes']))

    modes = eof_result['modes']

    # 获取第一个模态的坐标
    first_mode = modes[0]
    spatial_shape = first_mode['spatial_pattern'].shape
    time_coords = first_mode['time_series'].time
    lat_coords = first_mode['spatial_pattern'].lat
    lon_coords = first_mode['spatial_pattern'].lon

    # 初始化重构数据
    reconstructed = np.zeros((len(time_coords), *spatial_shape))

    # 累加选定的模态
    for idx in mode_indices:
        if idx >= len(modes):
            continue
        mode = modes[idx]
        spatial = mode['spatial_pattern'].values  # (lat, lon)
        temporal = mode['time_series'].values  # (time,)

        # 外积重构: data = PC ⊗ EOF
        reconstructed += temporal[:, np.newaxis, np.newaxis] * spatial[np.newaxis, :, :]

    # 创建DataArray
    result = xr.DataArray(
        reconstructed,
        coords={'time': time_coords, 'lat': lat_coords, 'lon': lon_coords},
        dims=['time', 'lat', 'lon'],
        attrs={
            'long_name': f"Reconstructed using modes {mode_indices}",
            'n_modes_used': len(mode_indices)
        }
    )

    return result
