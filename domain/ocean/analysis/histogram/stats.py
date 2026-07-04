"""
直方图分析工具
"""

from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

from domain.ocean.data_access.partitioned import find_partitioned_values, materialize_partitioned_xarray


def compute_histogram(
    data: xr.DataArray,
    n_bins: int = 50,
    bin_range: Optional[Tuple[float, float]] = None,
    normalize: bool = True,
    mask: Optional[xr.DataArray] = None
) -> Dict:
    """
    计算一维直方图。
    """
    if find_partitioned_values((data, mask)):
        from packages.tool_loader.partitioned_execution import _compute_partitioned_histogram

        return _compute_partitioned_histogram(
            {
                "data": data,
                "n_bins": n_bins,
                "bin_range": bin_range,
                "normalize": normalize,
                "mask": mask,
            }
        )
    data = materialize_partitioned_xarray(data)
    mask = materialize_partitioned_xarray(mask)

    values = _flatten_values(data, mask)
    counts, bin_edges = np.histogram(values, bins=n_bins, range=bin_range, density=normalize)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        'bin_edges': bin_edges.tolist(),
        'bin_centers': bin_centers.tolist(),
        'density': counts.tolist(),
        'metadata': {
            'variable': data.name or 'unknown',
            'units': data.attrs.get('units', ''),
            'n_bins': n_bins,
            'normalized': normalize,
            'statistics': _compute_1d_statistics(values, counts, bin_centers),
        }
    }


def compute_2d_histogram(
    data_x: xr.DataArray,
    data_y: xr.DataArray,
    n_bins: int = 50,
    range_x: Optional[Tuple[float, float]] = None,
    range_y: Optional[Tuple[float, float]] = None,
    normalize: bool = True,
    mask: Optional[xr.DataArray] = None
) -> Dict:
    """
    计算二维联合直方图。
    """
    if find_partitioned_values((data_x, data_y, mask)):
        from packages.tool_loader.partitioned_execution import _compute_partitioned_2d_histogram

        return _compute_partitioned_2d_histogram(
            {
                "data_x": data_x,
                "data_y": data_y,
                "n_bins": n_bins,
                "range_x": range_x,
                "range_y": range_y,
                "normalize": normalize,
                "mask": mask,
            }
        )
    data_x = materialize_partitioned_xarray(data_x)
    data_y = materialize_partitioned_xarray(data_y)
    mask = materialize_partitioned_xarray(mask)

    values_x = _flatten_values(data_x, mask)
    values_y = _flatten_values(data_y, mask)

    n_samples = min(values_x.size, values_y.size)
    values_x = values_x[:n_samples]
    values_y = values_y[:n_samples]

    density, x_edges, y_edges = np.histogram2d(
        values_x,
        values_y,
        bins=n_bins,
        range=[range_x, range_y],
        density=normalize,
    )

    return {
        'x_bin_edges': x_edges.tolist(),
        'y_bin_edges': y_edges.tolist(),
        'x_bin_centers': (0.5 * (x_edges[:-1] + x_edges[1:])).tolist(),
        'y_bin_centers': (0.5 * (y_edges[:-1] + y_edges[1:])).tolist(),
        'density': density.tolist(),
        'metadata': {
            'x_variable': data_x.name or 'x',
            'y_variable': data_y.name or 'y',
            'x_units': data_x.attrs.get('units', ''),
            'y_units': data_y.attrs.get('units', ''),
            'n_bins': n_bins,
            'normalized': normalize,
            'statistics': {
                'n_samples': int(n_samples),
                'x_mean': float(np.mean(values_x)),
                'y_mean': float(np.mean(values_y)),
                'x_std': float(np.std(values_x)),
                'y_std': float(np.std(values_y)),
            }
        }
    }


def _flatten_values(data: xr.DataArray, mask: Optional[xr.DataArray] = None) -> np.ndarray:
    field = data.where(mask) if mask is not None else data
    values = np.asarray(field.values, dtype=float).ravel()
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("No valid values available for histogram analysis")
    return values


def _compute_1d_statistics(values: np.ndarray, counts: np.ndarray, bin_centers: np.ndarray) -> Dict:
    peak_index = int(np.argmax(counts))
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'peak_value': float(bin_centers[peak_index]),
        'n_samples': int(values.size),
    }
