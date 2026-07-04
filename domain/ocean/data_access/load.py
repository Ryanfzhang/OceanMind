"""
数据访问工具

提供数据加载、查询、缓存等功能
"""

from collections import OrderedDict
import difflib
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List, Union, Iterable

import numpy as np
import xarray as xr

from packages.runtime.dataset_config import get_active_dataset_config, get_dataset_config_for_path
from packages.tool_loader.progress import report_tool_progress
from domain.ocean.data_access.partitioned import PartitionedDataArray
from domain.ocean.dask_utils import chunk_summary, is_dask_backed, report_phase


_DEFAULT_OPEN_DATASET_CACHE_MAXSIZE = 64
_LOAD_RESOLVING_PROGRESS = 0.01
_LOAD_RESOLVED_PROGRESS = 0.02
_LOAD_OPENING_METADATA_PROGRESS = 0.05
_LOAD_METADATA_OPENED_PROGRESS = 0.15
_LOAD_PREPARE_SUBSET_PROGRESS = 0.25
_LOAD_PREPARE_VERTICAL_PROGRESS = 0.35
_LOAD_FILE_PROGRESS_FRACTION = 0.95
_LOAD_COMBINE_PROGRESS = 0.98
_LOAD_CONCAT_PROGRESS = 0.985
_LOAD_TIME_CLEANUP_PROGRESS = 0.99
_LOAD_COMPLETE_PROGRESS = 1.0
_OPEN_DATASET_CACHE: OrderedDict[str, xr.Dataset] = OrderedDict()
_OPEN_DATASET_CACHE_LOCK = threading.RLock()
_SENTINEL_DEPTH_ABS_THRESHOLD = 9000.0
logger = logging.getLogger(__name__)


def _load_progress_unit_label(backend: str) -> str:
    return "data store" if backend == "zarr" else "data source"


def _pluralize_unit(label: str, count: int) -> str:
    if count == 1:
        return label
    if label.endswith("source"):
        return f"{label}s"
    if label.endswith("store"):
        return f"{label}s"
    return f"{label}s"


def _report_load_progress(
    *,
    backend: str,
    phase: str,
    message: str,
    percent: float,
    completed_units: int,
    total_units: int,
    unit_label: str,
    current_unit: Optional[str] = None,
) -> None:
    """Report storage-neutral load progress to the API/frontend layer."""
    payload = {
        "phase": phase,
        "message": message,
        "percent": percent,
        "completed_units": completed_units,
        "total_units": total_units,
        "unit_label": unit_label,
        "current_unit": current_unit,
        "storage_backend": backend,
    }

    # Legacy compatibility for callers that still consume the old progress keys.
    # The frontend no longer reads these fields.
    payload["completed_files"] = completed_units
    payload["total_files"] = total_units
    payload["current_file"] = current_unit
    report_tool_progress(**payload)


def clear_open_dataset_cache() -> None:
    """Close and clear the process-local opened dataset cache."""
    with _OPEN_DATASET_CACHE_LOCK:
        cached_datasets = list(_OPEN_DATASET_CACHE.values())
        _OPEN_DATASET_CACHE.clear()

    for dataset in cached_datasets:
        _close_cached_dataset(dataset)


def _open_dataset_cache_maxsize() -> int:
    raw_value = os.environ.get("OCEAN_OPEN_DATASET_CACHE_MAXSIZE", str(_DEFAULT_OPEN_DATASET_CACHE_MAXSIZE))
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return _DEFAULT_OPEN_DATASET_CACHE_MAXSIZE
    return max(1, min(parsed, 256))


def load_dataset(
    variable: str,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    time_range: Optional[Tuple[str, str]] = None,
    season_filter: Optional[str] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    vertical_mode: Optional[str] = None,
    depth_value: Optional[float] = None,
    data_path: Optional[str] = None,
    dataset: Optional[str] = None,
) -> Union[xr.DataArray, PartitionedDataArray]:
    """
    加载海洋数据集

    Args:
        dataset: 数据集名称。当前单数据集模式下可省略或传 'current'
        variable: 变量名（如'temp', 'chlorophyll', 'u'）
        lon_range: 经度范围 (min, max)，单位度东
        lat_range: 纬度范围 (min, max)，单位度北
        time_range: 时间范围 (start, end)，格式'YYYY-MM-DD'
        season_filter: 季节过滤器，可选值为 DJF/MAM/JJA/SON 或 winter/spring/summer/fall/autumn
        depth_range: 深度范围 (min, max)，单位米，可选
        vertical_mode: 垂向选择语义（surface/bottom/fixed_depth/depth_range）
        depth_value: 固定深度值，通常与 vertical_mode='fixed_depth' 一起使用
        data_path: 数据路径，如果None则使用环境变量

    Returns:
        xarray.DataArray: 加载的数据

    Example:
        >>> data = load_dataset(
        ...     variable='temp',
        ...     lon_range=(110, 120),
        ...     lat_range=(18, 23),
        ...     time_range=('2020-01-01', '2020-12-31')
        ... )
    """
    if data_path is None:
        data_path = _load_data_path()
        dataset_config = get_active_dataset_config()
    else:
        dataset_config = get_dataset_config_for_path(data_path)

    backend = _dataset_backend(dataset_config, Path(data_path))
    unit_label = _load_progress_unit_label(backend)
    _report_load_progress(
        backend=backend,
        phase="resolving_sources",
        message=f"Resolving data source for {variable}",
        percent=_LOAD_RESOLVING_PROGRESS,
        completed_units=0,
        total_units=0,
        unit_label=unit_label,
    )

    resolved_dataset = dataset or dataset_config.id
    if resolved_dataset not in {dataset_config.id, dataset_config.name, "current"}:
        raise ValueError(
            f"Unsupported dataset '{resolved_dataset}'. Active dataset is '{dataset_config.name}' ({dataset_config.id})."
        )

    # 构建数据文件路径
    data_files = _resolve_dataset_files(
        variable=variable,
        time_range=time_range,
        data_path=data_path,
        dataset_config=dataset_config,
    )
    total_files = len(data_files)
    _report_load_progress(
        backend=backend,
        phase="resolving_sources",
        message=f"Resolved {total_files} {_pluralize_unit(unit_label, total_files)} for {variable}",
        percent=_LOAD_RESOLVED_PROGRESS,
        completed_units=0,
        total_units=total_files,
        unit_label=unit_label,
    )

    if not data_files or not data_files[0].exists():
        # Fallback: infer bathymetry from temp data's deepest valid level
        normalized = _normalize_variable_alias(variable, dataset_config=dataset_config)
        if normalized in ("bathymetry", "depth", "bottom_depth", "topo"):
            return _infer_bathymetry(
                lon_range=lon_range,
                lat_range=lat_range,
                time_range=time_range,
                data_path=data_path,
            )
        data_dir = Path(data_path)
        available = sorted(list(data_dir.glob("*.nc")) + list(data_dir.glob("*.zarr")))[:10]
        available_str = "\n  ".join([f.name for f in available]) if available else "(none)"
        if backend == "zarr":
            expected = data_files[0] if data_files else _expected_zarr_store_path(
                normalized_variable=normalized,
                data_path=data_dir,
                dataset_config=dataset_config,
            )
            raise FileNotFoundError(
                f"Missing Zarr store for variable {variable}: expected {expected.name}\n"
                f"Resolved data path: {data_dir}\n"
                f"Available data stores:\n  {available_str}"
            )
        raise FileNotFoundError(
            f"Dataset not found for variable='{variable}': {data_files[0] if data_files else '(none)'}\n"
            f"Resolved data path: {data_dir}\n"
            f"Available data sources:\n  {available_str}"
        )

    normalized_variable = _normalize_variable_alias(variable, dataset_config=dataset_config)
    if len(data_files) == 1:
        _report_load_progress(
            backend=backend,
            phase="opening_metadata",
            message=f"Opening {unit_label} metadata",
            current_unit=data_files[0].name,
            percent=_LOAD_OPENING_METADATA_PROGRESS,
            completed_units=0,
            total_units=1,
            unit_label=unit_label,
        )
        ds = _open_resolved_dataset(data_files, dataset_config=dataset_config)
        _report_load_progress(
            backend=backend,
            phase="metadata_opened",
            message=f"Opened {unit_label} metadata",
            current_unit=data_files[0].name,
            percent=_LOAD_METADATA_OPENED_PROGRESS,
            completed_units=0,
            total_units=1,
            unit_label=unit_label,
        )
        dataset_variable = _resolve_dataset_variable_name(
            ds,
            normalized_variable,
            dataset_config=dataset_config,
        )

        if dataset_variable not in ds:
            raise ValueError(f"Variable '{variable}' not found in dataset")

        data = ds[dataset_variable]
        subset = _subset_loaded_dataarray(
            data=data,
            variable=variable,
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=time_range,
            season_filter=season_filter,
            depth_range=depth_range,
            vertical_mode=vertical_mode,
            depth_value=depth_value,
            defer_bottom_selection=backend == "zarr",
            progress_context={
                "backend": backend,
                "completed_units": 0,
                "total_units": 1,
                "unit_label": unit_label,
                "current_unit": data_files[0].name,
            },
        )
        _report_load_progress(
            backend=backend,
            phase="lazy_subset_ready",
            message="Prepared lazy subset",
            current_unit=data_files[0].name,
            percent=1.0,
            completed_units=1,
            total_units=1,
            unit_label=unit_label,
        )
        _report_load_progress(
            backend=backend,
            phase="complete",
            message="Prepared lazy subset",
            current_unit=data_files[0].name,
            percent=1.0,
            completed_units=1,
            total_units=1,
            unit_label=unit_label,
        )
        return subset

    subsets: List[xr.DataArray] = []
    for index, data_file in enumerate(data_files):
        _report_load_progress(
            backend=backend,
            phase="opening_source",
            message=f"Opening {unit_label} {index + 1}/{total_files}",
            current_unit=data_file.name,
            percent=_LOAD_FILE_PROGRESS_FRACTION * (index / total_files) if total_files else 0.0,
            completed_units=index,
            total_units=total_files,
            unit_label=unit_label,
        )
        ds = _open_dataset_cached(data_file, dataset_config=dataset_config)
        dataset_variable = _resolve_dataset_variable_name(
            ds,
            normalized_variable,
            dataset_config=dataset_config,
        )
        if dataset_variable not in ds:
            raise ValueError(f"Variable '{variable}' not found in dataset")
        subset = _subset_loaded_dataarray(
            data=ds[dataset_variable],
            variable=variable,
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=time_range,
            season_filter=season_filter,
            depth_range=depth_range,
            vertical_mode=vertical_mode,
            depth_value=depth_value,
            defer_bottom_selection=backend == "zarr",
        )
        subsets.append(subset)
        _report_load_progress(
            backend=backend,
            phase="opening_source",
            message=f"Opened {unit_label} {index + 1}/{total_files}",
            current_unit=data_file.name,
            percent=_LOAD_FILE_PROGRESS_FRACTION * ((index + 1) / total_files) if total_files else _LOAD_FILE_PROGRESS_FRACTION,
            completed_units=index + 1,
            total_units=total_files,
            unit_label=unit_label,
        )

    if not subsets:
        raise ValueError(f"Variable '{variable}' not found in dataset")

    _report_load_progress(
        backend=backend,
        phase="subset_prepared",
        message=f"Prepared lazy partitioned subset from {total_files} {_pluralize_unit(unit_label, total_files)}",
        percent=_LOAD_CONCAT_PROGRESS,
        completed_units=total_files,
        total_units=total_files,
        unit_label=unit_label,
    )
    combined = PartitionedDataArray(
        tuple(subsets),
        partition_labels=tuple(data_file.name for data_file in data_files),
        attrs=dict(getattr(subsets[0], "attrs", {})),
    )
    _report_load_progress(
        backend=backend,
        phase="complete",
        message=f"Prepared lazy partitioned subset from {total_files} {_pluralize_unit(unit_label, total_files)}",
        percent=_LOAD_COMPLETE_PROGRESS,
        completed_units=total_files,
        total_units=total_files,
        unit_label=unit_label,
    )
    return combined


def _infer_bathymetry(
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    time_range: Optional[Tuple[str, str]] = None,
    data_path: Optional[str] = None,
) -> xr.DataArray:
    """Infer bathymetry from the deepest valid level of temp data."""
    # Use a single time step to keep it lightweight
    infer_time = time_range if time_range else ("2015-01-01", "2015-01-31")
    temp = load_dataset(
        variable="temp",
        lon_range=lon_range,
        lat_range=lat_range,
        time_range=infer_time,
        data_path=data_path,
    )
    if isinstance(temp, PartitionedDataArray):
        temp = temp.partitions[0]
    depth_dim = None
    for name in ("depth", "lev", "z"):
        if name in temp.dims:
            depth_dim = name
            break
    if depth_dim is None:
        raise ValueError("Cannot infer bathymetry: temp data has no depth dimension")

    valid = np.isfinite(temp)
    reduce_dims = [d for d in temp.dims if d not in {depth_dim, "lat", "lon"}]
    for d in reduce_dims:
        valid = valid.any(dim=d)

    raw_depth_values = np.asarray(temp[depth_dim].values, dtype=float)
    valid_depth = xr.DataArray(
        np.isfinite(raw_depth_values) & (np.abs(raw_depth_values) < _SENTINEL_DEPTH_ABS_THRESHOLD),
        coords={depth_dim: temp[depth_dim]},
        dims=(depth_dim,),
    )
    valid = valid & valid_depth

    depth_values = xr.DataArray(
        np.abs(raw_depth_values),
        coords={depth_dim: temp[depth_dim]},
        dims=(depth_dim,),
    )
    depth_grid = depth_values.broadcast_like(valid)
    bottom_depth = depth_grid.where(valid).max(dim=depth_dim, skipna=True)
    bottom_depth = bottom_depth.transpose("lat", "lon")
    bottom_depth.name = "bottom_depth"
    bottom_depth.attrs["units"] = "m"
    bottom_depth.attrs["source"] = "inferred_from_temp_valid_depth"
    return bottom_depth


def get_dataset_info(
    dataset: Optional[str] = None,
    data_path: Optional[str] = None,
    include_runtime_probe: bool = False,
) -> Dict:
    """
    获取数据集信息，默认只读取 dataset config。

    Args:
        dataset: 数据集名称
        data_path: 数据路径
        include_runtime_probe: 是否尝试打开底层数据补充 dimensions/coords/attributes

    Returns:
        包含数据集元信息的字典

    Example:
        >>> info = get_dataset_info('sst')
        >>> print(info['variables'])
        ['temp', 'u', 'v']
    """
    if data_path is None:
        data_path = _load_data_path()
        dataset_config = get_active_dataset_config()
    else:
        dataset_config = get_dataset_config_for_path(data_path)

    requested_dataset = dataset or dataset_config.id
    if requested_dataset not in {dataset_config.id, dataset_config.name, "current"}:
        raise ValueError(
            f"Unsupported dataset '{requested_dataset}'. Active dataset is '{dataset_config.name}' ({dataset_config.id})."
        )

    variables = list(dataset_config.variables)
    variable_names = dict(dataset_config.variable_names or {})
    if not variables and variable_names:
        variables = list(variable_names.keys())

    info = {
        'output_type': 'metadata_result',
        'id': dataset_config.id,
        'name': dataset_config.name,
        'description': dataset_config.description,
        'data_path_redacted': True,
        'data_path_policy': 'Local server data paths are hidden from public dataset descriptions.',
        'backend': getattr(dataset_config, "backend", "netcdf"),
        'chunks': getattr(dataset_config, "chunks", None),
        'zarr_store_pattern': getattr(dataset_config, "zarr_store_pattern", None),
        'variables': variables,
        'variable_names': variable_names,
        'spatial_extent': dataset_config.spatial_extent,
        'temporal_extent': dataset_config.temporal_extent,
        'depth_levels': list(dataset_config.depth_levels),
        'depth_range': dataset_config.depth_range,
        'resolution': dataset_config.resolution,
    }
    if _dataset_backend(dataset_config, Path(data_path)) == "zarr":
        info["data_stores"] = _zarr_store_status(
            variables=list(dataset_config.variables),
            data_path=data_path,
            dataset_config=dataset_config,
        )

    info["summary"] = _format_dataset_config_summary(info)

    if not include_runtime_probe:
        return info

    probe_variable = (
        variables[0]
        if variables
        else next(iter(variable_names.keys()), None)
    )
    if probe_variable is None:
        return info

    try:
        data_files = _resolve_dataset_files(
            variable=probe_variable,
            time_range=None,
            data_path=data_path,
            dataset_config=dataset_config,
        )
        data_file = data_files[0]
    except Exception as exc:
        info["runtime_probe_error"] = _redact_sensitive_path_text(str(exc), data_path)
        return info

    if not data_file.exists():
        return info

    try:
        ds = _open_resolved_dataset([data_file], dataset_config=dataset_config)
    except Exception as exc:
        info["runtime_probe_error"] = _redact_sensitive_path_text(str(exc), data_path)
        return info

    if not info["variables"]:
        info["variables"] = list(ds.data_vars.keys())
    info["dimensions"] = dict(ds.sizes)
    info["coords"] = list(ds.coords.keys())
    info["attributes"] = dict(ds.attrs)
    return info


def _format_dataset_config_summary(info: Dict[str, Any]) -> str:
    summary_parts: List[str] = []
    name = str(info.get("name") or "the active dataset")
    dataset_id = str(info.get("id") or "current")
    description = str(info.get("description") or "").strip()
    if description:
        summary_parts.append(f"{name} ({dataset_id}) is configured as: {description}")
    else:
        summary_parts.append(f"{name} ({dataset_id}) is the active dataset.")

    variables = info.get("variables") if isinstance(info.get("variables"), list) else []
    if variables:
        summary_parts.append(f"It exposes {len(variables)} variables: {', '.join(variables)}.")

    spatial_extent = info.get("spatial_extent") or {}
    if isinstance(spatial_extent, dict):
        lon = spatial_extent.get("lon")
        lat = spatial_extent.get("lat")
        if isinstance(lon, (list, tuple)) and len(lon) == 2 and isinstance(lat, (list, tuple)) and len(lat) == 2:
            summary_parts.append(f"Spatial coverage is lon {lon[0]} to {lon[1]}, lat {lat[0]} to {lat[1]}.")

    temporal_extent = info.get("temporal_extent") or {}
    if isinstance(temporal_extent, dict) and temporal_extent.get("start") and temporal_extent.get("end"):
        summary_parts.append(f"Temporal coverage is {temporal_extent['start']} to {temporal_extent['end']}.")

    depth_range = info.get("depth_range")
    if isinstance(depth_range, list) and depth_range:
        summary_parts.append(f"Depth coverage is {depth_range[0]} to {depth_range[-1]}.")

    resolution = info.get("resolution")
    if resolution:
        summary_parts.append(f"Resolution is {resolution}.")

    summary_parts.append(f"The configured backend is {info.get('backend') or 'unknown'}.")

    return " ".join(summary_parts)


def _redact_sensitive_path_text(text: str, data_path: Optional[str]) -> str:
    if not text or not data_path:
        return text
    redacted = text
    path_variants = {
        str(data_path),
        str(Path(data_path)),
        Path(data_path).as_posix(),
    }
    try:
        path_variants.add(str(Path(data_path).resolve()))
        path_variants.add(Path(data_path).resolve().as_posix())
    except OSError:
        pass
    for variant in sorted(path_variants, key=len, reverse=True):
        if variant:
            redacted = redacted.replace(variant, "[hidden data path]")
    return redacted


def _zarr_store_status(
    *,
    variables: List[str],
    data_path: str,
    dataset_config=None,
) -> Dict[str, Dict[str, Any]]:
    status: Dict[str, Dict[str, Any]] = {}
    for variable in variables:
        try:
            stores = _resolve_dataset_files(
                variable=variable,
                time_range=None,
                data_path=data_path,
                dataset_config=dataset_config,
            )
            path = stores[0] if stores else _expected_zarr_store_path(
                normalized_variable=_normalize_variable_alias(variable, dataset_config=dataset_config),
                data_path=Path(data_path),
                dataset_config=dataset_config,
            )
            status[variable] = {"store": path.name, "exists": path.exists()}
        except Exception as exc:
            status[variable] = {
                "store": None,
                "exists": False,
                "error": _redact_sensitive_path_text(str(exc), data_path),
            }
    return status


def list_available_datasets(data_path: Optional[str] = None) -> List[Dict]:
    """
    列出所有可用的数据集

    Args:
        data_path: 数据路径

    Returns:
        数据集列表，每个元素是字典

    Example:
        >>> datasets = list_available_datasets()
        >>> for ds in datasets:
        ...     print(ds['name'], ds['size_mb'])
    """
    if data_path is None:
        data_path = _load_data_path()
        config = get_active_dataset_config()
    else:
        config = get_dataset_config_for_path(data_path)

    try:
        info = get_dataset_info(config.id, data_path)
        return [{
            'id': config.id,
            'name': config.name,
            'description': config.description,
            'data_path_redacted': True,
            'data_path_policy': 'Local server data paths are hidden from public dataset descriptions.',
            'backend': info.get('backend', getattr(config, "backend", "netcdf")),
            'variables': info.get('variables', []),
            'spatial_extent': info.get('spatial_extent'),
            'temporal_extent': info.get('temporal_extent'),
            'depth_levels': info.get('depth_levels'),
            'depth_range': info.get('depth_range'),
            'resolution': info.get('resolution'),
        }]
    except Exception:
        return [{
            'id': config.id,
            'name': config.name,
            'description': config.description,
            'data_path_redacted': True,
            'data_path_policy': 'Local server data paths are hidden from public dataset descriptions.',
            'backend': getattr(config, "backend", "netcdf"),
            'variables': config.variables,
            'spatial_extent': config.spatial_extent,
            'temporal_extent': config.temporal_extent,
            'depth_levels': config.depth_levels,
            'depth_range': config.depth_range,
            'resolution': config.resolution,
        }]


def extract_4d_subset(
    data: Union[xr.Dataset, xr.DataArray],
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    time_range: Optional[Tuple[str, str]] = None,
    depth_levels: Optional[List[float]] = None,
    time_indices: Optional[List[int]] = None
) -> Union[xr.Dataset, xr.DataArray]:
    """
    从 4D 海洋数据中提取时空子集。

    Args:
        data: 输入的 xarray 数据集或数组
        lon_range: 经度范围 (west, east)
        lat_range: 纬度范围 (south, north)
        depth_range: 深度范围
        time_range: 时间范围
        depth_levels: 精确深度层列表（与 depth_range 互斥）
        time_indices: 时间索引列表（与 time_range 互斥）

    Returns:
        提取后的子集数据
    """
    if depth_range is not None and depth_levels is not None:
        raise ValueError("depth_range and depth_levels are mutually exclusive")
    if time_range is not None and time_indices is not None:
        raise ValueError("time_range and time_indices are mutually exclusive")

    subset = data

    if lon_range is not None and 'lon' in subset.coords:
        subset = subset.sel(lon=_build_coord_slice(subset.lon.values, lon_range))

    if lat_range is not None and 'lat' in subset.coords:
        subset = subset.sel(lat=_build_coord_slice(subset.lat.values, lat_range))

    if time_range is not None and 'time' in subset.coords:
        subset = subset.sel(time=slice(*time_range))
    elif time_indices is not None and 'time' in subset.dims:
        subset = subset.isel(time=time_indices)

    depth_dim = get_depth_dim(subset)
    if depth_dim is not None:
        if depth_range is not None:
            _dv = np.asarray(subset[depth_dim].values, dtype=float)
            _nr = _normalize_depth_range(_dv, depth_range)
            subset = subset.sel({depth_dim: _build_coord_slice(_dv, _nr)})
        elif depth_levels is not None:
            nearest_indices = [
                int(np.nanargmin(np.abs(np.asarray(subset[depth_dim].values, dtype=float) - level)))
                for level in depth_levels
            ]
            subset = subset.isel({depth_dim: nearest_indices})

    return subset


def _subset_loaded_dataarray(
    *,
    data: xr.DataArray,
    variable: str,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    time_range: Optional[Tuple[str, str]],
    season_filter: Optional[str],
    depth_range: Optional[Tuple[float, float]],
    vertical_mode: Optional[str],
    depth_value: Optional[float],
    defer_bottom_selection: bool = False,
    progress_context: Optional[Dict[str, Any]] = None,
) -> xr.DataArray:
    """Apply spatial, temporal, and vertical slicing to an already opened field."""
    subset = data.sel(
        lon=_build_coord_slice(data.lon.values, lon_range),
        lat=_build_coord_slice(data.lat.values, lat_range),
    )
    _raise_if_empty_subset(
        subset,
        source=data,
        variable=variable,
        stage="spatial selection",
        lon_range=lon_range,
        lat_range=lat_range,
        time_range=time_range,
        season_filter=season_filter,
        depth_range=depth_range,
        vertical_mode=vertical_mode,
        depth_value=depth_value,
    )

    _report_subset_progress(
        progress_context,
        phase="preparing_spatial_time_subset",
        message="Preparing spatial and time subset",
        percent=_LOAD_PREPARE_SUBSET_PROGRESS,
    )

    if time_range is not None:
        subset = subset.sel(time=slice(*time_range))

    subset = _filter_dataarray_by_season(subset, season_filter)
    _raise_if_empty_subset(
        subset,
        source=data,
        variable=variable,
        stage="time/season selection",
        lon_range=lon_range,
        lat_range=lat_range,
        time_range=time_range,
        season_filter=season_filter,
        depth_range=depth_range,
        vertical_mode=vertical_mode,
        depth_value=depth_value,
            )

    mode = str(vertical_mode).strip().lower() if vertical_mode else None
    if mode:
        if _should_defer_bottom_selection(subset, mode=mode, defer_bottom_selection=defer_bottom_selection):
            vertical_message = "Bottom selection deferred to compute step"
        elif mode == "bottom":
            vertical_message = "Computing per-cell bottom valid layer"
        else:
            vertical_message = "Preparing vertical selection"
        _report_subset_progress(
            progress_context,
            phase="preparing_vertical_selection",
            message=vertical_message,
            percent=_LOAD_PREPARE_VERTICAL_PROGRESS,
        )

    single_depth_target = _single_depth_target(depth_range)
    effective_depth_value = depth_value
    if mode == "fixed_depth" and effective_depth_value is None and single_depth_target is not None:
        effective_depth_value = single_depth_target

    semantic_mode_selects_depth = mode in {"surface", "bottom"} or (
        mode == "fixed_depth" and effective_depth_value is not None
    )

    if depth_range is not None and not semantic_mode_selects_depth:
        depth_dim = get_depth_dim(subset)
        if depth_dim is not None:
            _dv = np.asarray(subset[depth_dim].values, dtype=float)
            _nr = _normalize_depth_range(_dv, depth_range)
            subset = subset.sel({depth_dim: _build_coord_slice(_dv, _nr)})
            _raise_if_empty_subset(
                subset,
                source=data,
                variable=variable,
                stage="depth selection",
                lon_range=lon_range,
                lat_range=lat_range,
                time_range=time_range,
                season_filter=season_filter,
                depth_range=depth_range,
                vertical_mode=vertical_mode,
                depth_value=depth_value,
            )

    subset = _apply_vertical_mode_subset(
        subset,
        vertical_mode=vertical_mode,
        depth_value=effective_depth_value,
        defer_bottom_selection=defer_bottom_selection,
    )
    _raise_if_empty_subset(
        subset,
        source=data,
        variable=variable,
        stage="vertical selection",
        lon_range=lon_range,
        lat_range=lat_range,
        time_range=time_range,
        season_filter=season_filter,
        depth_range=depth_range,
        vertical_mode=vertical_mode,
        depth_value=effective_depth_value,
    )

    return subset


def _report_subset_progress(
    progress_context: Optional[Dict[str, Any]],
    *,
    phase: str,
    message: str,
    percent: float,
) -> None:
    if not progress_context:
        return
    _report_load_progress(
        backend=str(progress_context["backend"]),
        phase=phase,
        message=message,
        percent=percent,
        completed_units=int(progress_context["completed_units"]),
        total_units=int(progress_context["total_units"]),
        unit_label=str(progress_context["unit_label"]),
        current_unit=(
            str(progress_context["current_unit"])
            if progress_context.get("current_unit") is not None
            else None
        ),
    )


def _raise_if_empty_subset(
    subset: xr.DataArray,
    *,
    source: xr.DataArray,
    variable: str,
    stage: str,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    time_range: Optional[Tuple[str, str]],
    season_filter: Optional[str],
    depth_range: Optional[Tuple[float, float]],
    vertical_mode: Optional[str],
    depth_value: Optional[float],
) -> None:
    empty_dims = [name for name, size in subset.sizes.items() if int(size) == 0]
    if not empty_dims:
        return

    requested_parts = [
        f"lon_range={_format_requested_range(lon_range)}",
        f"lat_range={_format_requested_range(lat_range)}",
    ]
    if time_range is not None:
        requested_parts.append(f"time_range={_format_requested_range(time_range)}")
    if season_filter:
        requested_parts.append(f"season_filter={season_filter}")
    if depth_range is not None:
        requested_parts.append(f"depth_range={_format_requested_range(depth_range)}")
    if vertical_mode:
        requested_parts.append(f"vertical_mode={vertical_mode}")
    if depth_value is not None:
        requested_parts.append(f"depth_value={float(depth_value):g}")

    coverage_parts = []
    for coord_name in ("lon", "lat", "time", "depth", "z"):
        coord_range = _format_coord_coverage(source, coord_name)
        if coord_range is not None:
            coverage_parts.append(f"{coord_name}={coord_range}")

    requested = "; ".join(requested_parts)
    coverage = "; ".join(coverage_parts) if coverage_parts else "unknown"
    raise ValueError(
        f"Dataset does not cover the requested subset for variable '{variable}'. "
        f"Empty dimension(s) after {stage}: {', '.join(empty_dims)}. "
        f"Requested: {requested}. Available dataset coverage: {coverage}. "
        "Choose a region/time/depth within the active dataset coverage or switch datasets."
    )


def _format_requested_range(value: Tuple[object, object]) -> str:
    return f"[{value[0]}, {value[1]}]"


def _format_coord_coverage(data: xr.DataArray, coord_name: str) -> Optional[str]:
    if coord_name not in data.coords:
        return None
    coord = data[coord_name]
    if coord.size == 0:
        return None
    values = np.asarray(coord.values)
    if coord_name == "time":
        return f"[{str(values[0])[:19]}, {str(values[-1])[:19]}]"
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError):
        return f"[{str(values[0])}, {str(values[-1])}]"
    finite = numeric[np.isfinite(numeric)]
    if coord_name in {"depth", "z"}:
        finite = finite[np.abs(finite) < _SENTINEL_DEPTH_ABS_THRESHOLD]
    if finite.size == 0:
        return None
    return f"[{float(np.nanmin(finite)):g}, {float(np.nanmax(finite)):g}]"


_SEASON_MONTHS: Dict[str, Tuple[int, int, int]] = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}

_SEASON_ALIASES = {
    "djf": "DJF",
    "winter": "DJF",
    "mam": "MAM",
    "spring": "MAM",
    "jja": "JJA",
    "summer": "JJA",
    "son": "SON",
    "fall": "SON",
    "autumn": "SON",
}


def normalize_season_filter(season_filter: Optional[str]) -> Optional[str]:
    """Normalize accepted season aliases to canonical DJF/MAM/JJA/SON labels."""
    if season_filter is None:
        return None

    normalized = str(season_filter).strip()
    if not normalized:
        return None

    canonical = _SEASON_ALIASES.get(normalized.lower())
    if canonical is None:
        valid = ", ".join(["DJF", "MAM", "JJA", "SON", "winter", "spring", "summer", "fall", "autumn"])
        raise ValueError(f"Unsupported season_filter '{season_filter}'. Valid values: {valid}")
    return canonical


def _filter_dataarray_by_season(data: xr.DataArray, season_filter: Optional[str]) -> xr.DataArray:
    canonical = normalize_season_filter(season_filter)
    if canonical is None:
        return data

    result = data
    if "time" in result.coords:
        season_months = _SEASON_MONTHS[canonical]
        month_index = result["time"].dt.month
        result = result.where(month_index.isin(season_months), drop=True)

    result.attrs = dict(result.attrs)
    result.attrs["season_filter"] = canonical
    result.attrs["season_months"] = list(_SEASON_MONTHS[canonical])
    return result


def _can_skip_time_cleanup(subsets: List[xr.DataArray]) -> Tuple[bool, str]:
    """Return whether yearly subsets already have ordered, non-overlapping time."""
    if not subsets:
        return False, "no_subsets"

    previous_end = None
    saw_time_values = False
    for index, subset in enumerate(subsets):
        if "time" not in getattr(subset, "coords", {}):
            return False, f"missing_time_coord_in_subset_{index + 1}"

        values = np.asarray(subset["time"].values)
        if values.size == 0:
            continue
        saw_time_values = True

        if not _time_values_are_monotonic_increasing(values):
            return False, f"non_monotonic_time_in_subset_{index + 1}"

        current_start = values[0]
        current_end = values[-1]
        if previous_end is not None:
            try:
                overlaps_or_reverses = bool(previous_end >= current_start)
            except TypeError:
                return False, f"uncomparable_time_between_subsets_{index}_{index + 1}"
            if overlaps_or_reverses:
                return False, f"overlapping_time_between_subsets_{index}_{index + 1}"
        previous_end = current_end

    if not saw_time_values:
        return False, "no_time_values"
    return True, "ordered_non_overlapping_time"


def _time_values_are_monotonic_increasing(values: np.ndarray) -> bool:
    if values.size <= 1:
        return True
    try:
        return bool(np.all(values[1:] >= values[:-1]))
    except TypeError:
        return False


def _apply_vertical_mode_subset(
    data: xr.DataArray,
    *,
    vertical_mode: Optional[str],
    depth_value: Optional[float],
    defer_bottom_selection: bool = False,
) -> xr.DataArray:
    """Apply a semantic vertical selection while preserving a depth dimension."""
    depth_dim = get_depth_dim(data)
    if depth_dim is None or not vertical_mode:
        return data

    mode = str(vertical_mode).strip().lower()
    if mode == "depth_range":
        return data
    if mode == "surface":
        return _select_nearest_surface_depth(data, depth_dim)
    if mode == "bottom":
        if _should_defer_bottom_selection(data, mode=mode, defer_bottom_selection=defer_bottom_selection):
            return _mark_pending_bottom_selection(data)
        return _select_deepest_valid_depth(data, depth_dim)
    if mode != "fixed_depth":
        return data
    if depth_value is None:
        return data

    return _select_nearest_depth(data, depth_dim, float(depth_value))


def _should_defer_bottom_selection(
    data: xr.DataArray,
    *,
    mode: Optional[str],
    defer_bottom_selection: bool,
) -> bool:
    return bool(
        defer_bottom_selection
        and mode == "bottom"
        and get_depth_dim(data) is not None
        and is_dask_backed(data)
    )


def _mark_pending_bottom_selection(data: xr.DataArray) -> xr.DataArray:
    marked = data.copy(deep=False)
    marked.attrs = dict(getattr(data, "attrs", {}))
    marked.attrs["pending_vertical_mode"] = "bottom"
    marked.attrs["bottom_selection"] = "deferred_per_cell_deepest_finite"
    marked.attrs["bottom_depth_coordinate"] = "per_cell"
    return marked


def has_pending_bottom_selection(data: Any) -> bool:
    return (
        isinstance(data, xr.DataArray)
        and str(data.attrs.get("pending_vertical_mode") or "").lower() == "bottom"
        and get_depth_dim(data) is not None
    )


def resolve_pending_bottom_selection(
    data: xr.DataArray,
    *,
    label: str = "per-cell bottom valid layer",
    start: float = 0.0,
    end: float = 0.1,
) -> xr.DataArray:
    """Apply a deferred local bottom selection before the actual compute boundary."""
    if not has_pending_bottom_selection(data):
        return data

    depth_dim = get_depth_dim(data)
    if depth_dim is None:
        return data

    report_phase(
        phase="preparing_vertical_selection",
        message="Preparing per-cell bottom valid layer",
        percent=start,
        compute_backend="dask" if is_dask_backed(data) else "xarray",
        chunks=chunk_summary(data) if is_dask_backed(data) else None,
    )
    selected = _select_deepest_valid_depth(
        data,
        depth_dim,
        progress_start=start,
        progress_end=end,
    )
    selected.attrs = dict(getattr(selected, "attrs", {}))
    selected.attrs.pop("pending_vertical_mode", None)
    selected.attrs["vertical_mode"] = "bottom"
    selected.attrs.setdefault("bottom_selection", "local_deepest_finite")
    selected.attrs["bottom_depth_coordinate"] = "per_cell"
    report_phase(
        phase="vertical_selection_ready",
        message=f"Prepared {label}",
        percent=end,
        compute_backend="dask" if is_dask_backed(selected) else "xarray",
        chunks=chunk_summary(selected) if is_dask_backed(selected) else None,
    )
    return selected


def _single_depth_target(depth_range: Optional[Tuple[float, float]]) -> Optional[float]:
    if depth_range is None or len(depth_range) != 2:
        return None
    first = float(depth_range[0])
    second = float(depth_range[1])
    if abs(first - second) > 1e-6:
        return None
    return first


def _select_nearest_surface_depth(data: xr.DataArray, depth_dim: str) -> xr.DataArray:
    depth_values = np.asarray(data[depth_dim].values, dtype=float)
    index = _nearest_surface_depth_index(depth_values)
    return data.isel({depth_dim: slice(index, index + 1)})


def _select_deepest_valid_depth(
    data: xr.DataArray,
    depth_dim: str,
    *,
    progress_start: float = 0.0,
    progress_end: float = 0.1,
) -> xr.DataArray:
    if _should_use_reference_day_bottom_selection(data, depth_dim):
        return _select_deepest_valid_depth_from_reference_day(
            data,
            depth_dim,
            progress_start=progress_start,
            progress_end=progress_end,
        )
    return _select_deepest_valid_depth_full(data, depth_dim)


def _select_deepest_valid_depth_full(data: xr.DataArray, depth_dim: str) -> xr.DataArray:
    depth_values = np.asarray(data[depth_dim].values, dtype=float)
    valid_indices = _valid_depth_indices(depth_values)
    if valid_indices.size == 0:
        fallback = data.isel({depth_dim: slice(-1, None)})
        fallback.attrs = dict(data.attrs)
        fallback.attrs["vertical_mode"] = "bottom"
        fallback.attrs["bottom_selection"] = "fallback_last_depth_no_valid_coordinate"
        return fallback

    valid_depth = xr.DataArray(
        np.isfinite(depth_values) & (np.abs(depth_values) < _SENTINEL_DEPTH_ABS_THRESHOLD),
        coords={depth_dim: data[depth_dim]},
        dims=(depth_dim,),
    )
    depth_score = xr.DataArray(
        np.abs(depth_values),
        coords={depth_dim: data[depth_dim]},
        dims=(depth_dim,),
    )

    finite_wet_mask = xr.apply_ufunc(
        np.isfinite,
        data,
        dask="parallelized",
        output_dtypes=[bool],
    ) & valid_depth
    has_finite_bottom = finite_wet_mask.any(dim=depth_dim)
    score = depth_score.where(finite_wet_mask, -np.inf)
    bottom_depth = score.idxmax(dim=depth_dim)
    selected = data.sel({depth_dim: bottom_depth}).where(has_finite_bottom)
    selected = selected.drop_vars(depth_dim, errors="ignore")
    selected = selected.expand_dims({depth_dim: [0.0]})

    ordered_dims = [dim for dim in data.dims if dim in selected.dims]
    remaining_dims = [dim for dim in selected.dims if dim not in ordered_dims]
    selected = selected.transpose(*(ordered_dims + remaining_dims))
    selected.attrs = dict(data.attrs)
    selected.attrs["vertical_mode"] = "bottom"
    selected.attrs["bottom_selection"] = "local_deepest_finite"
    selected.attrs["bottom_depth_coordinate"] = "per_cell"
    return selected


def _should_use_reference_day_bottom_selection(data: xr.DataArray, depth_dim: str) -> bool:
    return bool(
        is_dask_backed(data)
        and "time" in data.dims
        and depth_dim in data.dims
        and data.sizes.get("time", 0) > 0
        and data.sizes.get(depth_dim, 0) > 0
    )


def _select_deepest_valid_depth_from_reference_day(
    data: xr.DataArray,
    depth_dim: str,
    *,
    progress_start: float,
    progress_end: float,
) -> xr.DataArray:
    depth_values = np.asarray(data[depth_dim].values, dtype=float)
    valid_indices = _valid_depth_indices(depth_values)
    if valid_indices.size == 0:
        return _select_deepest_valid_depth_full(data, depth_dim)

    reference_time_value = data["time"].values[0] if "time" in data.coords else 0
    reference_time_label = _format_bottom_reference_time(reference_time_value)
    progress_span = max(0.0, progress_end - progress_start)
    selecting_percent = progress_start + progress_span * 0.25
    building_percent = progress_start + progress_span * 0.55
    applying_percent = progress_start + progress_span * 0.85

    report_phase(
        phase="selecting_bottom_reference_day",
        message=f"Selecting bottom reference day {reference_time_label}",
        percent=selecting_percent,
        compute_backend="dask",
        chunks=chunk_summary(data),
    )

    reference = data.isel(time=0, drop=True)
    valid_depth = xr.DataArray(
        np.isfinite(depth_values) & (np.abs(depth_values) < _SENTINEL_DEPTH_ABS_THRESHOLD),
        coords={depth_dim: data[depth_dim]},
        dims=(depth_dim,),
    )
    depth_score = xr.DataArray(
        np.abs(depth_values),
        coords={depth_dim: data[depth_dim]},
        dims=(depth_dim,),
    )

    finite_wet_mask = xr.apply_ufunc(
        np.isfinite,
        reference,
        dask="parallelized",
        output_dtypes=[bool],
    ) & valid_depth
    has_finite_bottom = finite_wet_mask.any(dim=depth_dim)
    score = depth_score.where(finite_wet_mask, -np.inf)
    bottom_index = score.argmax(dim=depth_dim).astype(np.int64)

    report_phase(
        phase="building_bottom_index_map",
        message="Computing bottom index map from reference day",
        percent=building_percent,
        compute_backend="dask",
        chunks=chunk_summary(reference),
    )
    reference_bottom = xr.Dataset(
        {
            "bottom_index": bottom_index,
            "has_finite_bottom": has_finite_bottom,
        }
    ).compute()

    valid_map = reference_bottom["has_finite_bottom"].astype(bool)
    if not bool(valid_map.any()):
        raise ValueError(
            "No finite water-column values were found on the bottom reference time step "
            f"{reference_time_label}. Choose a time range with valid data or inspect the dataset."
        )

    indexer = xr.DataArray(
        reference_bottom["bottom_index"].astype(np.int64).values,
        dims=reference_bottom["bottom_index"].dims,
        coords=reference_bottom["bottom_index"].coords,
    )
    valid_mask = xr.DataArray(
        valid_map.values,
        dims=valid_map.dims,
        coords=valid_map.coords,
    )

    report_phase(
        phase="applying_bottom_index_map",
        message="Applying reference-day bottom index map",
        percent=applying_percent,
        compute_backend="dask",
        chunks=chunk_summary(data),
    )
    selected = data.isel({depth_dim: indexer}).where(valid_mask)
    selected = selected.drop_vars(depth_dim, errors="ignore")
    selected = selected.expand_dims({depth_dim: [0.0]})

    ordered_dims = [dim for dim in data.dims if dim in selected.dims]
    remaining_dims = [dim for dim in selected.dims if dim not in ordered_dims]
    selected = selected.transpose(*(ordered_dims + remaining_dims))
    selected.attrs = dict(data.attrs)
    selected.attrs["vertical_mode"] = "bottom"
    selected.attrs["bottom_selection"] = "reference_day_deepest_finite"
    selected.attrs["bottom_depth_coordinate"] = "per_cell"
    selected.attrs["bottom_reference_time"] = reference_time_label
    return selected


def _format_bottom_reference_time(value: Any) -> str:
    try:
        array_value = np.asarray(value)
        if np.issubdtype(array_value.dtype, np.datetime64):
            return np.datetime_as_string(array_value.astype("datetime64[ns]"), unit="s")
    except Exception:
        pass
    return str(value)


def _select_nearest_depth(data: xr.DataArray, depth_dim: str, target_depth: float) -> xr.DataArray:
    depth_values = np.asarray(data[depth_dim].values, dtype=float)
    index = _nearest_depth_index_for_target(depth_values, target_depth)
    return data.isel({depth_dim: slice(index, index + 1)})


def _resolve_dataset_files(
    variable: str,
    time_range: Optional[Tuple[str, str]],
    data_path: str,
    dataset_config=None,
) -> List[Path]:
    """
    按变量名在 data_path 下扫描文件，不依赖 dataset 名称。

    搜索顺序：
    1. 若 time_range 跨多年，优先收集所有落在范围内的年度分文件
    2. 带年份的单年分文件：`*_{variable}_Zlev_{year}.nc` / `*_{variable}_{year}.nc`
    3. 不带年份的单文件：`*_{variable}_Zlev.nc` / `*_{variable}.nc`
    4. 按年份最近匹配的任意前缀文件
    """
    data_dir = Path(data_path)
    normalized = _normalize_variable_alias(variable, dataset_config=dataset_config)
    if _dataset_backend(dataset_config, data_dir) == "zarr":
        return _resolve_zarr_stores(
            normalized_variable=normalized,
            time_range=time_range,
            data_path=data_dir,
            dataset_config=dataset_config,
        )

    return _resolve_netcdf_files(
        normalized_variable=normalized,
        time_range=time_range,
        data_dir=data_dir,
    )


def _resolve_netcdf_files(
    *,
    normalized_variable: str,
    time_range: Optional[Tuple[str, str]],
    data_dir: Path,
) -> List[Path]:
    normalized = normalized_variable

    candidate_patterns: list[str] = []
    if time_range:
        yearly_matches = _collect_year_range_matches(
            data_dir=data_dir,
            normalized_variable=normalized,
            time_range=time_range,
        )
        if yearly_matches:
            _raise_if_yearly_matches_are_partial(
                data_dir=data_dir,
                normalized_variable=normalized,
                time_range=time_range,
                matches=yearly_matches,
            )
            return yearly_matches

        year = _extract_year(time_range[0])
        candidate_patterns += [f"*_{normalized}_Zlev_{year}.nc", f"*_{normalized}_{year}.nc"]
    candidate_patterns += [
        f"*_{normalized}_Zlev.nc",
        f"*_{normalized}.nc",
    ]

    for pattern in candidate_patterns:
        matches = sorted(data_dir.glob(pattern))
        if matches:
            return [matches[0]]

    # 任意前缀，按年份最近
    any_matches = sorted(data_dir.glob(f"*_{normalized}_*.nc"))
    if any_matches:
        if time_range:
            ranged_matches = _collect_year_range_matches(
                data_dir=data_dir,
                normalized_variable=normalized,
                time_range=time_range,
                matches=any_matches,
            )
            if ranged_matches:
                _raise_if_yearly_matches_are_partial(
                    data_dir=data_dir,
                    normalized_variable=normalized,
                    time_range=time_range,
                    matches=ranged_matches,
                )
                return ranged_matches
            yearly_years = _extract_years_from_files(any_matches)
            if yearly_years:
                requested_years = _requested_years(time_range)
                raise FileNotFoundError(
                    _missing_year_files_message(
                        data_dir=data_dir,
                        normalized_variable=normalized,
                        time_range=time_range,
                        missing_years=requested_years,
                        available_years=yearly_years,
                    )
                )
            selected = _select_closest_year_match(any_matches, _extract_year(time_range[0]))
            if selected is not None:
                return [selected]
        return [any_matches[-1]]

    # 最后 fallback：返回一个不存在的路径，让调用方报错
    return [data_dir / f"{normalized}.nc"]


def _resolve_zarr_stores(
    *,
    normalized_variable: str,
    time_range: Optional[Tuple[str, str]],
    data_path: Path,
    dataset_config=None,
) -> List[Path]:
    """Resolve Zarr stores for OceanMaster per-variable or per-year layouts."""
    if _is_zarr_store_path(data_path):
        return [data_path]

    data_dir = data_path
    pattern = getattr(dataset_config, "zarr_store_pattern", None)
    if pattern:
        resolved = _resolve_configured_zarr_pattern(
            data_dir=data_dir,
            pattern=str(pattern),
            normalized_variable=normalized_variable,
            time_range=time_range,
            dataset_config=dataset_config,
        )
        if resolved:
            return resolved

    if time_range:
        yearly_matches = _collect_year_range_matches(
            data_dir=data_dir,
            normalized_variable=normalized_variable,
            time_range=time_range,
            extension=".zarr",
        )
        if yearly_matches:
            _raise_if_yearly_matches_are_partial(
                data_dir=data_dir,
                normalized_variable=normalized_variable,
                time_range=time_range,
                matches=yearly_matches,
            )
            return yearly_matches

        year = _extract_year(time_range[0])
        for pattern_text in (
            f"*_{normalized_variable}_Zlev_{year}.zarr",
            f"*_{normalized_variable}_{year}.zarr",
        ):
            matches = sorted(data_dir.glob(pattern_text))
            if matches:
                return [matches[0]]

    for pattern_text in (
        f"*_{normalized_variable}.zarr",
        f"*_{normalized_variable}_Zlev.zarr",
        f"{normalized_variable}.zarr",
    ):
        matches = sorted(data_dir.glob(pattern_text))
        if matches:
            return [matches[0]]

    any_matches = sorted(data_dir.glob(f"*_{normalized_variable}_*.zarr"))
    if any_matches:
        if time_range:
            ranged_matches = _collect_year_range_matches(
                data_dir=data_dir,
                normalized_variable=normalized_variable,
                time_range=time_range,
                matches=any_matches,
                extension=".zarr",
            )
            if ranged_matches:
                _raise_if_yearly_matches_are_partial(
                    data_dir=data_dir,
                    normalized_variable=normalized_variable,
                    time_range=time_range,
                    matches=ranged_matches,
                )
                return ranged_matches
            yearly_years = _extract_years_from_files(any_matches)
            if yearly_years:
                requested_years = _requested_years(time_range)
                raise FileNotFoundError(
                    _missing_year_files_message(
                        data_dir=data_dir,
                        normalized_variable=normalized_variable,
                        time_range=time_range,
                        missing_years=requested_years,
                        available_years=yearly_years,
                    )
                )
            selected = _select_closest_year_match(any_matches, _extract_year(time_range[0]))
            if selected is not None:
                return [selected]
        return [any_matches[-1]]

    return [
        _expected_zarr_store_path(
            normalized_variable=normalized_variable,
            data_path=data_dir,
            dataset_config=dataset_config,
        )
    ]


def _expected_zarr_store_path(
    *,
    normalized_variable: str,
    data_path: Path,
    dataset_config=None,
) -> Path:
    pattern = getattr(dataset_config, "zarr_store_pattern", None)
    if pattern and "{year" not in str(pattern):
        try:
            return data_path / str(pattern).format(
                variable=normalized_variable,
                dataset=getattr(dataset_config, "id", "current"),
                name=getattr(dataset_config, "name", "dataset"),
            )
        except (KeyError, ValueError):
            pass

    prefix = str(
        getattr(dataset_config, "name", "")
        or getattr(dataset_config, "id", "")
        or ""
    ).strip()
    if prefix:
        safe_prefix = re.sub(r"\s+", "_", prefix)
        return data_path / f"{safe_prefix}_{normalized_variable}.zarr"
    return data_path / f"{normalized_variable}.zarr"


def _resolve_configured_zarr_pattern(
    *,
    data_dir: Path,
    pattern: str,
    normalized_variable: str,
    time_range: Optional[Tuple[str, str]],
    dataset_config=None,
) -> List[Path]:
    format_base = {
        "variable": normalized_variable,
        "dataset": getattr(dataset_config, "id", "current"),
        "name": getattr(dataset_config, "name", "dataset"),
    }
    if "{year" not in pattern:
        candidate = data_dir / pattern.format(**format_base)
        return [candidate] if candidate.exists() else []

    if not time_range:
        matches = sorted(data_dir.glob(pattern.format(**format_base, year="*")))
        return matches

    stores = []
    for year in _requested_years(time_range):
        candidate = data_dir / pattern.format(**format_base, year=year)
        if candidate.exists():
            stores.append(candidate)
    return stores


def _is_zarr_store_path(path: Path) -> bool:
    return path.suffix.lower() == ".zarr" or path.name.lower().endswith(".zarr")


def _open_resolved_dataset(data_files: List[Path], dataset_config=None) -> xr.Dataset:
    """
    Open one or more yearly files as a single normalized dataset.

    `xr.open_mfdataset` would normally work here, but it implicitly depends on
    dask in many environments. The benchmark environment is small enough that
    explicit open + concat is safer and easier to test.
    """
    if len(data_files) == 1:
        return _open_dataset_cached(data_files[0], dataset_config=dataset_config)

    datasets = [_open_dataset_cached(path, dataset_config=dataset_config) for path in data_files]
    if all("time" in ds.coords or "time" in ds.dims for ds in datasets):
        combined = xr.concat(
            datasets,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        )
        combined = combined.sortby("time")
        if "time" in combined.indexes:
            _, unique_indices = np.unique(combined["time"].values, return_index=True)
            combined = combined.isel(time=np.sort(unique_indices))
        return combined
    return xr.merge(datasets, compat="override", combine_attrs="override")


def _open_dataset_cached(path: Path, dataset_config=None) -> xr.Dataset:
    backend = _dataset_backend(dataset_config, path)
    chunks_key = _cache_chunks_key(_zarr_chunks(dataset_config)) if backend == "zarr" else ""
    cache_key = f"{backend}|{path.resolve()}|{chunks_key}"

    with _OPEN_DATASET_CACHE_LOCK:
        cached_dataset = _OPEN_DATASET_CACHE.pop(cache_key, None)
        if cached_dataset is not None:
            _OPEN_DATASET_CACHE[cache_key] = cached_dataset
            return cached_dataset

    opened_dataset = _normalize_dataset(_open_backend_dataset(path, backend=backend, dataset_config=dataset_config))
    evicted_dataset: Optional[xr.Dataset] = None

    with _OPEN_DATASET_CACHE_LOCK:
        cached_dataset = _OPEN_DATASET_CACHE.pop(cache_key, None)
        if cached_dataset is not None:
            _OPEN_DATASET_CACHE[cache_key] = cached_dataset
            evicted_dataset = opened_dataset
            opened_dataset = cached_dataset
        else:
            _OPEN_DATASET_CACHE[cache_key] = opened_dataset
            if len(_OPEN_DATASET_CACHE) > _open_dataset_cache_maxsize():
                _, evicted_dataset = _OPEN_DATASET_CACHE.popitem(last=False)

    if evicted_dataset is not None and evicted_dataset is not opened_dataset:
        _close_cached_dataset(evicted_dataset)

    return opened_dataset


def _open_backend_dataset(path: Path, *, backend: str, dataset_config=None) -> xr.Dataset:
    if backend == "zarr":
        try:
            return xr.open_zarr(
                path,
                chunks=_zarr_chunks(dataset_config),
                consolidated=None,
            )
        except ImportError as exc:
            raise ImportError(
                "Zarr backend requires optional dependencies. Install with: pip install -e '.[zarr]'"
            ) from exc
    return xr.open_dataset(path)


def _dataset_backend(dataset_config=None, path: Optional[Path] = None) -> str:
    configured = str(getattr(dataset_config, "backend", "") or "").strip().lower()
    if configured in {"zarr", "netcdf"}:
        return configured
    if path is not None and _is_zarr_store_path(path):
        return "zarr"
    if path is not None and path.is_dir():
        try:
            if any(path.glob("*.zarr")):
                return "zarr"
        except OSError:
            pass
    return "netcdf"


def _zarr_chunks(dataset_config=None):
    chunks = getattr(dataset_config, "chunks", None)
    if not chunks:
        return "auto"
    normalized = {}
    for name, value in dict(chunks).items():
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() == "auto":
            normalized[str(name)] = "auto"
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            normalized[str(name)] = parsed
    return normalized or "auto"


def _cache_chunks_key(chunks) -> str:
    if isinstance(chunks, dict):
        return ",".join(f"{key}:{chunks[key]}" for key in sorted(chunks))
    return str(chunks)


def _close_cached_dataset(dataset: xr.Dataset) -> None:
    try:
        dataset.close()
    except Exception:
        pass


def _collect_year_range_matches(
    *,
    data_dir: Path,
    normalized_variable: str,
    time_range: Tuple[str, str],
    matches: Optional[Iterable[Path]] = None,
    extension: str = ".nc",
) -> List[Path]:
    start_year = _extract_year(time_range[0])
    end_year = _extract_year(time_range[1])
    if not start_year.isdigit() or not end_year.isdigit():
        return []

    start = int(start_year)
    end = int(end_year)
    if end < start:
        start, end = end, start

    ranked: Dict[int, Path] = {}
    candidate_pool = list(matches) if matches is not None else []
    if not candidate_pool:
        for year in range(start, end + 1):
            for pattern in (
                f"*_{normalized_variable}_Zlev_{year}{extension}",
                f"*_{normalized_variable}_{year}{extension}",
            ):
                found = sorted(data_dir.glob(pattern))
                if found:
                    ranked[year] = found[0]
                    break
        return [ranked[year] for year in sorted(ranked)]

    for match in candidate_pool:
        year = _extract_year_from_path(match)
        if year is None:
            continue
        if start <= year <= end and year not in ranked:
            ranked[year] = match
    return [ranked[year] for year in sorted(ranked)]


def _requested_years(time_range: Tuple[str, str]) -> List[int]:
    start_year = _extract_year(time_range[0])
    end_year = _extract_year(time_range[1])
    if not start_year.isdigit() or not end_year.isdigit():
        return []

    start = int(start_year)
    end = int(end_year)
    if end < start:
        start, end = end, start
    return list(range(start, end + 1))


def _extract_years_from_files(matches: Iterable[Path]) -> List[int]:
    years: set[int] = set()
    for match in matches:
        year = _extract_year_from_path(match)
        if year is not None:
            years.add(year)
    return sorted(years)


def _extract_year_from_path(path: Path) -> Optional[int]:
    year_match = re.search(r"_(\d{4})(?:\.nc|\.zarr)$", path.name)
    if not year_match:
        return None
    return int(year_match.group(1))


def _raise_if_yearly_matches_are_partial(
    *,
    data_dir: Path,
    normalized_variable: str,
    time_range: Tuple[str, str],
    matches: Iterable[Path],
) -> None:
    requested_years = _requested_years(time_range)
    if not requested_years:
        return

    matched_years = _extract_years_from_files(matches)
    if not matched_years:
        return

    missing_years = [year for year in requested_years if year not in matched_years]
    if not missing_years:
        return

    available_years = _extract_years_from_files(
        list(data_dir.glob(f"*_{normalized_variable}_*.nc"))
        + list(data_dir.glob(f"*_{normalized_variable}_*.zarr"))
    )
    raise FileNotFoundError(
        _missing_year_files_message(
            data_dir=data_dir,
            normalized_variable=normalized_variable,
            time_range=time_range,
            missing_years=missing_years,
            available_years=available_years,
        )
    )


def _missing_year_files_message(
    *,
    data_dir: Path,
    normalized_variable: str,
    time_range: Tuple[str, str],
    missing_years: List[int],
    available_years: List[int],
) -> str:
    missing_text = ", ".join(str(year) for year in missing_years) or "(none)"
    if available_years:
        available_text = f"{available_years[0]}-{available_years[-1]}"
        if len(available_years) <= 12:
            available_text = ", ".join(str(year) for year in available_years)
    else:
        available_text = "(none)"
    return (
        f"Missing yearly data store(s) for variable='{normalized_variable}' in requested "
        f"time_range {time_range[0]} to {time_range[1]}: missing year(s) {missing_text}.\n"
        f"Available yearly file years for this variable: {available_text}.\n"
        f"Resolved data path: {data_dir}"
    )


def _normalize_variable_alias(variable: str, dataset_config=None) -> str:
    """将常见变量别名标准化为文件命名中使用的名称。

    Resolution order:
    1. User-friendly aliases (Chinese / English variants) → canonical name
    2. Config ``variable_names`` mapping (canonical → actual netCDF name)
    """
    normalized = _normalize_variable_name_input(variable)
    aliases = {
        "sst": "temp",
        "sea_surface_temperature": "temp",
        "temperature": "temp",
        "温度": "temp",
        "海温": "temp",
        "海表温度": "temp",
        "salinity": "salt",
        "sal": "salt",
        "psal": "salt",
        "盐度": "salt",
        "chlorophll": "chlorophyll",
        "chlorophyl": "chlorophyll",
        "chlor": "chlorophyll",
        "chl": "chlorophyll",
        "chla": "chlorophyll",
        "chl_a": "chlorophyll",
        "叶绿素": "chlorophyll",
        "oxygen_concentration": "oxygen",
        "o2": "oxygen",
        "do": "oxygen",
        "dissolved_oxygen": "oxygen",
        "氧气": "oxygen",
        "溶解氧": "oxygen",
        "zonal_velocity": "u",
        "zonal_current": "u",
        "东向流速": "u",
        "纬向流速": "u",
        "meridional_velocity": "v",
        "meridional_current": "v",
        "北向流速": "v",
        "经向流速": "v",
    }
    canonical = aliases.get(normalized, normalized)

    config = dataset_config or get_active_dataset_config()
    return config.resolve_variable(canonical)

def _resolve_dataset_variable_name(ds: xr.Dataset, variable: str, dataset_config=None) -> str:
    config = dataset_config or get_active_dataset_config()
    normalized = _normalize_variable_alias(variable, dataset_config=config)
    dataset_vars = {name.lower(): name for name in ds.data_vars}

    # Direct match (already mapped through config)
    if normalized in dataset_vars:
        return dataset_vars[normalized]

    # Try config mapping explicitly for the original variable name
    mapped = config.resolve_variable(variable.strip().lower())
    if mapped.lower() in dataset_vars:
        return dataset_vars[mapped.lower()]

    close_matches = difflib.get_close_matches(normalized, list(dataset_vars.keys()), n=1, cutoff=0.72)
    if close_matches:
        return dataset_vars[close_matches[0]]

    return variable


def _normalize_variable_name_input(variable: object) -> str:
    if isinstance(variable, str):
        return variable.strip().lower()

    if isinstance(variable, (list, tuple)):
        for item in variable:
            if isinstance(item, str) and item.strip():
                return item.strip().lower()
        raise ValueError("Variable list must contain at least one non-empty string.")

    raise TypeError(
        f"Variable name must be a string or a list/tuple of strings, received {type(variable).__name__}."
    )


def _list_dataset_variables(dataset: str, data_dir: Path) -> List[str]:
    pattern = re.compile(rf"^{re.escape(dataset)}_(.+?)(?:_Zlev)?(?:_\d{{4}})?\.(?:nc|zarr)$")
    variables = set()
    for candidate in list(data_dir.glob(f"{dataset}_*.nc")) + list(data_dir.glob(f"{dataset}_*.zarr")):
        match = pattern.match(candidate.name)
        if match:
            variables.add(match.group(1).lower())
    return sorted(variables)


def _select_closest_year_match(matches: List[Path], requested_year: str) -> Optional[Path]:
    if not requested_year.isdigit():
        return None

    target = int(requested_year)
    ranked: List[Tuple[int, int, Path]] = []
    for match in matches:
        year = _extract_year_from_path(match)
        if year is None:
            continue
        ranked.append((abs(year - target), -year, match))

    if not ranked:
        return None

    ranked.sort()
    return ranked[0][2]


def _load_data_path() -> str:
    """Load the currently active dataset directory from runtime configuration."""
    config = get_active_dataset_config()
    if config.data_path:
        return config.data_path
    return './data'


def _extract_year(date_str: str) -> str:
    """从 YYYY-MM-DD 或 YYYY-MM-DDTHH:MM:SS 中提取年份。"""
    return str(date_str)[:4]


def _normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """统一常见坐标名，减少下游工具对真实文件命名的耦合。"""
    rename_map = {}
    if 'z' in ds.dims and 'depth' not in ds.dims:
        rename_map['z'] = 'depth'
    if 'z' in ds.coords and 'depth' not in ds.coords:
        rename_map['z'] = 'depth'

    if rename_map:
        ds = ds.rename(rename_map)

    ds = _drop_sentinel_depth_coordinates(ds)
    return ds


def _drop_sentinel_depth_coordinates(data: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """Remove placeholder depth coordinates such as 9999 while preserving lazy data."""
    depth_dim = get_depth_dim(data)
    if depth_dim is None or depth_dim not in getattr(data, "coords", {}):
        return data

    try:
        depth_values = np.asarray(data[depth_dim].values, dtype=float)
    except (TypeError, ValueError):
        return data

    valid_mask = np.isfinite(depth_values) & (np.abs(depth_values) < _SENTINEL_DEPTH_ABS_THRESHOLD)
    if valid_mask.size == 0 or valid_mask.all() or not valid_mask.any():
        return data

    return data.isel({depth_dim: np.where(valid_mask)[0]})


def _normalize_depth_range(
    depth_values: np.ndarray,
    depth_range: Tuple[float, float],
) -> Tuple[float, float]:
    """Map an arbitrary-sign depth_range onto the dataset's actual coordinate system.

    Users (and the planner) may express depth as positive-down ([0, 200]) or
    negative-down ([0, -200]), while the underlying NetCDF may use either
    convention.  This helper:

    1. Strips obvious sentinel values (e.g. 9999).
    2. Converts *both* the user range and the data coordinates to absolute
       "distance from surface" so sign mismatches are eliminated.
    3. Clamps to the actual coordinate extent.
    4. Maps the result back to the dataset's native sign convention.

    Examples (negative-down data ``[0, -5, ..., -8000]``):
        depth_range=[0, 200]   → (-200, 0)
        depth_range=[0, -200]  → (-200, 0)
        depth_range=[50, 200]  → (-200, -50)

    Examples (positive-down data ``[0, 5, ..., 5000]``):
        depth_range=[0, 200]   → (0, 200)
        depth_range=[0, -200]  → (0, 200)
    """
    dv = np.asarray(depth_values, dtype=float)
    # Strip sentinel values (e.g. 9999 placeholder at the end of some grids)
    dv = dv[np.abs(dv) < _SENTINEL_DEPTH_ABS_THRESHOLD]
    if dv.size == 0:
        return tuple(depth_range)

    single_depth = _single_depth_target(depth_range)
    if single_depth is not None:
        nearest = _nearest_depth_coordinate(depth_values, single_depth)
        return (nearest, nearest)

    # Physical depth = distance from surface, always >= 0
    abs_shallow = min(abs(depth_range[0]), abs(depth_range[1]))
    abs_deep    = max(abs(depth_range[0]), abs(depth_range[1]))

    coord_min = float(dv.min())
    coord_max = float(dv.max())

    # Detect negative-down convention: surface ≈ 0, deeper values < 0
    if coord_min < 0 and coord_max <= 1e-3:
        # Coordinate's absolute depth extent: 0 … |coord_min|
        abs_data_max = abs(coord_min)
        # Clamp in absolute space then convert back to negative
        clamped_shallow = min(abs_shallow, abs_data_max)
        clamped_deep    = min(abs_deep,    abs_data_max)
        lo = -clamped_deep      # more negative = deeper
        hi = -clamped_shallow   # closer to 0   = shallower
    else:
        # Positive-down (or all-zero surface-only)
        clamped_shallow = max(abs_shallow, coord_min)
        clamped_deep    = min(abs_deep,    coord_max)
        lo = clamped_shallow
        hi = clamped_deep

    return (lo, hi)


def _build_coord_slice(values, target_range: Tuple[float, float]) -> slice:
    """根据坐标顺序构造安全的 slice。"""
    start, end = target_range
    if len(values) >= 2 and values[0] > values[1]:
        return slice(end, start)
    return slice(start, end)


def _valid_depth_indices(depth_values: np.ndarray) -> np.ndarray:
    values = np.asarray(depth_values, dtype=float)
    return np.where(np.isfinite(values) & (np.abs(values) < _SENTINEL_DEPTH_ABS_THRESHOLD))[0]


def _uses_negative_down_depths(depth_values: np.ndarray) -> bool:
    valid_indices = _valid_depth_indices(depth_values)
    if valid_indices.size == 0:
        return False
    clean = np.asarray(depth_values, dtype=float)[valid_indices]
    return bool(float(clean.min()) < 0 and float(clean.max()) <= 1e-3)


def _depth_target_in_coordinate_system(depth_values: np.ndarray, target_depth: float) -> float:
    target = float(target_depth)
    if _uses_negative_down_depths(depth_values):
        return -abs(target)
    return abs(target)


def _nearest_surface_depth_index(depth_values: np.ndarray) -> int:
    values = np.asarray(depth_values, dtype=float)
    valid_indices = _valid_depth_indices(values)
    if valid_indices.size == 0:
        return 0
    return int(valid_indices[np.nanargmin(np.abs(values[valid_indices]))])


def _nearest_depth_index_for_target(depth_values: np.ndarray, target_depth: float) -> int:
    values = np.asarray(depth_values, dtype=float)
    valid_indices = _valid_depth_indices(values)
    if valid_indices.size == 0:
        return int(np.nanargmin(np.abs(values - float(target_depth))))

    target = _depth_target_in_coordinate_system(values, target_depth)
    return int(valid_indices[np.nanargmin(np.abs(values[valid_indices] - target))])


def _nearest_depth_coordinate(depth_values: np.ndarray, target_depth: float) -> float:
    values = np.asarray(depth_values, dtype=float)
    index = _nearest_depth_index_for_target(values, target_depth)
    return float(values[index])


def get_depth_dim(data: xr.DataArray) -> Optional[str]:
    """Return the normalized depth dimension name if present."""
    for name in ("depth", "lev", "z"):
        if name in data.dims:
            return name
    return None
