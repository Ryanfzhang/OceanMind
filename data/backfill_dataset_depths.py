#!/usr/bin/env python3
"""Backfill dataset metadata in configs/dataset_config.yaml from data stores.

The OceanMaster frontend reads spatial, temporal, depth, and resolution metadata
from the dataset config. This utility scans the configured NetCDF or Zarr
dataset path and writes the discovered metadata back to the config.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import xarray as xr
import yaml

try:
    from netCDF4 import Dataset as NetCDFDataset
    from netCDF4 import num2date
except Exception:  # pragma: no cover - optional runtime dependency.
    NetCDFDataset = None  # type: ignore[assignment]
    num2date = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_config.yaml"
DEFAULT_SENTINEL_DEPTH_THRESHOLD = 9000.0

DEPTH_NAME_HINTS = ("depth", "deptht", "depthu", "depthv", "depthw", "lev", "z")
LON_NAME_HINTS = ("lon", "longitude", "nav_lon", "x")
LAT_NAME_HINTS = ("lat", "latitude", "nav_lat", "y")
TIME_NAME_HINTS = ("time", "datetime", "date")


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset config must be a YAML mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def _resolve_data_path(raw_path: str, config_path: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if config_path.resolve() == DEFAULT_CONFIG_PATH.resolve():
        return (PROJECT_ROOT / candidate).resolve()
    return (config_path.parent / candidate).resolve()


def _is_zarr_store_path(path: Path) -> bool:
    return path.suffix.lower() == ".zarr" or path.name.lower().endswith(".zarr")


def _iter_netcdf_files(data_path: Path) -> list[Path]:
    if data_path.is_file():
        if data_path.suffix.lower() != ".nc":
            raise ValueError(f"Configured data_path is a file but not NetCDF: {data_path}")
        return [data_path]
    return sorted(path for path in data_path.rglob("*.nc") if path.is_file())


def _iter_zarr_stores(data_path: Path) -> list[Path]:
    if _is_zarr_store_path(data_path):
        if not data_path.exists():
            raise FileNotFoundError(f"Configured Zarr store does not exist: {data_path}")
        return [data_path]
    if data_path.is_file():
        return []
    return sorted(path for path in data_path.rglob("*.zarr") if path.is_dir())


def _iter_metadata_sources(data_path: Path) -> list[Path]:
    if _is_zarr_store_path(data_path):
        return _iter_zarr_stores(data_path)
    if data_path.is_file():
        return _iter_netcdf_files(data_path)
    zarr_stores = _iter_zarr_stores(data_path)
    netcdf_files = _iter_netcdf_files(data_path)
    return sorted([*zarr_stores, *netcdf_files])


def _metadata_file_group_key(path: Path) -> str:
    return re.sub(r"([_-])\d{4}(?=\.nc$|\.zarr$|$)", r"\1YYYY", path.name)


def _default_metadata_files(files: Sequence[Path]) -> list[Path]:
    grouped: dict[str, list[Path]] = {}
    for path in files:
        grouped.setdefault(_metadata_file_group_key(path), []).append(path)

    selected: list[Path] = []
    for key in sorted(grouped):
        group = sorted(grouped[key])
        selected.append(group[0])
        if group[-1] != group[0]:
            selected.append(group[-1])
    return selected


def _candidate_names(names: Iterable[str], exact_hints: Sequence[str], fuzzy_terms: Sequence[str]) -> list[str]:
    source_names = list(dict.fromkeys(names))
    lower_lookup = {name: name.strip().lower() for name in source_names}
    exact = [name for name in source_names if lower_lookup[name] in exact_hints]
    fuzzy = [
        name
        for name in source_names
        if name not in exact and any(term in lower_lookup[name] for term in fuzzy_terms)
    ]
    return exact + fuzzy


def _finite_numeric(values: Any) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.array([], dtype=float)
    return array[np.isfinite(array)]


def _round_float(value: float) -> float:
    return float(f"{float(value):.10g}")


def _normalize_depth_levels(
    values: Sequence[float],
    *,
    include_sentinel_depths: bool = False,
    sentinel_threshold: float = DEFAULT_SENTINEL_DEPTH_THRESHOLD,
) -> tuple[list[float], int]:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    filtered_values = finite_values
    filtered_count = 0
    if not include_sentinel_depths:
        filtered_values = [value for value in finite_values if abs(value) < sentinel_threshold]
        filtered_count = len(finite_values) - len(filtered_values)
        if not filtered_values and finite_values:
            filtered_values = finite_values
            filtered_count = 0

    unique_values = sorted(set(filtered_values), key=lambda value: (abs(value), value))
    return [_round_float(value) for value in unique_values], filtered_count


def _numeric_extent(values: Sequence[float]) -> list[float] | None:
    finite = _finite_numeric(values)
    if finite.size == 0:
        return None
    return [_round_float(float(np.nanmin(finite))), _round_float(float(np.nanmax(finite)))]


def _median_spacing(values: Sequence[float]) -> float | None:
    finite = np.unique(_finite_numeric(values))
    if finite.size < 2:
        return None
    diffs = np.diff(np.sort(finite))
    diffs = np.abs(diffs[np.isfinite(diffs) & (np.abs(diffs) > 0)])
    if diffs.size == 0:
        return None
    return _round_float(float(np.nanmedian(diffs)))


def _format_spatial_resolution(lon_values: Sequence[float], lat_values: Sequence[float]) -> str | None:
    lon_spacing = _median_spacing(lon_values)
    lat_spacing = _median_spacing(lat_values)
    if lon_spacing is None or lat_spacing is None:
        return None
    return f"{lon_spacing:g} deg x {lat_spacing:g} deg"


def _as_iso_date(value: Any) -> str | None:
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        return str(value.astype("datetime64[D]"))
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if all(hasattr(value, attr) for attr in ("year", "month", "day")):
        return f"{int(value.year):04d}-{int(value.month):02d}-{int(value.day):02d}"
    try:
        parsed = np.datetime64(value, "D")
    except (TypeError, ValueError):
        return None
    if np.isnat(parsed):
        return None
    return str(parsed)


def _decode_time_values(values: Any, *, units: str | None = None, calendar: str | None = None) -> list[str]:
    array = np.asarray(values).ravel()
    if array.size == 0:
        return []

    decoded_values: Iterable[Any]
    if np.issubdtype(array.dtype, np.datetime64):
        decoded_values = array.astype("datetime64[D]")
    elif units and num2date is not None:
        try:
            decoded_values = num2date(
                array,
                units=units,
                calendar=calendar or "standard",
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
        except Exception:
            decoded_values = array
    else:
        decoded_values = array

    dates = [_as_iso_date(value) for value in decoded_values]
    return [date for date in dates if date is not None]


def _format_temporal_resolution(dates: Sequence[str]) -> str | None:
    unique_days = np.unique(np.asarray(dates, dtype="datetime64[D]"))
    if unique_days.size < 2:
        return None
    diffs = np.diff(np.sort(unique_days)).astype("timedelta64[D]").astype(int)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    median_days = float(np.nanmedian(diffs))
    if median_days <= 1.1:
        return "daily"
    if 6.5 <= median_days <= 7.5:
        return "weekly"
    if 27 <= median_days <= 31:
        return "monthly"
    if float(median_days).is_integer():
        return f"{int(median_days)} days"
    return f"{median_days:.2f} days"


def _read_netcdf_values(dataset: Any, hints: Sequence[str], fuzzy_terms: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    names = list(dataset.variables.keys()) + [
        name for name in dataset.dimensions.keys() if name not in dataset.variables
    ]
    for name in _candidate_names(names, hints, fuzzy_terms):
        variable = dataset.variables.get(name)
        if variable is None:
            continue
        values = np.asarray(variable[:]).ravel()
        if values.size == 0:
            continue
        attrs = {
            "units": getattr(variable, "units", None),
            "calendar": getattr(variable, "calendar", None),
        }
        return values, attrs
    return np.array([]), {}


def _read_xarray_values(dataset: xr.Dataset, hints: Sequence[str], fuzzy_terms: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    names = list(dataset.coords) + [name for name in dataset.dims if name not in dataset.coords]
    for name in _candidate_names(names, hints, fuzzy_terms):
        if name in dataset.coords:
            variable = dataset.coords[name]
        elif name in dataset.dims and name in dataset:
            variable = dataset[name]
        else:
            continue
        values = np.asarray(variable.values).ravel()
        if values.size == 0:
            continue
        return values, dict(variable.attrs)
    return np.array([]), {}


def _metadata_from_file(path: Path) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    if _is_zarr_store_path(path):
        with xr.open_zarr(path, consolidated=None, chunks=None) as dataset:
            return {
                "depth": _read_xarray_values(dataset, DEPTH_NAME_HINTS, ("depth",)),
                "lon": _read_xarray_values(dataset, LON_NAME_HINTS, ("lon", "longitude")),
                "lat": _read_xarray_values(dataset, LAT_NAME_HINTS, ("lat", "latitude")),
                "time": _read_xarray_values(dataset, TIME_NAME_HINTS, ("time", "date")),
            }

    if NetCDFDataset is not None:
        with NetCDFDataset(path, "r") as dataset:
            return {
                "depth": _read_netcdf_values(dataset, DEPTH_NAME_HINTS, ("depth",)),
                "lon": _read_netcdf_values(dataset, LON_NAME_HINTS, ("lon", "longitude")),
                "lat": _read_netcdf_values(dataset, LAT_NAME_HINTS, ("lat", "latitude")),
                "time": _read_netcdf_values(dataset, TIME_NAME_HINTS, ("time", "date")),
            }

    with xr.open_dataset(path, decode_times=True, mask_and_scale=False) as dataset:
        return {
            "depth": _read_xarray_values(dataset, DEPTH_NAME_HINTS, ("depth",)),
            "lon": _read_xarray_values(dataset, LON_NAME_HINTS, ("lon", "longitude")),
            "lat": _read_xarray_values(dataset, LAT_NAME_HINTS, ("lat", "latitude")),
            "time": _read_xarray_values(dataset, TIME_NAME_HINTS, ("time", "date")),
        }


def discover_dataset_metadata(
    data_path: Path,
    max_files: int | None = None,
    *,
    all_files: bool = False,
    include_sentinel_depths: bool = False,
    sentinel_threshold: float = DEFAULT_SENTINEL_DEPTH_THRESHOLD,
) -> dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(f"Configured data_path does not exist: {data_path}")

    files = _iter_metadata_sources(data_path)
    if not all_files:
        files = _default_metadata_files(files)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No .nc files or .zarr stores found under data_path: {data_path}")

    depth_values: list[float] = []
    lon_values: list[float] = []
    lat_values: list[float] = []
    dates: list[str] = []
    used_files = 0
    skipped_files = 0

    for path in files:
        try:
            file_metadata = _metadata_from_file(path)
        except Exception as exc:  # pragma: no cover - surfaced in CLI summary.
            print(f"[warn] Skipping {path}: {exc}")
            skipped_files += 1
            continue

        file_used = False
        depth = _finite_numeric(file_metadata["depth"][0])
        lon = _finite_numeric(file_metadata["lon"][0])
        lat = _finite_numeric(file_metadata["lat"][0])
        time_values, time_attrs = file_metadata["time"]
        file_dates = _decode_time_values(
            time_values,
            units=time_attrs.get("units"),
            calendar=time_attrs.get("calendar"),
        )

        if depth.size:
            depth_values.extend(float(value) for value in depth)
            file_used = True
        if lon.size:
            lon_values.extend(float(value) for value in lon)
            file_used = True
        if lat.size:
            lat_values.extend(float(value) for value in lat)
            file_used = True
        if file_dates:
            dates.extend(file_dates)
            file_used = True

        if file_used:
            used_files += 1
        else:
            skipped_files += 1

    depth_levels, filtered_count = _normalize_depth_levels(
        depth_values,
        include_sentinel_depths=include_sentinel_depths,
        sentinel_threshold=sentinel_threshold,
    )
    spatial_extent: dict[str, list[float]] = {}
    lon_extent = _numeric_extent(lon_values)
    lat_extent = _numeric_extent(lat_values)
    if lon_extent is not None:
        spatial_extent["lon"] = lon_extent
    if lat_extent is not None:
        spatial_extent["lat"] = lat_extent

    temporal_extent = None
    if dates:
        sorted_dates = sorted(set(dates))
        temporal_extent = {"start": sorted_dates[0], "end": sorted_dates[-1]}

    spatial_resolution = _format_spatial_resolution(lon_values, lat_values)
    temporal_resolution = _format_temporal_resolution(dates)
    resolution: dict[str, str] = {}
    if spatial_resolution:
        resolution["spatial"] = spatial_resolution
    if temporal_resolution:
        resolution["temporal"] = temporal_resolution

    metadata: dict[str, Any] = {
        "used_files": used_files,
        "skipped_files": skipped_files,
        "filtered_sentinel_depths": filtered_count,
        "depth_levels": depth_levels,
        "depth_range": [depth_levels[0], depth_levels[-1]] if depth_levels else None,
        "spatial_extent": spatial_extent or None,
        "temporal_extent": temporal_extent,
        "resolution": resolution or None,
    }
    return metadata


def backfill_config_metadata(
    config_path: Path,
    *,
    dry_run: bool = False,
    backup: bool = True,
    max_files: int | None = None,
    all_files: bool = False,
    include_sentinel_depths: bool = False,
    sentinel_threshold: float = DEFAULT_SENTINEL_DEPTH_THRESHOLD,
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    payload = _read_yaml(config_path)
    raw_data_path = str(payload.get("data_path") or "").strip()
    if not raw_data_path:
        raise ValueError(f"Dataset config has no data_path: {config_path}")

    data_path = _resolve_data_path(raw_data_path, config_path)
    discovered = discover_dataset_metadata(
        data_path,
        max_files=max_files,
        all_files=all_files,
        include_sentinel_depths=include_sentinel_depths,
        sentinel_threshold=sentinel_threshold,
    )

    next_payload = dict(payload)
    if discovered["depth_levels"]:
        next_payload["depth_levels"] = discovered["depth_levels"]
        next_payload["depth_range"] = discovered["depth_range"]
    if discovered["spatial_extent"]:
        next_payload["spatial_extent"] = discovered["spatial_extent"]
    if discovered["temporal_extent"]:
        next_payload["temporal_extent"] = discovered["temporal_extent"]
    if discovered["resolution"]:
        existing_resolution = payload.get("resolution") if isinstance(payload.get("resolution"), dict) else {}
        next_payload["resolution"] = {**existing_resolution, **discovered["resolution"]}

    if not dry_run:
        if backup:
            backup_path = config_path.with_suffix(config_path.suffix + ".bak")
            shutil.copy2(config_path, backup_path)
        _write_yaml(config_path, next_payload)

    return {
        "config_path": str(config_path),
        "data_path": str(data_path),
        "dry_run": dry_run,
        **discovered,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill dataset_config.yaml depth, spatial extent, temporal extent, "
            "and resolution from NetCDF or Zarr metadata."
        ),
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to dataset_config.yaml. Defaults to configs/dataset_config.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered metadata without modifying the config file.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak file before writing.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Only inspect the first N NetCDF files or Zarr stores. Useful for quick checks.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help=(
            "Inspect every NetCDF file or Zarr store. By default, only the first and last source "
            "per yearly variable group are inspected."
        ),
    )
    parser.add_argument(
        "--include-sentinel-depths",
        action="store_true",
        help="Keep sentinel-like depth values such as 9999 instead of filtering them.",
    )
    parser.add_argument(
        "--sentinel-threshold",
        type=float,
        default=DEFAULT_SENTINEL_DEPTH_THRESHOLD,
        help="Absolute depth values >= this threshold are treated as sentinels unless included explicitly.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = backfill_config_metadata(
        Path(args.config_path),
        dry_run=args.dry_run,
        backup=not args.no_backup,
        max_files=args.max_files,
        all_files=args.all_files,
        include_sentinel_depths=args.include_sentinel_depths,
        sentinel_threshold=args.sentinel_threshold,
    )
    levels = result["depth_levels"]
    print(f"Config: {result['config_path']}")
    print(f"Data path: {result['data_path']}")
    print(f"Inspected files with usable metadata: {result['used_files']}")
    print(f"Skipped files without usable metadata: {result['skipped_files']}")
    print(f"Filtered sentinel depth values: {result['filtered_sentinel_depths']}")
    print(f"Depth levels: {len(levels)}")
    if result["depth_range"]:
        print(f"Depth range: {result['depth_range'][0]} to {result['depth_range'][1]} m")
        print(f"First levels: {levels[:8]}")
        print(f"Last levels: {levels[-8:]}")
    print(f"Spatial extent: {result['spatial_extent']}")
    print(f"Temporal extent: {result['temporal_extent']}")
    print(f"Resolution: {result['resolution']}")
    if result["dry_run"]:
        print("Dry run only: config was not modified.")
    else:
        print("Updated dataset metadata in the dataset config.")


if __name__ == "__main__":
    main()
