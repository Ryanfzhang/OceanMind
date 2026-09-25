"""Bounded inspection of a local NetCDF or Zarr data source."""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from packages.runtime.dataset_config import get_active_dataset_config


def _value(value: Any) -> Any:
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value, unit="auto"))
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, bool)):
        return value if not isinstance(value, str) else value[:120]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return str(value)[:120]


def _coordinate(name: str, coordinate: xr.DataArray) -> dict[str, Any]:
    first = coordinate.isel({dim: 0 for dim in coordinate.dims}).values
    last = coordinate.isel({dim: -1 for dim in coordinate.dims}).values
    return {
        "name": name,
        "dims": list(coordinate.dims),
        "shape": list(coordinate.shape),
        "units": _value(coordinate.attrs.get("units", "")),
        "first": _value(first[()]),
        "last": _value(last[()]),
    }


def _window_lengths(data: xr.DataArray, budget: int) -> list[int]:
    lengths = [1] * data.ndim
    while data.ndim:
        advanced = False
        for i, size in enumerate(data.shape):
            if lengths[i] < size and np.prod(lengths) // lengths[i] * (lengths[i] + 1) <= budget:
                lengths[i] += 1
                advanced = True
        if not advanced:
            break
    return lengths


def _sample(data: xr.DataArray, max_values: int) -> dict[str, Any]:
    budgets = [max_values] if max_values == 1 else [max_values // 2, max_values - max_values // 2]
    windows: list[dict[str, list[int]]] = []
    chunks: list[np.ndarray] = []
    for offset, budget in enumerate(budgets):
        lengths = _window_lengths(data, budget)
        spatial_dims = {"lat", "latitude", "lon", "longitude", "x", "y"}
        starts = [
            max(0, (size - length) // 2) if offset and dim.lower() in spatial_dims else 0
            for dim, size, length in zip(data.dims, data.shape, lengths)
        ]
        selection = {dim: slice(start, start + length) for dim, start, length in zip(data.dims, starts, lengths)}
        windows.append({dim: [start, start + length] for dim, start, length in zip(data.dims, starts, lengths)})
        chunks.append(np.asarray(data.isel(selection).values).reshape(-1))
    values = np.concatenate(chunks)
    missing = np.asarray(xr.DataArray(values).isnull().values)
    observed = values[~missing]
    result: dict[str, Any] = {
        "variable": data.name,
        "index_windows": windows,
        "sample_count": int(values.size),
        "missing_count": int(missing.sum()),
        "valid_count": int(observed.size),
        "values": [_value(value) for value in values],
        "scope": "sample only; not whole-dataset statistics",
    }
    if observed.size and values.dtype.kind in "iuf":
        result["observed_valid_range"] = [float(np.min(observed)), float(np.max(observed))]
    else:
        result["observed_valid_range"] = None
    return result


def inspect_data(
    source: str | Path | None = None,
    *,
    variable: str | None = None,
    max_variables: int = 24,
    max_coordinates: int = 12,
    max_sample_values: int = 16,
) -> dict[str, Any]:
    """Read metadata and one small data window without materializing whole fields."""
    if not 1 <= max_variables <= 100 or not 1 <= max_coordinates <= 40:
        raise ValueError("variable/coordinate limits must be positive and bounded")
    if not 1 <= max_sample_values <= 64:
        raise ValueError("max_sample_values must be between 1 and 64")
    path = Path(source or get_active_dataset_config().data_path).expanduser().resolve()
    if path.is_dir() and path.suffix != ".zarr":
        candidates = (p for p in path.iterdir() if p.suffix in {".nc", ".zarr"})
        listed = list(islice(candidates, max_variables + 1))
        return {
            "source": str(path), "kind": "directory",
            "sources": [str(p) for p in listed[:max_variables]],
            "truncated": len(listed) > max_variables,
        }
    if not path.exists():
        raise FileNotFoundError(path)
    opener = xr.open_zarr if path.suffix == ".zarr" else xr.open_dataset
    with opener(path) as dataset:
        names = list(islice(dataset.data_vars, max_variables + 1))
        coordinates = list(islice(dataset.coords, max_coordinates + 1))
        if variable is not None and variable not in dataset.data_vars:
            raise ValueError(f"Variable {variable!r} is not a data variable in {path.name}")
        selected = variable or (names[0] if names else None)
        listed_names = names[:max_variables]
        if variable and variable not in listed_names:
            listed_names = [*listed_names[:-1], variable]
        return {
            "source": str(path),
            "kind": "zarr" if path.suffix == ".zarr" else "netcdf",
            "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
            "variables": [
                {
                    "name": name, "dims": list(dataset[name].dims),
                    "shape": list(dataset[name].shape), "dtype": str(dataset[name].dtype),
                    "units": _value(dataset[name].attrs.get("units", "")),
                }
                for name in listed_names
            ],
            "variables_truncated": len(names) > max_variables,
            "coordinates": [_coordinate(name, dataset[name]) for name in coordinates[:max_coordinates]],
            "coordinates_truncated": len(coordinates) > max_coordinates,
            "sample": _sample(dataset[selected], max_sample_values) if selected else None,
        }
