"""Lightweight partitioned xarray containers for multi-file ocean workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr


_SENTINEL_DEPTH_ABS_THRESHOLD = 9000.0


@dataclass(frozen=True)
class PartitionedDataArray:
    """A logical DataArray stored as ordered per-file/per-year DataArray parts."""

    partitions: Tuple[xr.DataArray, ...]
    partition_labels: Tuple[str, ...] = field(default_factory=tuple)
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "partitions", tuple(self.partitions))
        if not self.partitions:
            raise ValueError("PartitionedDataArray requires at least one partition")
        labels = tuple(self.partition_labels)
        if labels and len(labels) != len(self.partitions):
            raise ValueError("partition_labels length must match partitions")
        if not labels:
            labels = tuple(f"partition_{index}" for index in range(len(self.partitions)))
        object.__setattr__(self, "partition_labels", labels)
        if not self.attrs:
            object.__setattr__(self, "attrs", dict(getattr(self.partitions[0], "attrs", {})))

    @property
    def dims(self) -> Tuple[str, ...]:
        return tuple(self.partitions[0].dims)

    @property
    def sizes(self) -> Mapping[str, int]:
        return _logical_sizes(self.partitions)

    @property
    def shape(self) -> Tuple[int, ...]:
        sizes = self.sizes
        return tuple(int(sizes[dim]) for dim in self.dims)

    @property
    def size(self) -> int:
        return int(reduce(mul, self.shape, 1))

    @property
    def name(self) -> Optional[str]:
        return self.partitions[0].name

    @property
    def coords(self):
        return self.partitions[0].coords

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.partitions[0].coords

    def __getitem__(self, name: str) -> xr.DataArray:
        if name in self.partitions[0].coords:
            return _logical_coord(self.partitions, name)
        raise KeyError(name)

    def __getattr__(self, name: str) -> Any:
        if name in self.partitions[0].coords:
            return _logical_coord(self.partitions, name)
        raise AttributeError(name)

    def coord_ranges(self) -> Dict[str, Any]:
        return _partitioned_coord_ranges(self.partitions)

    def map_partitions(self, func) -> "PartitionedDataArray":
        mapped = tuple(func(partition) for partition in self.partitions)
        return PartitionedDataArray(
            mapped,
            partition_labels=self.partition_labels,
            attrs=dict(getattr(mapped[0], "attrs", {})) if mapped else dict(self.attrs),
        )


@dataclass(frozen=True)
class PartitionedDataset:
    """A logical Dataset stored as ordered per-file/per-year Dataset parts."""

    partitions: Tuple[xr.Dataset, ...]
    partition_labels: Tuple[str, ...] = field(default_factory=tuple)
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "partitions", tuple(self.partitions))
        if not self.partitions:
            raise ValueError("PartitionedDataset requires at least one partition")
        labels = tuple(self.partition_labels)
        if labels and len(labels) != len(self.partitions):
            raise ValueError("partition_labels length must match partitions")
        if not labels:
            labels = tuple(f"partition_{index}" for index in range(len(self.partitions)))
        object.__setattr__(self, "partition_labels", labels)
        if not self.attrs:
            object.__setattr__(self, "attrs", dict(getattr(self.partitions[0], "attrs", {})))

    @property
    def dims(self) -> Tuple[str, ...]:
        return tuple(self.partitions[0].sizes.keys())

    @property
    def sizes(self) -> Mapping[str, int]:
        return _logical_sizes(self.partitions)

    @property
    def shape(self) -> Tuple[int, ...]:
        sizes = self.sizes
        return tuple(int(sizes[dim]) for dim in self.dims)

    @property
    def size(self) -> int:
        return int(reduce(mul, self.shape, 1))

    @property
    def data_vars(self):
        return self.partitions[0].data_vars

    @property
    def coords(self):
        return self.partitions[0].coords

    def __contains__(self, name: object) -> bool:
        return (
            isinstance(name, str)
            and (name in self.partitions[0].data_vars or name in self.partitions[0].coords)
        )

    def __getitem__(self, name: Union[str, Sequence[str]]) -> Union[PartitionedDataArray, "PartitionedDataset", xr.DataArray]:
        if isinstance(name, str):
            if name in self.partitions[0].data_vars:
                arrays = tuple(partition[name] for partition in self.partitions)
                attrs = dict(getattr(arrays[0], "attrs", {})) if arrays else {}
                return PartitionedDataArray(arrays, partition_labels=self.partition_labels, attrs=attrs)
            if name in self.partitions[0].coords:
                return _logical_coord(self.partitions, name)
            raise KeyError(name)

        selected = tuple(partition[list(name)] for partition in self.partitions)
        return PartitionedDataset(
            selected,
            partition_labels=self.partition_labels,
            attrs=dict(getattr(selected[0], "attrs", {})) if selected else dict(self.attrs),
        )

    def __getattr__(self, name: str) -> Any:
        if name in self.partitions[0].coords:
            return _logical_coord(self.partitions, name)
        raise AttributeError(name)

    def coord_ranges(self) -> Dict[str, Any]:
        return _partitioned_coord_ranges(self.partitions)


PartitionedXarray = Union[PartitionedDataArray, PartitionedDataset]


def is_partitioned_xarray(value: Any) -> bool:
    return isinstance(value, (PartitionedDataArray, PartitionedDataset))


def iter_partitions(value: PartitionedXarray) -> Tuple[Union[xr.DataArray, xr.Dataset], ...]:
    return value.partitions


def partition_count(value: PartitionedXarray) -> int:
    return len(value.partitions)


def partition_labels(value: PartitionedXarray) -> Tuple[str, ...]:
    return value.partition_labels


def validate_compatible_partitioned_inputs(
    values: Sequence[Any],
    *,
    context: str = "partitioned inputs",
) -> Tuple[PartitionedXarray, ...]:
    partitioned_values = tuple(value for value in values if is_partitioned_xarray(value))
    if not partitioned_values:
        return tuple()

    expected_count = len(partitioned_values[0].partitions)
    expected_ranges = _partition_time_ranges(partitioned_values[0])
    for value in partitioned_values[1:]:
        if len(value.partitions) != expected_count:
            raise ValueError(f"{context} have different partition counts")
        ranges = _partition_time_ranges(value)
        if expected_ranges and ranges and ranges != expected_ranges:
            raise ValueError(f"{context} have different time partitions")
    return partitioned_values


def materialize_partitioned_xarray(value: Any) -> Any:
    """Explicitly concat a partitioned xarray object for tools without a streaming path."""
    if not is_partitioned_xarray(value):
        return value
    validate_compatible_partitioned_inputs((value,), context="partitioned input")
    return xr.concat(value.partitions, dim="time", coords="minimal", compat="override")


def materialize_partitioned_inputs(*values: Any, context: str = "partitioned inputs") -> Tuple[Any, ...]:
    validate_compatible_partitioned_inputs(values, context=context)
    return tuple(materialize_partitioned_xarray(value) for value in values)


def wrap_partitioned(
    partitions: Sequence[Union[xr.DataArray, xr.Dataset]],
    *,
    labels: Sequence[str],
) -> PartitionedXarray:
    if not partitions:
        raise ValueError("Cannot wrap an empty partition list")
    first = partitions[0]
    if isinstance(first, xr.Dataset):
        return PartitionedDataset(tuple(partitions), partition_labels=tuple(labels))
    return PartitionedDataArray(tuple(partitions), partition_labels=tuple(labels))


def partitioned_metadata(value: PartitionedXarray, source: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "source": source,
        "python_type": type(value).__name__,
        "dims": list(value.dims),
        "shape": list(value.shape),
        "n_partitions": partition_count(value),
    }
    if isinstance(value, PartitionedDataArray):
        metadata.update(
            {
                "variable": value.name or "unknown",
                "units": value.attrs.get("units", "") if isinstance(value.attrs, Mapping) else "",
            }
        )
    elif isinstance(value, PartitionedDataset):
        metadata["variables"] = list(value.data_vars.keys())
    return metadata


def _logical_sizes(partitions: Sequence[Union[xr.DataArray, xr.Dataset]]) -> Dict[str, int]:
    first_sizes = dict(partitions[0].sizes)
    if "time" not in first_sizes:
        return {name: int(size) for name, size in first_sizes.items()}

    logical = {name: int(size) for name, size in first_sizes.items()}
    logical["time"] = int(sum(int(partition.sizes.get("time", 0)) for partition in partitions))
    return logical


def _logical_coord(
    partitions: Sequence[Union[xr.DataArray, xr.Dataset]],
    coord_name: str,
) -> xr.DataArray:
    if coord_name == "time":
        values = [
            np.asarray(partition.coords[coord_name].values)
            for partition in partitions
            if coord_name in partition.coords
        ]
        coord = np.concatenate(values) if values else np.asarray([])
    else:
        coord = np.asarray(partitions[0].coords[coord_name].values)
    return xr.DataArray(coord, coords={coord_name: coord}, dims=(coord_name,), name=coord_name)


def _partitioned_coord_ranges(partitions: Sequence[Union[xr.DataArray, xr.Dataset]]) -> Dict[str, Any]:
    coord_ranges: Dict[str, Any] = {}
    for coord_name in ("time", "depth", "lat", "lon"):
        values = []
        for partition in partitions:
            if coord_name not in getattr(partition, "coords", {}):
                continue
            coord = partition.coords[coord_name].values
            if getattr(coord, "size", 0) == 0:
                continue
            values.append(coord)
        if not values:
            continue
        if coord_name == "time":
            first = values[0][0]
            last = values[-1][-1]
            coord_ranges["time_range"] = [str(first), str(last)]
        else:
            concatenated = np.concatenate([np.asarray(value, dtype=float).reshape(-1) for value in values])
            finite = concatenated[np.isfinite(concatenated)]
            if coord_name == "depth":
                finite = finite[np.abs(finite) < _SENTINEL_DEPTH_ABS_THRESHOLD]
            if finite.size == 0:
                continue
            coord_ranges[f"{coord_name}_range"] = [
                float(np.nanmin(finite)),
                float(np.nanmax(finite)),
            ]
    return coord_ranges


def _partition_time_ranges(value: PartitionedXarray) -> Tuple[Tuple[str, str], ...]:
    ranges = []
    for partition in value.partitions:
        if "time" not in getattr(partition, "coords", {}):
            return tuple()
        time_values = partition["time"].values
        if getattr(time_values, "size", 0) == 0:
            ranges.append(("", ""))
        else:
            ranges.append((str(time_values[0]), str(time_values[-1])))
    return tuple(ranges)


def find_partitioned_values(value: Any) -> Tuple[PartitionedXarray, ...]:
    found = []

    def _visit(item: Any) -> None:
        if is_partitioned_xarray(item):
            found.append(item)
        elif isinstance(item, Mapping):
            for child in item.values():
                _visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                _visit(child)

    _visit(value)
    return tuple(found)


def replace_partitioned_at_index(value: Any, index: int) -> Any:
    if is_partitioned_xarray(value):
        return value.partitions[index]
    if isinstance(value, dict):
        return {key: replace_partitioned_at_index(child, index) for key, child in value.items()}
    if isinstance(value, list):
        return [replace_partitioned_at_index(child, index) for child in value]
    if isinstance(value, tuple):
        return tuple(replace_partitioned_at_index(child, index) for child in value)
    return value
