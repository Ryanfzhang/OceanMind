"""Dask/Zarr execution helpers for OceanMaster tools.

The helpers in this module keep large Zarr-backed arrays lazy through the
physics/reduction steps, then make the final compute boundary explicit so the
agent can stream progress while Dask executes the task graph.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping, Sequence, TypeVar

import numpy as np
import xarray as xr

from packages.tool_loader.progress import report_tool_progress


T = TypeVar("T")


def is_dask_backed(value: Any) -> bool:
    """Return True if an xarray object or nested collection contains Dask arrays."""
    if isinstance(value, xr.DataArray):
        return getattr(value.data, "chunks", None) is not None
    if isinstance(value, xr.Dataset):
        return any(is_dask_backed(array) for array in value.data_vars.values())
    if isinstance(value, Mapping):
        return any(is_dask_backed(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(is_dask_backed(item) for item in value)
    return False


def chunk_summary(value: Any) -> dict[str, Any]:
    """Small, JSON-safe chunk metadata for result/debug payloads."""
    if isinstance(value, xr.DataArray):
        chunks = getattr(value.data, "chunks", None)
        return {
            "dims": list(value.dims),
            "shape": [int(value.sizes[dim]) for dim in value.dims],
            "chunks": _format_dask_chunks(value.dims, chunks),
        }
    if isinstance(value, xr.Dataset):
        return {
            name: chunk_summary(array)
            for name, array in value.data_vars.items()
        }
    return {}


def report_phase(
    *,
    phase: str,
    message: str,
    percent: float | None = None,
    completed_units: int | None = None,
    total_units: int | None = None,
    unit_label: str | None = None,
    current_unit: str | None = None,
    compute_backend: str | None = None,
    chunks: Any | None = None,
) -> None:
    """Centralize tool progress payload shape across domain modules.

    This wrapper is intentionally kept because many domain tools emit progress
    and the frontend depends on a stable set of progress keys.
    """
    report_tool_progress(
        phase=phase,
        message=message,
        percent=percent,
        completed_units=completed_units,
        total_units=total_units,
        unit_label=unit_label,
        current_unit=current_unit,
        compute_backend=compute_backend,
        chunks=chunks,
    )


def compute_with_progress(
    value: T,
    *,
    label: str,
    start: float = 0.0,
    end: float = 1.0,
) -> T:
    """Compute a Dask-backed object with throttled tool progress events."""
    if not is_dask_backed(value):
        return value

    try:
        from dask.callbacks import Callback
        import dask
    except ImportError:
        report_phase(
            phase="computing",
            message=f"Computing {label}",
            percent=start,
            compute_backend="xarray",
        )
        return value.compute()  # type: ignore[return-value, union-attr]

    progress = _DaskProgressReporter(label=label, start=start, end=end)
    report_phase(
        phase="compute_graph_prepared",
        message=f"Prepared Dask graph for {label}",
        percent=start,
        unit_label="task",
        compute_backend="dask",
        chunks=chunk_summary(value),
    )
    with Callback(
        start=progress.start_graph,
        posttask=progress.posttask,
        finish=progress.finish_graph,
    ):
        if isinstance(value, tuple):
            computed = dask.compute(*value)
        elif isinstance(value, list):
            computed = list(dask.compute(*value))
        else:
            computed = value.compute()  # type: ignore[union-attr]
    report_phase(
        phase="compute_complete",
        message=f"Computed {label}",
        percent=end,
        compute_backend="dask",
    )
    return computed  # type: ignore[return-value]


def compute_together_with_progress(
    values: Sequence[T],
    *,
    label: str,
    start: float = 0.0,
    end: float = 1.0,
) -> tuple[T, ...]:
    """Compute multiple xarray objects in one shared Dask graph."""
    if not any(is_dask_backed(value) for value in values):
        return tuple(values)
    computed = compute_with_progress(tuple(values), label=label, start=start, end=end)
    return tuple(computed)  # type: ignore[arg-type]


def dataarray_to_numpy(
    data: xr.DataArray,
    *,
    label: str,
    dtype: Any = float,
    start: float = 0.0,
    end: float = 1.0,
) -> np.ndarray:
    """Compute a final DataArray boundary and return a NumPy array."""
    computed = compute_with_progress(data, label=label, start=start, end=end)
    return np.asarray(computed.values, dtype=dtype)


def _format_dask_chunks(dims: Iterable[str], chunks: Any) -> dict[str, list[int]] | None:
    if chunks is None:
        return None
    formatted: dict[str, list[int]] = {}
    for dim, dim_chunks in zip(dims, chunks):
        formatted[str(dim)] = [int(value) for value in dim_chunks]
    return formatted


class _DaskProgressReporter:
    def __init__(self, *, label: str, start: float, end: float) -> None:
        self.label = label
        self.start = max(0.0, min(1.0, float(start)))
        self.end = max(self.start, min(1.0, float(end)))
        self.total = 0
        self.completed = 0
        self._next_report_fraction = 0.0
        self._last_report_time = 0.0

    def start_graph(self, dsk: Any) -> None:
        try:
            self.total = max(1, len(dsk))
        except Exception:
            self.total = 1
        self.completed = 0
        self._next_report_fraction = 0.0
        self._last_report_time = 0.0
        report_phase(
            phase="computing",
            message=f"Computing {self.label}",
            percent=self.start,
            completed_units=0,
            total_units=self.total,
            unit_label="task",
            compute_backend="dask",
        )

    def posttask(self, key: Any, result: Any, dsk: Any, state: Any, worker_id: Any) -> None:
        self.completed += 1
        if not self._should_report():
            return
        fraction = min(1.0, self.completed / max(1, self.total))
        report_phase(
            phase="computing",
            message=f"Computing {self.label}",
            percent=self.start + (self.end - self.start) * fraction,
            completed_units=self.completed,
            total_units=self.total,
            unit_label="task",
            compute_backend="dask",
        )

    def finish_graph(self, dsk: Any, state: Any, errored: bool) -> None:
        if errored:
            report_phase(
                phase="compute_failed",
                message=f"Failed while computing {self.label}",
                percent=None,
                completed_units=self.completed,
                total_units=self.total or None,
                unit_label="task",
                compute_backend="dask",
            )

    def _should_report(self) -> bool:
        total = max(1, self.total)
        fraction = self.completed / total
        now = time.monotonic()
        if self.completed >= total:
            return True
        if fraction >= self._next_report_fraction:
            self._next_report_fraction = min(1.0, fraction + 0.05)
            self._last_report_time = now
            return True
        if now - self._last_report_time >= 2.0:
            self._last_report_time = now
            return True
        return False
