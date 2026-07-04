"""Helpers for keeping large analysis results array-backed until JSON output."""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import numpy as np


DEFAULT_RESULT_FULL_LIST_LIMIT = 100_000
DEFAULT_WORKSPACE_MAX_MATRIX_POINTS = 120_000


def result_full_list_limit() -> int:
    return _positive_int_env("OCEAN_RESULT_FULL_LIST_LIMIT", DEFAULT_RESULT_FULL_LIST_LIMIT)


def workspace_max_matrix_points() -> int:
    return _positive_int_env("OCEAN_WORKSPACE_MAX_MATRIX_POINTS", DEFAULT_WORKSPACE_MAX_MATRIX_POINTS)


def as_numeric_array(value: Any, *, dtype: Any = float) -> np.ndarray:
    """Return a numeric ndarray without forcing callers to know list/array/xarray shape."""
    if value is None:
        return np.asarray([], dtype=dtype)
    if hasattr(value, "values"):
        value = value.values
    try:
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return np.asarray([], dtype=dtype)


def array_shape(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "shape"):
        try:
            return [int(dim) for dim in value.shape]
        except Exception:
            return []
    try:
        return [int(dim) for dim in np.asarray(value).shape]
    except Exception:
        return []


def array_size(value: Any) -> int:
    if value is None:
        return 0
    if hasattr(value, "size"):
        try:
            return int(value.size)
        except Exception:
            return 0
    shape = array_shape(value)
    if not shape:
        return 0
    size = 1
    for dim in shape:
        size *= int(dim)
    return int(size)


def json_safe_array(value: np.ndarray, *, limit: Optional[int] = None) -> Any:
    """Serialize small arrays, but summarize large internal arrays at JSON boundaries."""
    resolved_limit = result_full_list_limit() if limit is None else max(0, int(limit))
    if int(value.size) <= resolved_limit:
        return value.tolist()
    return {
        "__array_omitted__": True,
        "shape": [int(dim) for dim in value.shape],
        "dtype": str(value.dtype),
        "size": int(value.size),
    }


def matrix_sample_indices(shape: tuple[int, int], *, max_points: Optional[int] = None) -> tuple[list[int], list[int]]:
    """Choose row/column indices so a 2D matrix stays under the workspace payload limit."""
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        return [], []
    limit = workspace_max_matrix_points() if max_points is None else max(1, int(max_points))
    if rows * cols <= limit:
        return list(range(rows)), list(range(cols))

    scale = math.sqrt((rows * cols) / limit)
    row_limit = max(1, min(rows, int(rows / scale)))
    col_limit = max(1, min(cols, int(cols / scale)))

    while row_limit * col_limit > limit and (row_limit > 1 or col_limit > 1):
        if row_limit >= col_limit and row_limit > 1:
            row_limit -= 1
        elif col_limit > 1:
            col_limit -= 1
        else:
            break

    return sample_indices(rows, row_limit), sample_indices(cols, col_limit)


def sample_indices(length: int, limit: int) -> list[int]:
    if length <= 0:
        return []
    if length <= limit:
        return list(range(length))
    raw = np.linspace(0, length - 1, num=max(1, int(limit)))
    return sorted({int(round(value)) for value in raw})


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return int(default)
    return value if value > 0 else int(default)
