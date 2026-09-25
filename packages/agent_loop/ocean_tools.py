"""A skill-independent way to inspect the existing ocean tool catalog."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from packages.analysis_runtime.catalog import find_tools
from packages.analysis_runtime.inspection import inspect_data


FIND_TOOLS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "find_tools",
        "description": (
            "Find existing ocean Python tools and inspect their actual arguments, "
            "defaults, and return conventions. Available without reading a skill."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


INSPECT_DATA_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_data",
        "description": "Inspect metadata and a bounded sample of an authorized NetCDF or Zarr source.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Path within an authorized data root."},
                "variable": {"type": "string"},
                "max_sample_values": {"type": "integer", "minimum": 1, "maximum": 64},
            },
            "additionalProperties": False,
        },
    },
}


def make_inspect_data(allowed_roots: Iterable[str | Path]) -> Callable[..., dict[str, Any]]:
    """Expose only explicitly allowed sources to the read-only inspector."""
    roots = tuple(Path(root).expanduser().resolve(strict=True) for root in allowed_roots)
    if not roots:
        raise ValueError("At least one authorized data root is required")

    def authorized(source: str | None = None, *, variable: str | None = None,
                   max_sample_values: int = 16) -> dict[str, Any]:
        if source is None:
            if len(roots) != 1:
                raise ValueError("Choose one authorized data source")
            path = roots[0]
        else:
            path = Path(source).expanduser().resolve()
        if not any(path == root or root.is_dir() and path.is_relative_to(root) for root in roots):
            raise ValueError("Data source is outside authorized roots")
        return inspect_data(path, variable=variable, max_sample_values=max_sample_values)

    return authorized


__all__ = [
    "FIND_TOOLS_SCHEMA", "INSPECT_DATA_SCHEMA", "find_tools", "make_inspect_data",
]
