"""
Dataset assembly helpers for orchestrated workflows.
"""

import json
from typing import Any, Dict, Optional

import numpy as np
import xarray as xr

from domain.ocean.data_access.partitioned import find_partitioned_values, materialize_partitioned_xarray


def assemble_dataset(
    variables: Optional[Dict[str, xr.DataArray]] = None,
    arrays: Optional[Dict[str, xr.DataArray]] = None,
    fields: Optional[Dict[str, xr.DataArray]] = None,
) -> xr.Dataset:
    """
    Assemble multiple named DataArray objects into a single Dataset.

    Args:
        variables: Preferred mapping from variable name to DataArray.
        arrays: Backward-compatible alias for variables.
        fields: Backward-compatible alias for variables.

    Returns:
        Aligned xarray Dataset.
    """
    if find_partitioned_values((variables, arrays, fields)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="assemble_dataset",
            tool_func=assemble_dataset,
            params={"variables": variables, "arrays": arrays, "fields": fields},
        )

    if variables is None:
        variables = arrays
    if variables is None:
        variables = fields

    variables = _normalize_variable_mapping(variables)

    if not variables:
        raise ValueError("assemble_dataset requires at least one variable")

    aligned = xr.align(*variables.values(), join="inner")
    dataset_vars = {
        name: array.rename(name)
        for (name, _), array in zip(variables.items(), aligned)
    }
    return _drop_sentinel_depth_coordinates(xr.Dataset(dataset_vars))


def _normalize_variable_mapping(variables: Any) -> Dict[str, xr.DataArray]:
    """Coerce common planner output variants into the expected mapping."""
    if variables is None:
        return {}

    if isinstance(variables, xr.Dataset):
        return {name: data for name, data in variables.data_vars.items()}

    if isinstance(variables, xr.DataArray):
        return {variables.name or "var_1": variables}

    if isinstance(variables, dict):
        return {
            str(name): _coerce_data_array(value, fallback_name=str(name))
            for name, value in variables.items()
        }

    if isinstance(variables, (list, tuple)):
        if all(isinstance(item, tuple) and len(item) == 2 for item in variables):
            return {
                str(name): _coerce_data_array(value, fallback_name=str(name))
                for name, value in variables
            }

        if all(isinstance(item, dict) and "name" in item for item in variables):
            normalized: Dict[str, xr.DataArray] = {}
            for item in variables:
                name = str(item["name"])
                value = item.get("data", item.get("array", item.get("value")))
                normalized[name] = _coerce_data_array(value, fallback_name=name)
            return normalized

        normalized = {}
        for index, value in enumerate(variables, start=1):
            array = _coerce_data_array(value, fallback_name=f"var_{index}")
            normalized[array.name or f"var_{index}"] = array
        return normalized

    if isinstance(variables, str):
        stripped = variables.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            parsed = json.loads(stripped)
            return _normalize_variable_mapping(parsed)
        raise TypeError(
            "assemble_dataset expects a variable mapping like "
            "{\"u\": $ref:u_field.data}. Received a string instead."
        )

    raise TypeError(
        f"assemble_dataset received unsupported input type: {type(variables).__name__}"
    )


def _coerce_data_array(value: Any, fallback_name: str) -> xr.DataArray:
    value = materialize_partitioned_xarray(value)

    if isinstance(value, xr.DataArray):
        if not value.name:
            value = value.rename(fallback_name)
        return value

    if isinstance(value, xr.Dataset):
        if len(value.data_vars) != 1:
            raise TypeError(
                "assemble_dataset cannot infer a single variable from a Dataset "
                f"with {len(value.data_vars)} variables."
            )
        name = next(iter(value.data_vars))
        data = value[name]
        return data.rename(fallback_name or name)

    raise TypeError(
        "assemble_dataset expects DataArray inputs after reference resolution. "
        f"Received {type(value).__name__} for '{fallback_name}'."
    )


def _drop_sentinel_depth_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    if "depth" not in dataset.dims or "depth" not in dataset.coords:
        return dataset

    try:
        depth_values = np.asarray(dataset["depth"].values, dtype=float)
    except (TypeError, ValueError):
        return dataset

    valid_mask = np.isfinite(depth_values) & (np.abs(depth_values) < 9000.0)
    if valid_mask.size == 0 or valid_mask.all() or not valid_mask.any():
        return dataset

    return dataset.isel(depth=np.where(valid_mask)[0])
