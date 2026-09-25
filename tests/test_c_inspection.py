"""Small real NetCDF and real tool discovery checks for C2–C4."""

import json

import numpy as np
import xarray as xr

from packages.analysis_runtime.catalog import find_tools
from packages.analysis_runtime.inspection import inspect_data


def test_inspect_unregistered_variable_with_bounded_sample(tmp_path):
    values = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)
    values[0, 0, 0, 0] = np.nan
    dataset = xr.Dataset(
        {"new_oxygen": (("time", "depth", "lat", "lon"), values, {"units": "mmol m-3"})},
        coords={"time": [0, 1], "depth": [0, 20, 50], "lat": [10, 11, 12, 13],
                "lon": [110, 111, 112, 113, 114]},
    )
    path = tmp_path / "new_project.nc"
    dataset.to_netcdf(path)

    result = inspect_data(path, variable="new_oxygen", max_sample_values=8)

    assert result["dimensions"]["depth"] == 3
    assert result["variables"][0]["name"] == "new_oxygen"
    assert result["variables"][0]["units"] == "mmol m-3"
    assert next(c for c in result["coordinates"] if c["name"] == "lon")["last"] == 114
    assert result["sample"]["sample_count"] <= 8
    assert result["sample"]["missing_count"] == 1
    assert result["sample"]["valid_count"] == result["sample"]["sample_count"] - 1
    assert result["sample"]["observed_valid_range"] is not None
    assert len(json.dumps(result)) < 4000


def test_directory_listing_and_unknown_variable(tmp_path):
    path = tmp_path / "fresh.nc"
    xr.Dataset({"a": ("x", [1, 2]), "new_variable": ("x", [3, 4])}).to_netcdf(path)
    listing = inspect_data(tmp_path)
    assert listing["sources"] == [str(path)]
    assert not listing["truncated"]
    selected = inspect_data(path, variable="new_variable", max_variables=1)
    assert selected["variables"][0]["name"] == "new_variable"
    assert selected["variables_truncated"]
    try:
        inspect_data(path, variable="missing")
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("unknown variable should raise")


def test_find_tools_uses_real_signatures_and_result_conventions():
    loader = find_tools("load_dataset")["tools"][0]
    assert loader["name"] == "load_dataset"
    params = {item["name"]: item for item in loader["parameters"]}
    assert params["variable"]["required"]
    assert params["lon_range"]["required"]
    assert params["data_path"]["default"] is None
    assert "DataArray" in loader["python_return_type"]
    assert "legacy_registry_output_type" not in loader

    eddy = find_tools("detect_eddies")["tools"][0]
    params = {item["name"]: item for item in eddy["parameters"]}
    assert params["ow_threshold"]["default"] == -2e-12
    assert params["min_pixels"]["default"] == 10
    assert {"events", "statistics", "mask"} <= set(eddy["literal_dict_return_keys"])
    assert "legacy_registry_output_type" not in eddy

    trend = find_tools("compute_trend")["tools"][0]
    assert trend["module"] == "domain.ocean.analysis.timeseries.extract"
    assert {item["name"]: item for item in trend["parameters"]}["method"]["default"] == "linear"
    assert find_tools("no_such_tool_xyz")["total_matches"] == 0
