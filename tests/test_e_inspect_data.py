"""The agent can inspect only the data paths authorized for its run."""

import numpy as np
import pytest
import xarray as xr

from packages.agent_loop.ocean_tools import make_inspect_data


def test_inspection_supports_unknown_variable_and_denies_other_paths(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "new_project.nc"
    xr.Dataset({"new_tracer": (("time", "lat", "lon"),
                               np.arange(8, dtype=float).reshape(2, 2, 2))}).to_netcdf(source)
    outside = tmp_path / "private.nc"
    xr.Dataset({"secret": ("x", [1])}).to_netcdf(outside)
    (allowed / "escape.nc").symlink_to(outside)

    inspect = make_inspect_data([allowed])
    result = inspect(str(source), variable="new_tracer", max_sample_values=4)
    assert result["variables"][0]["name"] == "new_tracer"
    assert result["sample"]["sample_count"] <= 4
    with pytest.raises(ValueError, match="outside authorized"):
        inspect(str(outside))
    with pytest.raises(ValueError, match="outside authorized"):
        inspect(str(allowed / "escape.nc"))
