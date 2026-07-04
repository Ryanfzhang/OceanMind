from __future__ import annotations

import numpy as np
import xarray as xr

from domain.ocean.data_access.load import get_depth_dim


def test_get_depth_dim_detects_depth_dimension() -> None:
    data = xr.DataArray(np.zeros((2, 3)), dims=("time", "depth"))

    assert get_depth_dim(data) == "depth"


def test_get_depth_dim_detects_z_dimension() -> None:
    data = xr.DataArray(np.zeros((2, 3)), dims=("time", "z"))

    assert get_depth_dim(data) == "z"


def test_get_depth_dim_detects_lev_dimension() -> None:
    data = xr.DataArray(np.zeros((2, 3)), dims=("time", "lev"))

    assert get_depth_dim(data) == "lev"


def test_get_depth_dim_returns_none_without_depth_dimension() -> None:
    data = xr.DataArray(np.zeros((2, 3)), dims=("time", "lat"))

    assert get_depth_dim(data) is None
