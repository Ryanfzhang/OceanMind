"""Retained EOF modes must be measured against the full field variance."""

import numpy as np
import pytest
import xarray as xr

from domain.ocean.analysis.advanced.eof import perform_eof_analysis


def test_retaining_one_mode_does_not_renormalize_it_to_100_percent():
    values = np.random.default_rng(12).normal(size=(8, 2, 2))
    field = xr.DataArray(values, dims=("time", "lat", "lon"), coords={
        "time": np.arange(8), "lat": [20.0, 21.0], "lon": [120.0, 121.0],
    })
    first = perform_eof_analysis(field, n_modes=1, weight_by_latitude=False)
    all_modes = perform_eof_analysis(field, n_modes=4, weight_by_latitude=False)

    assert first["modes"][0]["variance_explained"] < 100
    assert first["modes"][0]["variance_explained"] == all_modes["modes"][0]["variance_explained"]
    assert first["total_variance"] == all_modes["total_variance"]
    assert sum(mode["variance_explained"] for mode in all_modes["modes"]) == pytest.approx(100)
