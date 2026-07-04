from __future__ import annotations

import numpy as np
import xarray as xr

from data.zarr_utils import (
    encoding_for_dataset,
    format_bytes,
    format_chunks,
    normalize_chunks,
    normalize_dataset_dimensions,
)


def test_format_chunks() -> None:
    assert format_chunks({"time": 30, "lat": 110, "lon": 86}) == "time=30, lat=110, lon=86"


def test_format_bytes_units() -> None:
    assert format_bytes(12) == "12.00 B"
    assert format_bytes(1024) == "1.00 KiB"
    assert format_bytes(1024**2) == "1.00 MiB"
    assert format_bytes(1024**3) == "1.00 GiB"


def test_normalize_dataset_dimensions() -> None:
    dataset = xr.Dataset(
        {"temp": (("time", "z", "latitude", "longitude"), np.zeros((1, 2, 3, 4)))},
        coords={
            "time": [0],
            "z": [0, 10],
            "latitude": [20, 21, 22],
            "longitude": [110, 111, 112, 113],
        },
    )

    normalized = normalize_dataset_dimensions(dataset)

    assert normalized.temp.dims == ("time", "depth", "lat", "lon")
    assert set(normalized.coords) >= {"depth", "lat", "lon"}


def test_normalize_chunks_clamps_to_positive_defaults() -> None:
    assert normalize_chunks({"time": 0, "lat": 12, "ignored": 5}, {"time": 30, "lat": 10}) == {
        "time": 1,
        "lat": 12,
    }


def test_encoding_for_dataset_preserves_chunk_limits_without_compressor() -> None:
    dataset = xr.Dataset({"temp": (("time", "lat"), np.zeros((5, 3)))})

    encoding = encoding_for_dataset(
        dataset,
        chunks={"time": 10, "lat": 2},
        compressor_name="none",
        compression_level=3,
    )

    assert encoding["temp"] == {"chunks": (5, 2)}
