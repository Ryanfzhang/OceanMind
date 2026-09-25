"""Land masks shared by ocean calculations and map previews."""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from typing import Optional

import numpy as np


def build_land_mask(*, lat: np.ndarray, lon: np.ndarray) -> Optional[np.ndarray]:
    """Return land pixels on a rectilinear grid when Natural Earth is available."""
    try:
        lat_key = tuple(float(value) for value in np.round(np.asarray(lat, dtype=float), 6))
        lon_key = tuple(float(value) for value in np.round(np.asarray(lon, dtype=float), 6))
        return _cached_land_mask(lat_key, lon_key)
    except Exception:
        return None


@lru_cache(maxsize=16)
def _cached_land_mask(lat_key: tuple[float, ...], lon_key: tuple[float, ...]) -> np.ndarray:
    from cartopy.io import shapereader
    import shapely

    lat = np.asarray(lat_key, dtype=float)
    lon = (np.asarray(lon_key, dtype=float) + 180) % 360 - 180
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    mask = np.zeros(lon_grid.shape, dtype=bool)
    domain = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))

    land_path = shapereader.natural_earth(resolution="10m", category="physical", name="land")
    for geometry in shapereader.Reader(land_path).geometries():
        bounds = geometry.bounds
        if (bounds[2] < domain[0] or bounds[0] > domain[2]
                or bounds[3] < domain[1] or bounds[1] > domain[3]):
            continue
        contains_xy = getattr(shapely, "contains_xy", None)
        if contains_xy is not None:
            mask |= contains_xy(geometry, lon_grid, lat_grid)
        else:
            from shapely import vectorized

            mask |= vectorized.contains(geometry, lon_grid, lat_grid)
    return mask


@lru_cache(maxsize=32)
def render_land_mask_image(
    lon_min: float, lon_max: float, lat_min: float, lat_max: float,
    width: int = 720, height: int = 520,
) -> str | None:
    """Encode a Web Mercator aligned, transparent-ocean land alpha mask."""
    if not (lon_max > lon_min and lat_max > lat_min and width > 0 and height > 0):
        return None
    from PIL import Image

    lon = lon_min + (np.arange(width) + 0.5) * (lon_max - lon_min) / width
    clipped = np.clip([lat_min, lat_max], -85.05112878, 85.05112878)
    mercator = np.log(np.tan(np.pi / 4 + np.deg2rad(clipped) / 2))
    y = mercator[1] - (np.arange(height) + 0.5) * (mercator[1] - mercator[0]) / height
    lat = np.rad2deg(2 * np.arctan(np.exp(y)) - np.pi / 2)
    land = build_land_mask(lat=lat, lon=lon)
    if land is None or not np.any(land):
        return None
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    image.putalpha(Image.fromarray(np.where(land, 255, 0).astype(np.uint8), mode="L"))
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")
