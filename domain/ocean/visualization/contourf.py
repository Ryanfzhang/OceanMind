"""Render a spatial field as a contourf PNG image (base64-encoded)."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


_WEB_MERCATOR_MAX_LAT = 85.05112878


def _lat_to_web_mercator_y(lat: np.ndarray) -> np.ndarray:
    clipped = np.clip(lat, -_WEB_MERCATOR_MAX_LAT, _WEB_MERCATOR_MAX_LAT)
    radians = np.deg2rad(clipped)
    return np.log(np.tan(np.pi / 4.0 + radians / 2.0))


def render_contourf_image(
    lon: List[float],
    lat: List[float],
    values: List[List[float]],
    variable: str = "",
    units: str = "",
    colormap: str = "ocean_diverging",
    n_levels: int = 20,
    dpi: int = 150,
    transparent: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> str:
    """Return a base64-encoded PNG of a matplotlib contourf plot.

    Parameters
    ----------
    lon, lat : 1-D coordinate arrays.
    values : 2-D field (lat × lon).
    colormap : matplotlib colormap name.
    n_levels : number of contour levels.
    dpi : output resolution.
    transparent : whether the PNG background is transparent.

    Returns
    -------
    Base64-encoded PNG string suitable for a ``data:image/png;base64,…`` URI.
    """
    _prepare_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    val_arr = np.ma.masked_invalid(np.asarray(values, dtype=float))
    mercator_y = _lat_to_web_mercator_y(lat_arr)
    cmap = _resolve_colormap(colormap, LinearSegmentedColormap)
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    levels = n_levels
    if finite_values.size:
        lower = float(vmin) if vmin is not None and np.isfinite(vmin) else float(np.nanmin(finite_values))
        upper = float(vmax) if vmax is not None and np.isfinite(vmax) else float(np.nanmax(finite_values))
        if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
            levels = np.linspace(lower, upper, n_levels)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(lon_arr, mercator_y, val_arr, levels=levels, cmap=cmap, extend="both")
    ax.set_xlim(lon_arr.min(), lon_arr.max())
    ax.set_ylim(mercator_y.min(), mercator_y.max())
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        transparent=transparent,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def render_multiregion_contourf_image(
    lon: Sequence[float],
    lat: Sequence[float],
    values: Any,
    regions: Sequence[Dict[str, Any]],
    *,
    n_levels: int = 20,
    dpi: int = 150,
    transparent: bool = True,
) -> str:
    """Return one transparent PNG with multiple masked contourf regions.

    Each region can use its own mask, color scale, and colormap. This is used
    for Fig10-style transport maps where WPO and China Seas must be rendered as
    two independent contourf fields on the same map extent.
    """
    _prepare_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    val_arr = np.asarray(values, dtype=float)
    if lon_arr.size == 0 or lat_arr.size == 0 or val_arr.ndim != 2:
        return ""
    if val_arr.shape != (lat_arr.size, lon_arr.size):
        return ""

    mercator_y = _lat_to_web_mercator_y(lat_arr)
    fig, ax = plt.subplots(figsize=(8, 6))
    rendered_any = False

    for region in regions:
        try:
            mask_arr = np.asarray(region.get("mask"), dtype=bool)
        except Exception:
            continue
        if mask_arr.shape != val_arr.shape:
            continue
        region_values = np.ma.masked_where(~mask_arr | ~np.isfinite(val_arr), val_arr)
        finite_values = np.asarray(region_values.compressed(), dtype=float)
        if finite_values.size == 0:
            continue

        lower = region.get("vmin")
        upper = region.get("vmax")
        lower = float(lower) if lower is not None and np.isfinite(float(lower)) else float(np.nanmin(finite_values))
        upper = float(upper) if upper is not None and np.isfinite(float(upper)) else float(np.nanmax(finite_values))
        if not np.isfinite(lower) or not np.isfinite(upper):
            continue
        if upper <= lower:
            pad = max(abs(lower) * 1e-6, 1e-9)
            lower -= pad
            upper += pad

        levels = np.linspace(lower, upper, max(2, int(n_levels)))
        cmap = _resolve_colormap(str(region.get("colormap") or "ocean_diverging"), LinearSegmentedColormap)
        ax.contourf(
            lon_arr,
            mercator_y,
            region_values,
            levels=levels,
            cmap=cmap,
            extend="both",
            antialiased=True,
        )
        rendered_any = True

    if not rendered_any:
        plt.close(fig)
        return ""

    ax.set_xlim(lon_arr.min(), lon_arr.max())
    ax.set_ylim(mercator_y.min(), mercator_y.max())
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        transparent=transparent,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _prepare_matplotlib_cache() -> None:
    mpl_config = Path(os.environ.get("MPLCONFIGDIR", "/tmp/oceanmaster-mplconfig"))
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", "/tmp/oceanmaster-cache"))
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def _resolve_colormap(name: str, colormap_cls):
    normalized = str(name or "").strip().lower()
    if normalized in {"blue_white_red", "transport_blue_red"}:
        stops = [
            (0.0, (33 / 255, 102 / 255, 172 / 255)),
            (0.18, (67 / 255, 147 / 255, 195 / 255)),
            (0.34, (146 / 255, 197 / 255, 222 / 255)),
            (0.5, (247 / 255, 247 / 255, 247 / 255)),
            (0.66, (244 / 255, 165 / 255, 130 / 255)),
            (0.82, (214 / 255, 96 / 255, 77 / 255)),
            (1.0, (178 / 255, 24 / 255, 43 / 255)),
        ]
        return colormap_cls.from_list(normalized, stops)
    if normalized != "ocean_diverging":
        return name
    stops = [
        (0.0, (112 / 255, 0 / 255, 168 / 255)),
        (0.125, (88 / 255, 42 / 255, 214 / 255)),
        (0.25, (35 / 255, 98 / 255, 221 / 255)),
        (0.375, (37 / 255, 185 / 255, 225 / 255)),
        (0.5, (244 / 255, 255 / 255, 246 / 255)),
        (0.625, (255 / 255, 244 / 255, 72 / 255)),
        (0.75, (255 / 255, 177 / 255, 45 / 255)),
        (0.875, (229 / 255, 72 / 255, 30 / 255)),
        (1.0, (164 / 255, 24 / 255, 12 / 255)),
    ]
    return colormap_cls.from_list("ocean_diverging", stops)
