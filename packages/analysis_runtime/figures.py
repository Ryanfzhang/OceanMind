"""Small, labelled PNG figures that can be saved as analysis artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Mapping


@dataclass(frozen=True)
class PngFigure:
    png_bytes: bytes
    metadata: dict[str, Any]


def png_figure(figure: Any, *, metadata: Mapping[str, Any]) -> PngFigure:
    """Render a matplotlib Figure without a display or an implicit global pyplot state."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    output = BytesIO()
    FigureCanvasAgg(figure).print_png(output)
    return PngFigure(output.getvalue(), dict(metadata))


def render_eddy_figure(
    result: Mapping[str, Any], *, date: str, depth: str | float | None = None,
) -> PngFigure:
    """Plot one actual detect_eddies result, separating candidates from accepted events.

    The detector's ``mask`` is only an OW-threshold candidate mask. Events have
    passed its component-size and radius filters; they get distinct symbols.
    ``date`` and optional ``depth`` describe the slice selected before detection.
    """
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    if not isinstance(date, str) or not date.strip():
        raise ValueError("A date or time label is required for an eddy figure")
    coordinates = result["coordinates"]
    lon = np.asarray(coordinates["lon"], dtype=float)
    lat = np.asarray(coordinates["lat"], dtype=float)
    mask = np.asarray(result["mask"], dtype=bool)
    ow = np.asarray(result["ow_field"], dtype=float)
    if (lon.ndim != 1 or lat.ndim != 1 or len(lon) < 2 or len(lat) < 2
            or mask.shape != (len(lat), len(lon)) or ow.shape != mask.shape
            or not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat))):
        raise ValueError("Eddy result must have aligned, finite 1D lon/lat coordinates")
    if not np.any(np.isfinite(ow)):
        raise ValueError("Eddy OW field has no finite values to plot")

    figure = Figure(figsize=(9, 6), dpi=130, constrained_layout=True)
    axis = figure.add_subplot(111)
    finite = ow[np.isfinite(ow)]
    bound = max(float(np.nanpercentile(np.abs(finite), 98)), float(np.finfo(float).tiny))
    colors = axis.pcolormesh(lon, lat, ow, cmap="RdBu_r", shading="auto",
                             vmin=-bound, vmax=bound, rasterized=True)
    figure.colorbar(colors, ax=axis, label="Okubo–Weiss (detector native units)")
    # An outline remains visually distinct over both positive and negative OW.
    if np.any(mask) and not np.all(mask):
        axis.contour(lon, lat, mask.astype(float), levels=[0.5],
                     colors=["#efb649"], linewidths=2.4)
    elif np.all(mask):
        axis.add_patch(Rectangle((float(lon.min()), float(lat.min())),
                                 float(lon.max() - lon.min()),
                                 float(lat.max() - lat.min()), fill=False,
                                 edgecolor="#efb649", linewidth=2.4))

    counts = {"cyclonic": 0, "anticyclonic": 0}
    styles = {"cyclonic": ("o", "#1f4a9d"),
              "anticyclonic": ("^", "#a32834")}
    for event in result["events"]:
        kind = event.get("type")
        if kind not in styles:
            raise ValueError(f"Unknown eddy event type: {kind}")
        center = event["center"]
        marker, color = styles[kind]
        axis.scatter(float(center["lon"]), float(center["lat"]), s=80,
                     marker=marker, facecolor="white", edgecolor=color,
                     linewidth=1.8, zorder=5)
        counts[kind] += 1

    legend = [Line2D([], [], color="#efb649", linewidth=2.4,
                     label=f"OW threshold candidates (outline; {int(mask.sum())} pixels)")]
    for kind, label in (("cyclonic", "Accepted cyclonic"),
                        ("anticyclonic", "Accepted anticyclonic")):
        marker, color = styles[kind]
        legend.append(Line2D([], [], marker=marker, linestyle="None",
                             markerfacecolor="white", markeredgecolor=color,
                             markersize=8, label=f"{label} (n={counts[kind]})"))
    axis.legend(handles=legend, loc="best", framealpha=0.9)
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    axis.set_title(f"Eddy detection — {date}" + (f", depth {depth}" if depth is not None else ""))
    axis.set_xlim(float(lon.min()), float(lon.max()))
    axis.set_ylim(float(lat.min()), float(lat.max()))
    axis.grid(alpha=0.2)

    params = result.get("statistics", {}).get("detection_params", {})
    metadata = {
        "variable": "Okubo–Weiss parameter",
        "units": "detector native units",
        "date": date,
        "depth": depth,
        "extent": {"lon": [float(lon.min()), float(lon.max())],
                   "lat": [float(lat.min()), float(lat.max())]},
        "color_scale": {"min": -bound, "max": bound,
                        "meaning": "Okubo–Weiss parameter"},
        "overlay": "Amber outline encloses OW-threshold candidates, not accepted events",
        "accepted_events": counts,
        "detection_params": params,
    }
    try:
        return png_figure(figure, metadata=metadata)
    finally:
        figure.clear()
