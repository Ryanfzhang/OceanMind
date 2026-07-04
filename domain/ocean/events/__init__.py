"""Event tool exports."""

from .analysis import (
    compare_event_periods,
    compute_event_frequency_map,
    compute_event_summary_map,
    compute_event_spatial_distribution,
    compute_event_statistics,
    compute_event_timeseries_count,
)
from .bloom import detect_algal_blooms
from .eddy import detect_eddies, track_eddies
from .eutrophication import detect_eutrophication
from .front import detect_fronts
from .heatwave import detect_heatwaves
from .hypoxia import detect_hypoxia
from .jet import detect_jets
from .meander import detect_meanders
from .upwelling import detect_upwelling

__all__ = [
    "detect_heatwaves",
    "detect_hypoxia",
    "detect_algal_blooms",
    "detect_eddies",
    "track_eddies",
    "detect_fronts",
    "detect_upwelling",
    "detect_jets",
    "detect_meanders",
    "detect_eutrophication",
    "compute_event_statistics",
    "compute_event_spatial_distribution",
    "compare_event_periods",
    "compute_event_frequency_map",
    "compute_event_summary_map",
    "compute_event_timeseries_count",
]
