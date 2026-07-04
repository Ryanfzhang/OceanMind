"""Ocean analysis tool exports."""

from .field_tools import (
    compute_field_anomaly,
    compute_field_climatology,
    compute_field_trend,
    compute_local_tendency,
)
from .advanced import perform_eof_analysis, reconstruct_from_eof
from .histogram import compute_2d_histogram, compute_histogram
from .profile import (
    analyze_vertical_structure,
    extract_vertical_profile,
    identify_mixed_layer_depth,
    identify_pycnocline_depth,
    identify_thermocline_depth,
)
from .sections import compute_section_hovmoller, extract_transect_section
from .spatial import compute_hovmoller, compute_spatial_field, extract_timeseries
from .statistics import compute_composite_field, compute_regression_map, compute_spectrum
from .timeseries import (
    compute_anomaly,
    compute_climatology,
    compute_layer_mean,
    compute_mixed_layer_mean,
    compute_trend,
    extract_point_timeseries,
    extract_regional_mean,
    remove_seasonal_cycle,
    resample_timeseries,
)
from .transports import (
    compute_freshwater_transport,
    compute_heat_transport,
    compute_salt_transport,
    compute_transect_normal_flux_hovmoller,
    compute_transport_by_layer,
    compute_transport_streamfunction_map,
    compute_volume_transport,
)
from .watermass import compute_isopycnal_layer_mean, compute_ts_diagram, extract_isopycnal_surface
from .weighted_means import (
    compute_area_integral,
    compute_area_weighted_mean,
    compute_volume_integral,
    compute_volume_weighted_mean,
)

__all__ = [
    "extract_regional_mean",
    "extract_point_timeseries",
    "compute_mixed_layer_mean",
    "compute_layer_mean",
    "compute_climatology",
    "compute_anomaly",
    "compute_trend",
    "compute_field_climatology",
    "compute_field_anomaly",
    "compute_field_trend",
    "compute_local_tendency",
    "compute_area_weighted_mean",
    "compute_volume_weighted_mean",
    "compute_area_integral",
    "compute_volume_integral",
    "resample_timeseries",
    "remove_seasonal_cycle",
    "extract_vertical_profile",
    "identify_mixed_layer_depth",
    "identify_thermocline_depth",
    "identify_pycnocline_depth",
    "analyze_vertical_structure",
    "compute_spatial_field",
    "extract_timeseries",
    "compute_hovmoller",
    "compute_histogram",
    "compute_2d_histogram",
    "perform_eof_analysis",
    "reconstruct_from_eof",
    "extract_transect_section",
    "compute_section_hovmoller",
    "compute_volume_transport",
    "compute_heat_transport",
    "compute_salt_transport",
    "compute_freshwater_transport",
    "compute_transport_by_layer",
    "compute_transport_streamfunction_map",
    "compute_transect_normal_flux_hovmoller",
    "compute_ts_diagram",
    "extract_isopycnal_surface",
    "compute_isopycnal_layer_mean",
    "compute_regression_map",
    "compute_composite_field",
    "compute_spectrum",
]
