---
skill_id: ocean_spectral_analysis
description: Computes a one-dimensional power spectrum from an ocean time series.
input_intent: Raw variable or index time series with region, time range, depth selection, and dominant-period or variance-by-timescale question.
output_intent: Power spectrum and dominant-period summary.
avoid_when:
- Use timeseries_analysis when no frequency or spectrum output is requested.
composes_with:
- ocean_masking_workflow
---
# Ocean Spectral Analysis

## Purpose

This skill computes a one-dimensional power spectrum from an internal time series built from the same West Pacific dataset. Use it for dominant-period or variance-by-timescale questions.

## Workflow

### Stage 1: Load the source field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.

raw_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Extract a regional-mean time series

```python
analysis_timeseries = extract_regional_mean(
    data=raw_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 3: Extract a point time series

```text
analysis_timeseries = extract_point_timeseries(
    data=raw_field.data,
    lon=location_lon,
    lat=location_lat,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 4: Compute the spectrum

```python
spectrum_result = compute_spectrum(
    timeseries=analysis_timeseries,
    method=method,
    detrend=detrend,
    window=window,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
