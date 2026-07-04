---
skill_id: ocean_timeseries_resampling
description: Extracts a regional mean time series and resamples it to a coarser temporal resolution.
input_intent: Raw variable with region, time range, depth selection, and requested weekly, monthly, seasonal, or other temporal aggregation.
output_intent: Resampled or downsampled time series.
avoid_when:
- Use ocean_timeseries for native-resolution time series without resampling.
composes_with:
- ocean_masking_workflow
---
# Ocean Timeseries Resampling

## Purpose

This skill extracts a regional mean time series and resamples it to a coarser temporal resolution (e.g. daily → weekly, daily → monthly). Use when the user asks to "resample", "aggregate to monthly", "weekly averages", "downsample the ti...

## Workflow

### Stage 1: Load variable

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.

raw_data = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Extract regional mean time series

```python
raw_timeseries = extract_regional_mean(
    data=raw_data.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 3: Resample time series

```python
resampled_timeseries = resample_timeseries(
    timeseries=raw_timeseries,
    freq=freq,
    method='mean',
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
