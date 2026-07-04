---
skill_id: ocean_timeseries
description: Analyzes temporal evolution of a raw ocean variable over a region or point.
input_intent: Raw variable with region or point selection, time range, depth or vertical-feature selection, and temporal evolution intent.
output_intent: Raw-variable time series, regional mean series, point series, or simple time-series summary.
avoid_when:
- Use derived_timeseries_analysis for derived diagnostics and event_count_timeseries for event activity counts.
composes_with:
- ocean_masking_workflow
---
# Ocean Time Series Analysis

## Purpose

This skill analyzes the temporal evolution of an ocean variable over a region or point. It supports the legacy fixed-depth workflow and the new vertical-feature workflow. Use this skill for raw tracer time series such as temperature, SST...

## Workflow

### Stage 1: Load the source field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
vertical_mode = None  # <-- MODIFY: surface, bottom, fixed_depth, depth_range, or unspecified.
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.

raw_data = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_range=depth_range,
)
```

### Stage 2: Optional polygon mask

```text
combined_mask = combine_masks(
    masks=[region_mask.data, isobath_mask.data],
    operation=mask_combination_operation,
)
```

### Stage 3: Legacy fixed-depth time series source

```text
masked_timeseries_source = apply_mask(
    data=raw_data.data,
    mask=combined_mask.data,
)
```

### Stage 4: Unweighted regional mean

```python
timeseries = extract_regional_mean(
    data=raw_data.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 5: Point time series

```text
timeseries = extract_point_timeseries(
    data=raw_data.data,
    lon=point_lon,
    lat=point_lat,
    method=method,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 6: Area-weighted regional mean

```text
timeseries = compute_area_weighted_mean(
    data=raw_data.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 7: Volume-weighted regional mean

```text
timeseries = compute_volume_weighted_mean(
    data=raw_data.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- For unmasked requests, reduce `raw_data.data` directly. For masked requests, build/apply the mask first and use `masked_timeseries_source.data` in the selected time-series reduction.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
