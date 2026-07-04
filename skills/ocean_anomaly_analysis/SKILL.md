---
skill_id: ocean_anomaly_analysis
description: Computes anomalies of a raw ocean variable relative to a climatological baseline.
input_intent: Raw variable such as temperature, salinity, oxygen, or chlorophyll with region, time range, depth or vertical mode, and optional climatology baseline.
output_intent: Regional anomaly time series or spatial anomaly map.
avoid_when:
- Use trend, climatology, event, or derived-diagnostic skills when the requested product is not an anomaly.
composes_with:
- ocean_masking_workflow
---
# Ocean Anomaly Analysis

## Purpose

This skill computes anomalies of an ocean variable relative to its climatological mean. It supports two modes: - Time-series anomaly: compute anomaly time series of a regional mean (timeseriesresult). - Spatial anomaly: compute a spatial...

## Workflow

### Stage 1: Load the source field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
climatology_time_range = None  # <-- MODIFY: baseline/reference time window when the query provides one.
season_filter = None  # <-- MODIFY: spring/summer/winter/autumn subset from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.
period = 'monthly'  # <-- MODIFY: supported climatology period: 'monthly' or 'seasonal'.

raw_data = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
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

### Stage 3: Optional mask application

```text
masked_data = apply_mask(
    data=raw_data.data,
    mask=combined_mask.data,
)
```

### Stage 4: Branch A: Time-series anomaly (`anomaly_mode == "timeseries"` or default)

```python
timeseries = extract_regional_mean(
    data=raw_data.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 5: Compute climatology

```python
climatology = compute_climatology(
    timeseries=timeseries,
    period=period,
)
```

### Stage 6: Compute anomaly time series

```python
timeseries_result = compute_anomaly(
    timeseries=timeseries,
    climatology=climatology,
)
```

### Stage 7: Branch B: Spatial anomaly (`anomaly_mode == "spatial"`)

```python
climatology_data = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=climatology_time_range,
    season_filter=season_filter,
    depth_range=depth_range,
)
```

### Stage 8: Optional mask application to the climatology reference

```text
masked_climatology_data = apply_mask(
    data=climatology_data.data,
    mask=combined_mask.data,
)
```

### Stage 9: Compute spatial field climatology

```python
field_climatology = compute_field_climatology(
    data=climatology_data.data,
    period=period,
)
```

### Stage 10: Compute spatial field anomaly

```python
anomaly_field = compute_field_anomaly(
    data=raw_data.data,
    climatology=field_climatology.data,
    period=period,
)
```

### Stage 11: Compute spatial field from anomaly

```python
spatial_field_result = compute_spatial_field(
    data=anomaly_field.data,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- For unmasked anomaly requests, use `raw_data.data` and `climatology_data.data` directly. For masked requests, use the corresponding masked artifacts in the selected branch.
- Loader contract: never pass `depth_aggregation` to `load_dataset`; loader depth controls are `vertical_mode`, `depth_value`, and `depth_range`.
- `compute_climatology`, `compute_field_climatology`, and `compute_field_anomaly` support `period='monthly'` or `period='seasonal'` only. Put spring/summer/winter/autumn intent in `season_filter`, not in `period`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
