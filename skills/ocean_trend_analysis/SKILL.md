---
skill_id: ocean_trend_analysis
description: Computes linear trend of a raw ocean variable over a specified time range.
input_intent: Raw variable with region, time range, depth or vertical mode, and trend intent.
output_intent: Regional time-series trend or pixel-wise spatial trend map.
avoid_when:
- Use derived_timeseries or derived_spatial_field for trends of derived diagnostics.
- Use climatology or anomaly skills when trend is not requested.
composes_with:
- ocean_masking_workflow
---
# Ocean Trend Analysis

## Purpose

This skill computes the linear trend of an ocean variable over a specified time range. It supports two modes: - Time-series trend: compute the trend of a regional-mean time series (trendresult). - Spatial trend: compute a pixel-wise tren...

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

raw_data = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
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

### Stage 4: Branch A: Time-series trend (`trend_mode == "timeseries"` or default)

```python
timeseries = extract_regional_mean(
    data=raw_data.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 5: Compute trend on time series

```python
trend_result = compute_trend(
    timeseries=timeseries,
    confidence_level=confidence_level,
)
```

### Stage 6: Branch B: Spatial trend (`trend_mode == "spatial"`)

```text
field_trend_result = compute_field_trend(
    data=raw_data.data,
    confidence_level=confidence_level,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- For unmasked trend requests, use `raw_data.data` directly. For masked trend requests, use `masked_data.data` in the chosen trend branch.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
