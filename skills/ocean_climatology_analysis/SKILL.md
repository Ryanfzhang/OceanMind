---
skill_id: ocean_climatology_analysis
description: Computes climatological mean state or seasonal cycle for a raw ocean variable.
input_intent: Raw variable with multi-year time range, region, depth or vertical mode, and optional monthly or seasonal grouping.
output_intent: Climatology time series, seasonal cycle, or climatological spatial map.
avoid_when:
- Use anomaly when comparing observations to climatology.
- Use trend when the user asks for change over time.
composes_with:
- ocean_masking_workflow
---
# Ocean Climatology Analysis

## Purpose

This skill computes the climatological mean state (seasonal cycle or annual mean) of an ocean variable. It supports two modes: - Time-series climatology: compute the seasonal/monthly climatology of a regional-mean time series (climatolog...

## Workflow

### Stage 1: Load the source field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
season_filter = None  # <-- MODIFY: spring/summer/winter/autumn subset from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
reduction_depth_aggregation = 'mean'  # <-- MODIFY: how downstream reducers reduce a depth range.
period = 'monthly'  # <-- MODIFY: supported grouping period: 'monthly' or 'seasonal'. Weekly is unsupported.

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

### Stage 4: Branch A: Time-series climatology (`climatology_mode == "timeseries"` or default)

```python
timeseries = extract_regional_mean(
    data=raw_data.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=reduction_depth_aggregation,
)
```

### Stage 5: Compute climatology

```python
climatology_result = compute_climatology(
    timeseries=timeseries,
    period=period,
)
```

### Stage 6: Branch B: Spatial climatology (`climatology_mode == "spatial"`)

```python
clim_field = compute_field_climatology(
    data=raw_data.data,
    period=period,
)
```

### Stage 7: Compute spatial field from climatology

```python
spatial_field_result = compute_spatial_field(
    data=clim_field.data,
    depth_range=depth_range,
    depth_aggregation=reduction_depth_aggregation,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- For unmasked climatology requests, use `raw_data.data` directly. For masked requests, use `masked_data.data` in the selected climatology branch.
- Loader contract: never pass `depth_aggregation` to `load_dataset`; loader depth controls are `vertical_mode`, `depth_value`, and `depth_range`.
- `compute_climatology` and `compute_field_climatology` support `period='monthly'` or `period='seasonal'` only. Weekly climatology is not supported by the current tools; do not emit a weekly period literal. If the user asks for weekly climatology, run the nearest supported monthly climatology and make the limitation explicit in the final answer.
- Put spring/summer/winter/autumn intent in `season_filter`; use `period='seasonal'` only for seasonal grouping.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
