---
skill_id: ocean_masking_workflow
description: Builds polygon, isobath, or combined masks and applies them before map or time-series analysis.
input_intent: Explicit non-rectangular region, drawn polygon, isobath threshold, bathymetry condition, or combined analysis mask.
output_intent: Masked spatial field or masked time series for a raw variable.
avoid_when:
- Use a domain skill with built-in mask support when the requested diagnostic or event workflow is otherwise clear.
composes_with: []
---
# Ocean Masking Workflow

## Purpose

This skill builds a custom spatial mask (polygon-based, isobath-based, or both) and applies it to an ocean variable before producing a spatial field or time series. Use this skill when the user explicitly defines a non-rectangular region...

## Workflow

### Stage 1: Load the variable

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.
threshold = None  # <-- MODIFY: threshold value from the query when requested.
mask_polygon = [[113.0, 18.0], [114.0, 18.0], [114.0, 19.0], [113.0, 19.0]]
mask_isobath_depth = 50.0
mask_isobath_comparison = 'deeper_or_equal'
mask_combination_operation = 'and'
mask_invert = False
time_aggregation = 'mean'

raw_data = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Build polygon mask (when `mask_polygon` is provided)

```python
polygon_mask = build_polygon_mask(
    data=raw_data.data,
    polygon_points=mask_polygon,
)
```

### Stage 3: Build isobath mask (when `mask_isobath_depth` is provided)

```python
isobath_mask = build_isobath_mask(
    data=raw_data.data,
    isobath_depth=mask_isobath_depth,
    comparison=mask_isobath_comparison,
)
```

### Stage 4: Optional condition mask (when `mask_condition_expression` is provided)

```text
condition_mask = build_condition_mask(
    fields={'field': raw_data.data},
    expression=mask_condition_expression,
    mask_name='condition_mask',
)
```

### Stage 5: Combine masks

```python
final_mask = combine_masks(
    masks=[polygon_mask.data, isobath_mask.data],
    operation=mask_combination_operation,
    invert=mask_invert,
)
```

### Stage 6: Apply mask

```python
masked_data = apply_mask(
    data=raw_data.data,
    mask=final_mask.data,
)
```

### Stage 7: Output — spatial map (`output_mode == "spatial"` or default)

```python
spatial_field_result = compute_spatial_field(
    data=masked_data.data,
    time_range=time_range,
    time_aggregation=time_aggregation,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 8: Output — time series (`output_mode == "timeseries"`)

```python
timeseries = extract_regional_mean(
    data=masked_data.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: polygon, isobath, threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
