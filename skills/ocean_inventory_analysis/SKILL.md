---
skill_id: ocean_inventory_analysis
description: Computes regional inventory or total amount of a tracer-like field.
input_intent: Tracer variable with region, time range, depth range, and optional area or volume integration intent.
output_intent: Inventory, integrated amount, or regional total time series/map.
avoid_when:
- Use vertical_integral for simple vertical integrals and budget_analysis for tendency budgets.
composes_with:
- ocean_masking_workflow
---
# Ocean Inventory Analysis

## Purpose

This skill computes regional inventories from a single gridded model dataset. Use it when the user wants a total amount, content, or reservoir over an area or volume rather than a regional mean.

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

### Stage 2: Optional polygon mask

```text
combined_mask = combine_masks(
    masks=[region_mask.data, isobath_mask.data],
    operation=mask_combination_operation,
)
```

### Stage 3: Optional mask application

```text
masked_field = apply_mask(
    data=raw_field.data,
    mask=combined_mask.data,
)
```

### Stage 4: Area integral

```python
inventory_timeseries = compute_area_integral(
    data=raw_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 5: Volume integral

```text
volume_inventory_timeseries = compute_volume_integral(
    data=raw_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- For unmasked inventory requests, pass `raw_field.data` directly. For masked requests, pass `masked_field.data` to the selected integral tool.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
