---
skill_id: ocean_spatial_field_analysis
description: Converts a raw ocean variable into a map-ready two-dimensional spatial field.
input_intent: Raw variable with region, time range, depth selection, optional polygon mask, temporal aggregation, or feature-depth controls.
output_intent: 2D map-ready field for a raw variable.
avoid_when:
- Use ocean_dynamics_diagnostics or derived_spatial_field_analysis for derived velocity diagnostics.
- Use timeseries, profile, histogram, or Hovmoller skills for non-map outputs.
composes_with:
- ocean_masking_workflow
---
# Ocean Spatial Field Analysis

## Purpose

This skill converts an ocean variable into a map-ready two-dimensional field. It supports direct spatial aggregation, feature-depth maps, layer-mean maps, polygon masking, and field-level temporal diagnostics before the final map is made.

## Workflow

### Stage 1: Load the source field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
vertical_mode = 'unspecified'  # <-- MODIFY: surface, bottom, fixed_depth, depth_range, or unspecified.
depth_value = None  # <-- MODIFY: fixed depth in meters only when the query names one depth.
depth_range = None  # <-- MODIFY: numeric depth interval only when the query names a layer; surface may use [0, 0].
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.

raw_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
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

### Stage 3: Legacy analysis base

```text
masked_analysis_base = apply_mask(
    data=raw_field.data,
    mask=combined_mask.data,
)
```

### Stage 4: Optional field temporal analysis

```python
spatial_field = compute_spatial_field(
    data=raw_field.data,
    time_range=time_range,
    time_aggregation=time_aggregation,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Workflow code must use assignment-form Python DSL, such as `raw_field = load_dataset(...)`; never use bare tool calls, `save_as=...`, or `$ref:...` JSON references.
- Bottom/near-bottom/seafloor/benthic requests must use `vertical_mode = 'bottom'` with `depth_value = None` and `depth_range = None`; do not approximate bottom as 8000 m, -8000 m, or any fixed deepest coordinate.
- Use `depth_range` only for an explicit numeric layer named by the user, such as upper 0-750 m or 100-200 m.
- `load_dataset` performs the horizontal/vertical subset. Do not pass `lon_range`, `lat_range`, `vertical_mode`, or `depth_value` to `compute_spatial_field`; it only consumes `data`, optional `time_range`, `time_aggregation`, optional `depth_range`, `depth_aggregation`, and optional `mask`.
- `compute_spatial_field` must use a supported `depth_aggregation`; for bottom maps use `depth_aggregation = 'mean'`, not `'none'` or `None`.
- For unmasked map requests, pass `raw_field.data` directly to `compute_spatial_field`. For masked requests, build/apply the mask before the map step and pass `masked_analysis_base.data` instead.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
