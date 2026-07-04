---
skill_id: ocean_field_visualization
description: Loads a raw physical or biogeochemical variable and returns the field container unchanged.
input_intent: Raw variable with region, time range, and optional depth range when the user wants to inspect or pass through the data.
output_intent: Raw data container for visualization or downstream analysis.
avoid_when:
- Use spatial_field_analysis for map-ready aggregation and timeseries/profile skills for reduced outputs.
composes_with:
- ocean_masking_workflow
---
# Ocean Raw Field Loading

## Purpose

This skill loads a single physical or biogeochemical variable for a target region, time range, and optional depth range and returns the raw field container unchanged.

## Workflow

### Stage 1: Load the requested field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
vertical_mode = 'unspecified'  # <-- MODIFY: surface, bottom, fixed_depth, depth_range, or unspecified.
depth_value = None  # <-- MODIFY: fixed depth in meters only when the query names one depth.
depth_range = None  # <-- MODIFY: numeric depth interval only when the query names a layer; surface may use [0, 0].

field_data = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Workflow code must use assignment-form Python DSL, such as `field_data = load_dataset(...)`; never use bare tool calls, `save_as=...`, or `$ref:...` JSON references.
- Bottom/near-bottom/seafloor/benthic requests must use `vertical_mode = 'bottom'` with `depth_value = None` and `depth_range = None`; do not approximate bottom as 8000 m, -8000 m, or any fixed deepest coordinate.
- Use `depth_range` only for an explicit numeric layer named by the user, such as upper 0-750 m or 100-200 m.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
