---
skill_id: ocean_vertical_integral
description: Computes a vertically integrated proxy of a raw ocean variable over a depth range.
input_intent: Raw variable with explicit depth range, region, time range, and vertical integration intent.
output_intent: Vertically integrated field, map, or time series proxy.
avoid_when:
- Use layer_mean_analysis for depth-averaged means and transport_analysis for physical transport integrals.
composes_with:
- ocean_masking_workflow
---
# Ocean Vertical Integral Analysis

## Purpose

This skill computes a vertically integrated proxy of an ocean variable over a specified depth range. For temperature, the result should be interpreted as a vertically integrated temperature or heat-content proxy rather than a full physic...

## Workflow

### Stage 1: Load source field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].

raw_data = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Compute vertical integral

```python
integral_field = compute_vertical_integral(
    data=raw_data.data,
    depth_range=depth_range,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Loader contract: never pass `depth_aggregation` to `load_dataset`; loader depth controls are `vertical_mode`, `depth_value`, and `depth_range`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
