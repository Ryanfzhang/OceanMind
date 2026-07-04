---
skill_id: ocean_mesoscale_organization_analysis
description: Diagnoses whether tracer patterns are organized by fronts, eddies, or background flow.
input_intent: Tracer or event hotspot pattern plus velocity or mesoscale context fields over a region and time range.
output_intent: Mesoscale organization evidence describing fronts, eddies, and flow-structure context.
avoid_when:
- Use front_detection, eddy_detection, or spatial diagnostics when only one structure diagnostic is requested.
composes_with:
- ocean_masking_workflow
---
# Ocean Mesoscale Organization Analysis

## Purpose

This skill diagnoses whether a tracer pattern is being organized by fronts, eddies, or the background flow. It focuses on mesoscale structure proxies rather than full dynamical attribution, and is best used for hotspot and boundary-shape...

## Workflow

### Stage 1: Load tracer field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: tracer or hotspot variable requested by the user.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.
time_aggregation = 'mean'  # <-- MODIFY: map aggregation over the requested time window.
front_percentile = 90.0  # <-- MODIFY: percentile used to normalize front-proximity index.

tracer_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Compute front-proximity proxy

```python
front_proximity = compute_front_proximity_index(
    data=tracer_field.data,
    percentile=front_percentile,
)
```

### Stage 3: Map the organization proxy

```python
front_proximity_map = compute_spatial_field(
    data=front_proximity.data,
    time_range=time_range,
    time_aggregation='mean',
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- For a front-proximity organization map, first compute `compute_front_proximity_index` from the loaded tracer field, then reduce that proxy with `compute_spatial_field`; do not call `load_dataset` on an already-derived proxy.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
