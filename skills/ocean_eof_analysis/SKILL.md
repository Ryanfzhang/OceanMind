---
skill_id: ocean_eof_analysis
description: Performs EOF analysis on a selected raw variable or feature-derived raw-variable field.
input_intent: Raw variable with region, time range, depth or layer selection, and optional feature-depth controls.
output_intent: EOF modes, principal components, and variance fractions for a raw field.
avoid_when:
- Use ocean_derived_eof_analysis for EOF of derived diagnostics such as vorticity or speed.
composes_with:
- ocean_masking_workflow
---
# Ocean EOF Analysis

## Purpose

This skill performs EOF analysis on a selected variable, a feature-depth field, or a layer-mean field.

## Workflow

### Stage 1: Load the source field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].

eof_input = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Legacy EOF path

```python
eof_result = perform_eof_analysis(
    data=eof_input.data,
    n_modes=n_modes,
    preprocessing=preprocessing,
    weight_by_latitude=weight_by_latitude,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
