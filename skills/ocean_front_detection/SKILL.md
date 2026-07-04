---
skill_id: ocean_front_detection
description: Detects fronts from temperature or salinity gradients on a target day and region.
input_intent: Temperature or salinity field for a date or short time window, region, and optional depth selection.
output_intent: Front mask or front detection map.
avoid_when:
- Use spatial diagnostics for velocity fronts or current structure diagnostics without tracer-gradient front detection.
composes_with:
- ocean_masking_workflow
---
# Ocean Front Detection

## Purpose

This skill detects fronts from temperature or salinity gradients on a target day and region.

## Workflow

### Stage 1: Load the source field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
gradient_threshold = None  # <-- MODIFY: optional absolute gradient threshold; None uses percentile behavior.
min_length_km = 10.0  # <-- MODIFY: minimum detected front length.
min_pixels = 3  # <-- MODIFY: minimum connected pixels.
smoothing_sigma = 1.0  # <-- MODIFY: smoothing width in grid cells.

front_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
)
```

### Stage 2: Detect fronts

```python
front_detection = detect_fronts(
    data=front_field.data,
    variable=variables[0],
    gradient_threshold=gradient_threshold,
    min_length_km=min_length_km,
    min_pixels=min_pixels,
    smoothing_sigma=smoothing_sigma,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
