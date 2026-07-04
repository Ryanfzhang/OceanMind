---
skill_id: ocean_meander_detection
description: Detects high-curvature meander-like current structures from u and v.
input_intent: Velocity fields u and v over a region/time/depth selection with meander or high-curvature current intent.
output_intent: Meander detection mask or meander structure map.
avoid_when:
- Use jet_detection for jet cores and eddy_detection for closed eddy structures.
composes_with:
- ocean_masking_workflow
---
# Ocean Meander Detection

## Purpose

This skill detects high-curvature meander-like current structures from u and v.

## Workflow

### Stage 1: Load velocity fields

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
curvature_threshold = None  # <-- MODIFY: absolute curvature threshold from the query; leave None to use percentile_threshold.
percentile_threshold = 90  # <-- MODIFY: relative curvature percentile computed over the analysis field/time; default meander detection uses p90.
min_length_km = 80.0  # <-- MODIFY: minimum meander length; default is 80 km.
min_pixels = 10  # <-- MODIFY: minimum connected pixels; default is 10.

u_field = load_dataset(
    variable='u',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)

v_field = load_dataset(
    variable='v',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Detection step

```python
meander_detection = detect_meanders(
    u=u_field.data,
    v=v_field.data,
    curvature_threshold=curvature_threshold,
    percentile_threshold=percentile_threshold,
    min_length_km=min_length_km,
    min_pixels=min_pixels,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Load both `u` and `v` before calling `detect_meanders`; the detector consumes velocity fields, not preexisting symbolic artifacts.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
