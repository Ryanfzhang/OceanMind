---
skill_id: ocean_jet_detection
description: Detects jet-like current cores from u and v velocity fields.
input_intent: Velocity fields u and v over a region/time/depth selection with jet or current-core intent.
output_intent: Jet detection mask, jet core map, or jet summary metrics.
avoid_when:
- Use spatial_diagnostics for generic current speed maps and meander_detection for curvature structures.
composes_with:
- ocean_masking_workflow
---
# Ocean Jet Detection

## Purpose

This skill detects jet-like, elongated high-speed current features from u and v.

## Workflow

### Stage 1: Load velocity fields

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
speed_threshold = None  # <-- MODIFY: absolute current-speed threshold from the query; leave None to use percentile_threshold.
percentile_threshold = 90  # <-- MODIFY: relative speed percentile computed over the analysis field/time; default jet detection uses p90.
min_length_km = 100.0  # <-- MODIFY: minimum jet length; default is 100 km.
min_aspect_ratio = 3.0  # <-- MODIFY: elongated-shape filter; default requires length/width >= 3.
min_pixels = 12  # <-- MODIFY: minimum connected pixels; default is 12.

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
jet_detection = detect_jets(
    u=u_field.data,
    v=v_field.data,
    speed_threshold=speed_threshold,
    percentile_threshold=percentile_threshold,
    min_length_km=min_length_km,
    min_aspect_ratio=min_aspect_ratio,
    min_pixels=min_pixels,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Load both `u` and `v` before calling `detect_jets`; the detector consumes velocity fields, not preexisting symbolic artifacts.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
