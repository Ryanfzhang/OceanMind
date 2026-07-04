---
skill_id: ocean_eddy_detection
description: Detects ocean eddies on a target day from surface u and v current fields.
input_intent: Surface velocity fields u and v for a specific date or short window and region.
output_intent: Eddy event detections, eddy masks, centers, and eddy summary properties.
avoid_when:
- Use ocean_eddy_tracking for trajectories through time.
- Use dynamics diagnostics for generic vorticity maps without eddy detection.
composes_with:
- ocean_masking_workflow
---
# Ocean Eddy Detection

## Purpose

This skill detects ocean eddies in a target region on a specific day using the existing detecteddies() tool and surface current fields: - eastward velocity u - northward velocity v

## Workflow

### Stage 1: Load eastward velocity `u`

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['u', 'v']  # Fixed current variables in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
ow_threshold = -2e-12  # <-- MODIFY: Okubo-Weiss threshold.
min_radius_km = 30.0  # <-- MODIFY: optional minimum eddy radius.
max_radius_km = 300.0  # <-- MODIFY: optional maximum eddy radius.
min_pixels = 10  # <-- MODIFY: minimum connected pixels.

u_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
)
```

### Stage 2: Load northward velocity `v`

```python
v_field = load_dataset(
    variable=variables[1],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
)
```

### Stage 3: Detect eddies

```python
eddy_detection = detect_eddies(
    u=u_field.data,
    v=v_field.data,
    ow_threshold=ow_threshold,
    min_radius_km=min_radius_km,
    max_radius_km=max_radius_km,
    min_pixels=min_pixels,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
