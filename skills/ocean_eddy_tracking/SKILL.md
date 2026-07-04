---
skill_id: ocean_eddy_tracking
description: Tracks detected eddies through time using velocity fields and nearest-neighbor linking.
input_intent: Time sequence of u and v velocity fields over a region with eddy detection parameters.
output_intent: Eddy tracks and trajectory summaries.
avoid_when:
- Use ocean_eddy_detection for a single-date eddy map or event list.
composes_with:
- ocean_masking_workflow
---
# Ocean Eddy Tracking

## Purpose

This skill tracks eddies through time using a multi-step velocity field and nearest-neighbor trajectory linking.

## Workflow

### Stage 1: Load velocity fields

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: multi-step tracking window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].

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

### Stage 2: Tracking step

```python
eddy_tracking = track_eddies(
    u=u_field.data,
    v=v_field.data,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- `track_eddies` currently consumes only the loaded `u` and `v` fields; do not pass detection thresholds or linking parameters unless the tool contract is extended.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
