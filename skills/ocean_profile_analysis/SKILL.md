---
skill_id: ocean_profile_analysis
description: Extracts a vertical profile of a raw ocean variable at a point or over a region.
input_intent: Raw variable with point or region selection, time selection, and vertical profile request.
output_intent: Vertical profile of a raw variable.
avoid_when:
- Use derived_profile_analysis for derived diagnostics and hovmoller_analysis for time-depth diagrams.
composes_with:
- ocean_masking_workflow
---
# Ocean Vertical Profile Analysis

## Purpose

This skill extracts a vertical profile at a target location. In v1 of the vertical-feature workflow, this skill supports overlay only.

## Workflow

### Stage 1: Load the source field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region; for point profiles, use a small non-zero window around profile_lon.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region; for point profiles, use a small non-zero window around profile_lat.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
profile_lon = 113.5  # <-- MODIFY: profile longitude.
profile_lat = 18.5  # <-- MODIFY: profile latitude.
profile_method = 'nearest'  # <-- MODIFY: nearest or linear.

raw_profile_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Optional field trend

```text
trend_field = compute_field_trend(
    data=raw_profile_field.data,
    confidence_level=trend_confidence_level,
)
```

### Stage 3: Extract the point profile

```python
vertical_profile = extract_vertical_profile(
    data=raw_profile_field.data,
    lon=profile_lon,
    lat=profile_lat,
    method=profile_method,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Use `raw_profile_field.data` for ordinary profile requests; use `trend_field.data` only when the user explicitly asks for a trend profile.
- Point profile loader contract: for a point profile, load a small non-zero lon/lat window around `profile_lon`/`profile_lat`; never use zero-width ranges such as `lon_range=[profile_lon, profile_lon]` or `lat_range=[profile_lat, profile_lat]`. The exact point is selected only in `extract_vertical_profile(lon=profile_lon, lat=profile_lat, ...)`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
