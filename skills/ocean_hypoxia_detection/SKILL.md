---
skill_id: ocean_hypoxia_detection
description: Detects hypoxia events from oxygen and produces hypoxia summary maps.
input_intent: Oxygen field over a multi-day window with oxygen threshold, duration, bottom or depth selection, region, and masks.
output_intent: Hypoxic-days map, oxygen-deficit burden map, and event detection artifacts.
avoid_when:
- Use event_frequency_map for hotspot/frequency maps.
- Use oxygen trend or timeseries skills when no hypoxia event detection is requested.
composes_with:
- ocean_masking_workflow
- ocean_event_frequency_map
- ocean_event_count_timeseries
---
# Ocean Hypoxia Detection

## Purpose

This skill detects hypoxia events from dissolved oxygen fields and now returns summary maps by default for long-window detection workflows. The primary output is a hypoxia oxygen-deficit burden map, followed by a hypoxic-days map, with r...

## Workflow

### Stage 1: Load oxygen

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['oxygen']  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
depth_value = None  # <-- MODIFY: fixed depth in meters when the query names one depth.
vertical_mode = 'bottom'  # <-- MODIFY: surface, bottom, fixed_depth, depth_range, or unspecified; default hypoxia detection uses bottom water.
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.
oxygen_threshold = 60  # <-- MODIFY: hypoxia oxygen threshold in mmol/m3; default is oxygen < 60.
severe_threshold = 20  # <-- MODIFY: severe hypoxia threshold in mmol/m3; default is oxygen < 20.
min_duration_days = 3  # <-- MODIFY: minimum event duration in days; default is 3.
min_area_km2 = 100  # <-- MODIFY: minimum connected event area; default is 100 km2.

oxygen_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
```

### Stage 2: Detect hypoxia

```python
hypoxia_detection = detect_hypoxia(
    oxygen=oxygen_field.data,
    oxygen_threshold=oxygen_threshold,
    severe_threshold=severe_threshold,
    min_area_km2=min_area_km2,
    min_duration_days=min_duration_days,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 3: Summarize hypoxic days

```python
hypoxic_days = compute_event_summary_map(
    event_detection=hypoxia_detection,
    data=oxygen_field.data,
    summary_mode='event_days',
)
```

### Stage 4: Summarize hypoxia burden

```python
hypoxia_oxygen_deficit_burden = compute_event_summary_map(
    event_detection=hypoxia_detection,
    data=oxygen_field.data,
    summary_mode='burden',
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
