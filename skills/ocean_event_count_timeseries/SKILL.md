---
skill_id: ocean_event_count_timeseries
description: Detects supported events and converts them into an event-count time series.
input_intent: Event type, variable, region, depth, time window, and optional grouping interval.
output_intent: Time series of event counts or event activity.
avoid_when:
- Use event_frequency_map for spatial hotspot/frequency maps.
- Use event detection skills for event-days or burden maps.
composes_with:
- ocean_masking_workflow
---
# Ocean Event Count Timeseries

## Purpose

This skill is the preferred choice for time-range event activity summaries when the user explicitly wants to know how event activity changes over time. It detects supported events over a period and converts the event list into a time ser...

## Workflow

### Stage 1: Shared scope and event driver field

```python
# IMPORTANT: Planner fills these values from the query before emitting workflow_code.
event_variable = 'temp'  # <-- MODIFY: temp for heatwaves, chlorophyll for blooms, oxygen for hypoxia.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
season_filter = None  # <-- MODIFY: seasonal subset such as spring/summer when requested.
vertical_mode = 'surface'  # <-- MODIFY: surface, bottom, fixed_depth, depth_range, or unspecified.
depth_value = None  # <-- MODIFY: fixed depth in meters when requested.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
percentile_threshold = 90.0  # <-- MODIFY: event percentile threshold when requested.
threshold = None  # <-- MODIFY: absolute threshold for bloom/hypoxia when requested.
min_duration_days = 3  # <-- MODIFY: minimum event duration.
min_area_km2 = 0.0  # <-- MODIFY: minimum event area.
weight_by = 'area'  # <-- MODIFY: area, days, or events depending on query wording.

event_field = load_dataset(
    variable=event_variable,
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
```

### Stage 2: Detect events

```python
event_detection = detect_heatwaves(
    temp=event_field.data,
    percentile_threshold=percentile_threshold,
    min_duration_days=min_duration_days,
    min_area_km2=min_area_km2,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
```

### Stage 3: Event-count time series

```python
event_count_timeseries = compute_event_timeseries_count(
    events=event_detection.events,
    weight_by=weight_by,
)
```

For bloom-count requests, use `detect_algal_blooms(chlorophyll=event_field.data, ...)` instead of `detect_heatwaves`. For hypoxia-count requests, use `detect_hypoxia(oxygen=event_field.data, ...)`.


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
