---
skill_id: ocean_event_statistics
description: Detects supported events and computes statistical summaries.
input_intent: Event type, variable, region, depth, time range, thresholds, and grouping or summary statistics request.
output_intent: Event statistics table or summary metrics.
avoid_when:
- Use event_frequency_map for gridded hotspot maps and event_count_timeseries for count time series.
composes_with:
- ocean_masking_workflow
---
# Ocean Event Statistics

## Purpose

This skill detects supported time-range events and then computes statistical summaries.

## Workflow

### Stage 1: Heatwave Branch

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['temp']  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
depth_value = None  # <-- MODIFY: fixed depth in meters when the query names one depth.
vertical_mode = 'surface'  # <-- MODIFY: surface, bottom, fixed_depth, depth_range, or unspecified.
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.
percentile_threshold = 90  # <-- MODIFY: default marine heatwave percentile.
min_duration_days = 5  # <-- MODIFY: minimum event duration in days.
min_area_km2 = 1000  # <-- MODIFY: minimum connected event area.

temperature_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
```

### Stage 2: Detect heatwave events

```python
event_detection = detect_heatwaves(
    temp=temperature_field.data,
    percentile_threshold=percentile_threshold,
    min_duration_days=min_duration_days,
    min_area_km2=min_area_km2,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 3: Compute event statistics

```python
event_statistics = compute_event_statistics(
    events=event_detection.events,
    group_by='month',
)
```

### Stage 4: Optional spatial distribution

```python
event_spatial_distribution = compute_event_spatial_distribution(
    events=event_detection.events,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Event statistics require a real `event_detection` artifact from a detector such as `detect_heatwaves`; do not call statistics tools on symbolic event placeholders.
- Loader contract: never pass `depth_aggregation` to `load_dataset`; loader depth controls are `vertical_mode`, `depth_value`, and `depth_range`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
