---
skill_id: ocean_event_comparison
description: Compares supported ocean events between two distinct time periods.
input_intent: Event type, variable, region, depth, and two explicit comparison periods.
output_intent: Period comparison summary of event count and mean intensity.
avoid_when:
- Use event_statistics for one-period summaries and event_frequency_map for hotspot maps.
composes_with:
- ocean_masking_workflow
---
# Ocean Event Period Comparison

## Purpose

This skill compares ocean events between two distinct time periods using the currently implemented comparison summary: total event count and mean event intensity. Use when the user asks to contrast supported events across periods at a su...

## Workflow

### Stage 1: Load variable for period 1

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
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

raw_p1 = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=period1_time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
```

### Stage 2: Load variable for period 2

```python
raw_p2 = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=period2_time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
```

### Stage 3: Detect events — period 1

```python
events_p1 = detect_heatwaves(
    temp=raw_p1.data,
    percentile_threshold=percentile_threshold,
    min_duration_days=min_duration_days,
    min_area_km2=min_area_km2,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 4: Detect events — period 2

```python
events_p2 = detect_heatwaves(
    temp=raw_p2.data,
    percentile_threshold=percentile_threshold,
    min_duration_days=min_duration_days,
    min_area_km2=min_area_km2,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 5: Compare event periods

```python
comparison = compare_event_periods(
    events1=events_p1.events,
    events2=events_p2.events,
    period1_label='Period 1',
    period2_label='Period 2',
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- For marine heatwave comparisons, detect events separately for each period before calling `compare_event_periods`; do not compare symbolic event placeholders.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
