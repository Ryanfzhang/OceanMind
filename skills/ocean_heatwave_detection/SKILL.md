---
skill_id: ocean_heatwave_detection
description: Detects marine heatwave events from temperature and produces heatwave burden or event-days maps.
input_intent: Temperature field over a multi-day window with heatwave percentile threshold, duration, region, depth, and masks.
output_intent: Marine heatwave burden map, heatwave event-days map, and event detection artifacts.
avoid_when:
- Use ocean_event_frequency_map for hotspot/frequency maps.
- Use trend or anomaly skills when no heatwave event detection is requested.
composes_with:
- ocean_masking_workflow
- ocean_event_frequency_map
- ocean_event_count_timeseries
---
# Ocean Heatwave Detection

## Purpose

This skill detects marine heatwave events over a multi-day analysis window using temperature data and now returns summary maps by default for long-window detection workflows. The primary output is a marine heatwave burden map, followed b...

## Workflow

### Stage 1: Load temperature

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
# If the user asks for an absolute temperature condition such as temp > 30, build a condition/threshold mask before this detection step instead of passing it to detect_heatwaves.
percentile_threshold = 90  # <-- MODIFY: relative anomaly percentile computed per grid cell over time; default heatwave detection uses p90.
min_duration_days = 5  # <-- MODIFY: minimum event duration in days; default is 5.
min_area_km2 = 1000  # <-- MODIFY: minimum connected event area; default is 1000 km2.

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

### Stage 2: Detect heatwaves

```python
heatwave_detection = detect_heatwaves(
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

### Stage 3: Summarize heatwave days

```python
marine_heatwave_days = compute_event_summary_map(
    event_detection=heatwave_detection,
    data=temperature_field.data,
    summary_mode='event_days',
)
```

### Stage 4: Summarize heatwave burden

```python
marine_heatwave_burden = compute_event_summary_map(
    event_detection=heatwave_detection,
    data=temperature_field.data,
    summary_mode='burden',
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
