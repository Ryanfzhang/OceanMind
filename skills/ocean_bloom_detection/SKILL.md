---
skill_id: ocean_bloom_detection
description: Detects sustained algal bloom events from chlorophyll and produces bloom summary maps.
input_intent: Chlorophyll field over a multi-day window with optional absolute bloom threshold, duration, region, depth, and masks.
output_intent: Bloom burden map, bloom event-days map, and event detection artifacts.
avoid_when:
- Use ocean_event_frequency_map for hotspot/frequency maps.
- Use raw chlorophyll trend or timeseries skills when no bloom event detection is requested.
composes_with:
- ocean_masking_workflow
- ocean_event_frequency_map
- ocean_event_count_timeseries
---
# Ocean Algal Bloom Detection

## Purpose

This skill detects sustained chlorophyll bloom events over a multi-day analysis window and now returns summary maps by default for long-window detection workflows. The primary output is a bloom chlorophyll burden map, followed by a bloom...

## Workflow

### Stage 1: Load chlorophyll

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['chlorophyll']  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
depth_value = None  # <-- MODIFY: fixed depth in meters when the query names one depth.
vertical_mode = 'surface'  # <-- MODIFY: surface, bottom, fixed_depth, depth_range, or unspecified.
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.
threshold = 1.0  # <-- MODIFY: absolute chlorophyll-value threshold; default bloom detection uses chlorophyll > 1.0 unless the query asks for percentile/anomaly detection.
percentile_threshold = None  # <-- MODIFY: relative anomaly percentile such as p85 or p90 computed per grid cell over time; set only when the query asks for percentile/anomalously high bloom detection.
min_duration_days = 5  # <-- MODIFY: minimum event duration in days; default is 5.
min_area_km2 = 500  # <-- MODIFY: minimum connected event area; default is 500 km2.
bloom_type = 'auto'  # <-- MODIFY: bloom subtype, one of auto, spring, harmful; default is auto.

chlorophyll_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
```

### Stage 2: Detect blooms

```python
bloom_detection = detect_algal_blooms(
    chlorophyll=chlorophyll_field.data,
    threshold=threshold,
    percentile_threshold=percentile_threshold,
    min_duration_days=min_duration_days,
    min_area_km2=min_area_km2,
    bloom_type=bloom_type,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 3: Summarize bloom event days

```python
bloom_event_days = compute_event_summary_map(
    event_detection=bloom_detection,
    data=chlorophyll_field.data,
    summary_mode='event_days',
)
```

### Stage 4: Summarize bloom burden

```python
bloom_chlorophyll_burden = compute_event_summary_map(
    event_detection=bloom_detection,
    data=chlorophyll_field.data,
    summary_mode='burden',
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
