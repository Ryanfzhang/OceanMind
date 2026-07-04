---
skill_id: ocean_eutrophication_detection
description: Detects persistent eutrophication-like events from chlorophyll anomalies with optional oxygen support.
input_intent: Chlorophyll field and optional oxygen field over a multi-day window with eutrophication thresholds and region/depth selection.
output_intent: Eutrophication burden map, event-days map, and event detection artifacts.
avoid_when:
- Use bloom_detection for chlorophyll bloom events without anomaly/oxygen eutrophication framing.
- Use event_frequency_map for hotspot/frequency maps.
composes_with:
- ocean_masking_workflow
- ocean_event_frequency_map
- ocean_event_count_timeseries
---
# Ocean Eutrophication Detection

## Purpose

This skill detects persistent eutrophication-like events over a multi-day analysis window based on chlorophyll anomalies, with optional oxygen support, and now returns summary maps by default for long-window detection workflows. The prim...

## Workflow

### Stage 1: Load chlorophyll

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['chlorophyll', 'oxygen']  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
depth_value = None  # <-- MODIFY: fixed depth in meters when the query names one depth.
vertical_mode = 'surface'  # <-- MODIFY: surface, bottom, fixed_depth, depth_range, or unspecified; default eutrophication detection uses surface water.
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.
chlorophyll_percentile = 90  # <-- MODIFY: high-chlorophyll percentile threshold; default eutrophication detection uses p90.
oxygen_threshold = None  # <-- MODIFY: optional low-oxygen threshold; leave None unless the query asks to combine chlorophyll with oxygen stress.
min_duration_days = 5  # <-- MODIFY: minimum event duration in days; default is 5.
min_area_km2 = 1000  # <-- MODIFY: minimum connected event area; default is 1000 km2.

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

### Stage 2: Optional oxygen load

```python
oxygen_field = load_dataset(
    variable=variables[1],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
```

### Stage 3: Detect eutrophication

```python
eutrophication_detection = detect_eutrophication(
    chlorophyll=chlorophyll_field.data,
    oxygen=oxygen_field.data,
    chlorophyll_percentile=chlorophyll_percentile,
    oxygen_threshold=oxygen_threshold,
    min_duration_days=min_duration_days,
    min_area_km2=min_area_km2,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 4: Summarize eutrophic days

```python
eutrophic_days = compute_event_summary_map(
    event_detection=eutrophication_detection,
    data=chlorophyll_field.data,
    summary_mode='event_days',
)
```

### Stage 5: Summarize eutrophication burden

```python
eutrophication_chlorophyll_burden = compute_event_summary_map(
    event_detection=eutrophication_detection,
    data=chlorophyll_field.data,
    summary_mode='burden',
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
