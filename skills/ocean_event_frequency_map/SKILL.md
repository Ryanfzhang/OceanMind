---
skill_id: ocean_event_frequency_map
description: Detects supported events and converts them into a gridded hotspot or frequency map.
input_intent: Event type such as heatwave, bloom, hypoxia, upwelling, or eutrophication with region, time range, depth, and event thresholds.
output_intent: Gridded event-frequency or hotspot map.
avoid_when:
- Use event detection skills for burden, exposure, persistence-days, cumulative-severity, or event-days summary maps.
- Use event_count_timeseries for temporal event activity.
composes_with:
- ocean_masking_workflow
---
# Ocean Event Frequency Map

## Purpose

This skill is the preferred choice for time-range event hotspot analysis when the user explicitly wants to know where events occur most often over a window. It detects supported events over a period and converts them into a gridded cente...

## Workflow

Resolve the requested dataset, event type, variables, region, vertical selection, and time range from the user request.

For heatwave hotspot or heatwave frequency map requests, use this executable branch:

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['temp']  # <-- MODIFY: event variable list in load order; heatwave uses temperature.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
vertical_mode = 'surface'  # <-- MODIFY: surface, bottom, fixed_depth, depth_range, or unspecified.
depth_value = None  # <-- MODIFY: fixed depth in meters when the query names one depth.
depth_range = [0, 0]  # <-- MODIFY: depth interval; heatwave frequency defaults to surface.
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.
normalize = False  # <-- MODIFY: true only when the query asks for normalized frequency or density.

event_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
event_detection = detect_heatwaves(
    temp=event_field.data,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
event_frequency_map = compute_event_frequency_map(
    event_detection=event_detection,
    data=event_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    normalize=normalize,
)
```

The task graph must use the actual harness tools shown above. Do not invent alternate tool names such as `load_zarr`, `detect_heatwave_events`, or `event_frequency`.
Use the source data lon/lat grid by default. Pass `resolution_deg` to `compute_event_frequency_map` only when the user explicitly asks for a coarse output resolution or bin size.

For non-heatwave event frequency requests, keep the same three-stage shape: load the event variable, run the matching event detector, then call `compute_event_frequency_map`.


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
