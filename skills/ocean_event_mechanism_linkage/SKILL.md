---
skill_id: ocean_event_mechanism_linkage
description: Links detected events to precursor fields, lead-lag structure, and background contrasts.
input_intent: Event type plus candidate driver fields, region, time range, depth, and lead-lag or precursor intent.
output_intent: Mechanism linkage evidence including precursor composites, lead-lag regression, and event/background contrasts.
avoid_when:
- Use event detection or event statistics skills when the user only asks where or when events occur.
composes_with:
- ocean_masking_workflow
---
# Ocean Event Mechanism Linkage

## Purpose

This skill links detected events to precursor fields, lead-lag structure, and event-vs-background contrasts. Use it when the user asks what conditions usually precede heatwaves, hypoxia, or blooms, or which driver changed as events becam...

## Workflow

### Stage 1: Shared scope and event/driver fields

```python
# IMPORTANT: Planner fills these values from the query before emitting workflow_code.
event_variable = 'oxygen'  # <-- MODIFY: oxygen for hypoxia, chlorophyll for blooms, temp for heatwaves.
driver_variable = 'temp'  # <-- MODIFY: candidate precursor or contrast field.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
season_filter = None  # <-- MODIFY: seasonal subset such as spring/summer when requested.
event_vertical_mode = 'bottom'  # <-- MODIFY: bottom for hypoxia, surface for blooms/heatwaves unless otherwise requested.
driver_depth_range = None  # <-- MODIFY: precursor depth interval such as [0, 200] when requested.
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce driver depth ranges.
lead_steps = 7  # <-- MODIFY: lead time in model steps/days for precursor composites.
max_lag = 30  # <-- MODIFY: maximum lag for lead-lag analysis.

event_field = load_dataset(
    variable=event_variable,
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode=event_vertical_mode,
)

driver_field = load_dataset(
    variable=driver_variable,
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    depth_range=driver_depth_range,
)
```

### Stage 2: Detect events

```python
event_detection = detect_hypoxia(
    oxygen=event_field.data,
    oxygen_threshold=2.0,
    min_area_km2=0.0,
    min_duration_days=1,
    vertical_mode=event_vertical_mode,
)
```

### Stage 3: Default robust event-condition context

```python
event_statistics = compute_event_statistics(
    events=event_detection.events,
    group_by='month',
)

driver_timeseries = extract_regional_mean(
    data=driver_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=driver_depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 4: Optional event-condition contrast

Use this only when the query explicitly asks for event-versus-background contrast and the requested window is likely to contain both event and non-event samples.
The supported tool is `compute_event_condition_contrast` with `field`, `events`, `lon_range`, `lat_range`, optional `depth_range`, and `depth_aggregation`.

### Stage 5: Optional precursor and lead-lag products

Use these tools only when the user explicitly asks for a precursor composite or lead-lag regression and the requested time window is long enough to contain pre-event samples.
The supported optional tools are `compute_event_precursor_composite(field=..., events=..., lead_steps=..., depth_range=..., depth_aggregation=...)` and `compute_event_lead_lag_regression(field=..., events=..., lon_range=..., lat_range=..., depth_range=..., depth_aggregation=..., max_lag=...)`.

Only when the user explicitly requests a grid/subregion partition, use `compute_event_condition_contrast` with `partition_mode='lon_lat_grid'`, `subregion_grid=[2, 2]`, and `subregion_weighting='area_weighted'`.

For bloom events, use `detect_algal_blooms(chlorophyll=event_field.data, ...)`. For heatwave events, use `detect_heatwaves(temp=event_field.data, ...)`.


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Loader contract: never pass `depth_aggregation` to `load_dataset`; loader depth controls are `vertical_mode`, `depth_value`, and `depth_range`.
- For ordinary event-vs-background contrast, do not pass `partition_mode` at all. Use `partition_mode='lon_lat_grid'` only for explicit subregion/grid contrast.
- Use `event_statistics` plus `driver_timeseries` as the default robust output for short windows and broad questions about event-associated or usually preceding conditions.
- Add `compute_event_condition_contrast` only when the query explicitly asks for event-versus-background contrast and the window likely contains both event and non-event samples.
- Do not make `compute_event_precursor_composite` or `compute_event_lead_lag_regression` the only final evidence for short event windows; they can fail when no valid pre-event time steps exist.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
