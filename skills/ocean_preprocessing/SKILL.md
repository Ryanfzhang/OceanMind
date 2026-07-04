---
skill_id: ocean_preprocessing
description: Preprocesses an ocean field before downstream analysis.
input_intent: Raw field with requested filtering, smoothing, anomaly removal, climatology replacement, or preprocessing operation.
output_intent: Preprocessed data field for downstream use.
avoid_when:
- Use analysis skills directly when no explicit preprocessing operation is requested.
composes_with:
- ocean_masking_workflow
---
# Ocean Preprocessing

## Purpose

This skill performs preprocessing on an ocean field before downstream analysis.

## Workflow

### Stage 1: Load the source field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
filter_type = 'lowpass'  # <-- MODIFY: lowpass, highpass, bandpass, or smoothing operation.
cutoff_period = 7.0  # <-- MODIFY: cutoff period or spatial scale for the requested filter.
dimension = 'time'  # <-- MODIFY: time or spatial.
method = 'running_mean'  # <-- MODIFY: butterworth, gaussian, running_mean, or supported local method.
order = 4  # <-- MODIFY: filter order where applicable.

raw_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Branch by operation

```python
processed_field = filter_data(
    data=raw_field.data,
    filter_type=filter_type,
    cutoff_period=cutoff_period,
    dimension=dimension,
    method=method,
    order=order,
)
```

### Stage 3: Interpolate branch

```text
processed_field = interpolate_data(
    data=raw_field.data,
    lon_points=lon_points,
    lat_points=lat_points,
    depth_points=depth_points,
    time_points=time_points,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
