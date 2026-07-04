---
skill_id: ocean_composite_analysis
description: Builds an internal index and computes positive, negative, and difference composites of a target field.
input_intent: Target field plus index definition, phase threshold, region, time range, and optional lag or season.
output_intent: Composite maps or fields for positive phase, negative phase, and phase difference.
avoid_when:
- Use regression or correlation skills when the user asks for continuous relationship strength rather than phase composites.
composes_with:
- ocean_masking_workflow
---
# Ocean Composite Analysis

## Purpose

This skill builds an internal index time series from the same West Pacific dataset and computes positive-phase, negative-phase, and difference composites of a target field.

## Workflow

### Stage 1: Load the target field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
field_variable = 'temp'
index_variable = 'salt'
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.
quantile = 0.2
lag = 0
anomaly = True

field_data = load_dataset(
    variable=field_variable,
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Load the index field

```python
index_field = load_dataset(
    variable=index_variable,
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 3: Build the internal index time series

```python
index_timeseries = extract_regional_mean(
    data=index_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation='mean',
)
```

### Stage 4: Compute composites

```python
composite_result = compute_composite_field(
    field=field_data.data,
    index_timeseries=index_timeseries,
    quantile=quantile,
    lag=lag,
    anomaly=anomaly,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
