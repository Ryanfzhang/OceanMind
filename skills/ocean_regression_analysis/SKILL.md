---
skill_id: ocean_regression_analysis
description: Regresses a gridded ocean field onto an internally built index time series.
input_intent: Target gridded variable plus index definition, region, time range, depth selection, and optional lag or anomaly intent.
output_intent: Spatial regression map and regression summary.
avoid_when:
- Use lag_correlation for lag-correlation curves and composite_analysis for phase composites.
composes_with:
- ocean_masking_workflow
---
# Ocean Regression Analysis

## Purpose

This skill builds an internal index time series from the same West Pacific dataset and regresses a gridded field onto that index. It is the v1 regression workflow for single-dataset teleconnection or local process analysis.

## Workflow

### Stage 1: Load the target field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
field_variable = 'chlorophyll'
index_variable = 'temp'
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.
lag = 0
remove_seasonal_cycle = False
significance_level = 0.05

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

### Stage 4: Compute the regression map

```python
regression_map = compute_regression_map(
    field=field_data.data,
    index_timeseries=index_timeseries,
    lag=lag,
    remove_seasonal_cycle=remove_seasonal_cycle,
    significance_level=significance_level,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
