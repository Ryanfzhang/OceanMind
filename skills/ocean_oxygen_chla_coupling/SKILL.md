---
skill_id: ocean_oxygen_chla_coupling
description: Diagnoses whether oxygen and chlorophyll vary synchronously, with lags, or independently.
input_intent: Oxygen and chlorophyll fields over matching region, time range, and depth/vertical selections.
output_intent: Oxygen-chlorophyll coupling metrics and lag/synchrony summary.
avoid_when:
- Use lag_correlation for generic two-variable lag correlations and mechanism_linkage for event precursor analysis.
composes_with:
- ocean_masking_workflow
---
# Ocean Oxygen Chla Coupling

## Purpose

This skill diagnoses whether oxygen and chla behave synchronously, with short lags, or mostly independently. Use it when the user asks whether low oxygen and chlorophyll changes are coupled strongly enough to support a linked mechanism.

## Workflow

### Stage 1: Load oxygen and chlorophyll fields

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['oxygen', 'chla']  # <-- MODIFY: oxygen and chlorophyll variable names in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: shared analysis time window from the user query.
depth_range = None  # <-- MODIFY: shared depth interval from query; surface requests may use [0, 0].
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.

oxygen_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)

chla_field = load_dataset(
    variable=variables[1],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Reduce both fields to matching regional time series

```python
oxygen_timeseries = extract_regional_mean(
    data=oxygen_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)

chla_timeseries = extract_regional_mean(
    data=chla_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 3: Compute coupling metrics

```python
oxygen_chla_coupling = compute_oxygen_chla_coupling_metrics(
    oxygen_timeseries=oxygen_timeseries,
    chla_timeseries=chla_timeseries,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Coupling metrics require reduced `timeseries_result` inputs for oxygen and chlorophyll; do not pass raw fields directly to `compute_oxygen_chla_coupling_metrics`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
