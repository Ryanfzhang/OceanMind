---
skill_id: ocean_timeseries_extraction
description: Extracts a time series directly from a tightly loaded 4D data field.
input_intent: Raw variable with already constrained spatial extent and time range, usually from a narrow load step.
output_intent: Direct extracted time series without extra spatial aggregation configuration.
avoid_when:
- Use ocean_timeseries for ordinary regional or point time-series analysis.
composes_with:
- ocean_masking_workflow
---
# Ocean Timeseries Extraction

## Purpose

This skill extracts a time series directly from a loaded 4-D data field without explicit spatial aggregation parameters. Use when the user wants a time series from a dataset and the spatial extent is already constrained by the load step...

## Workflow

### Stage 1: Load variable (with tight spatial bounds)

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].

raw_data = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Extract timeseries

```python
timeseries = extract_timeseries(
    data=raw_data.data,
    lon_range=lon_range,
    lat_range=lat_range,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
