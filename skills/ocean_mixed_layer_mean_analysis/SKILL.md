---
skill_id: ocean_mixed_layer_mean_analysis
description: Computes the mean of a raw ocean variable over the dynamically diagnosed mixed layer.
input_intent: Raw variable plus temperature/salinity context needed to diagnose mixed-layer depth over region and time.
output_intent: Mixed-layer mean field, map, or time series.
avoid_when:
- Use layer_mean_analysis for fixed-depth layers or non-mixed-layer feature bounds.
composes_with:
- ocean_masking_workflow
---
# Ocean Mixed-Layer Mean Analysis

## Purpose

This skill computes the mean of an ocean variable over the dynamically diagnosed mixed layer. Unlike fixed-depth layer means, the integration depth adapts to the actual mixed-layer depth at each grid point and time step. Use when the use...

## Workflow

### Stage 1: Load target variable

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['temp', 'salt']  # Fixed temperature/salinity context in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].

var_data = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Load temperature (for MLD diagnosis)

```python
temp_data = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 3: Load salinity (for MLD diagnosis)

```python
salt_data = load_dataset(
    variable=variables[1],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 4: Assemble temp + salt dataset

```python
ts_dataset = assemble_dataset(
    variables={'temp': temp_data.data, 'salt': salt_data.data},
)
```

### Stage 5: Compute density

```python
density_field = compute_density(
    data=ts_dataset.data,
)
```

### Stage 6: Identify mixed-layer depth

```python
mld_field = identify_mixed_layer_depth(
    density=density_field.data,
)
```

### Stage 7: Compute mixed-layer mean

```python
mld_mean_field = compute_mixed_layer_mean(
    data=var_data.data,
    mixed_layer_depth=mld_field.data,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
