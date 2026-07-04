---
skill_id: ocean_vertical_structure
description: Diagnoses mixed-layer, thermocline, and pycnocline depth features of the water column.
input_intent: Temperature and salinity fields with region, time range, and vertical-structure overview intent.
output_intent: Mixed-layer depth, thermocline depth, and pycnocline depth diagnostic artifacts.
avoid_when:
- Use stratification_diagnostics for broader density/stability mechanism analysis.
composes_with:
- ocean_masking_workflow
---
# Ocean Vertical Structure Analysis

## Purpose

This skill diagnoses the main vertical structure features of the water column and returns mixed-layer depth, thermocline depth, and pycnocline depth diagnostics. Use when the user wants an overview of vertical...

## Workflow

### Stage 1: Load temperature

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['temp', 'salt']  # Fixed temperature/salinity context in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].

temp_data = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Load salinity

```python
salt_data = load_dataset(
    variable=variables[1],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 3: Assemble multi-variable dataset

```python
ts_dataset = assemble_dataset(
    variables={'temp': temp_data.data, 'salt': salt_data.data},
)
```

### Stage 4: Compute density

```python
density_field = compute_density(
    data=ts_dataset.data,
)
```

### Stage 5: Identify mixed-layer depth

```python
mixed_layer_depth = identify_mixed_layer_depth(
    density=density_field.data,
)
```

### Stage 6: Identify thermocline depth

```python
thermocline_depth = identify_thermocline_depth(
    temp=temp_data.data,
)
```

### Stage 7: Identify pycnocline depth

```python
pycnocline_depth = identify_pycnocline_depth(
    density=density_field.data,
)
```

## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Compute `density_field = compute_density(data=ts_dataset.data)` before calling mixed-layer or pycnocline tools.
- Treat `mixed_layer_depth`, `thermocline_depth`, and `pycnocline_depth` as the final artifacts; do not reassemble these diagnostic outputs with `assemble_dataset`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
