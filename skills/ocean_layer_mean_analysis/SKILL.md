---
skill_id: ocean_layer_mean_analysis
description: Computes a depth-averaged mean between fixed numeric depths or supported dynamic boundary fields.
input_intent: Raw variable with explicit upper/lower depth bounds, MLD/thermocline boundary fields, region, and time range.
output_intent: Layer-mean field, map, or time series for a raw variable.
avoid_when:
- Use mixed_layer_mean_analysis when the layer is the dynamically diagnosed mixed layer.
- Use derived_* skills for derived diagnostic layer means.
composes_with:
- ocean_masking_workflow
---
# Ocean Layer Mean Analysis

## Purpose

This skill computes the depth-averaged mean of a raw ocean variable between fixed numeric depths or supported dynamic boundary fields. Fixed-depth layer means use numeric `upper_bound_value` and `lower_bound_value`. Feature-bound layer means must first diagnose the boundary fields, then pass them as `upper_bound_field` and/or `lower_bound_field`.

## Workflow

### Stage 1: Load the target variable

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
search_depth_range = depth_range  # <-- MODIFY: vertical search interval for raw variable and boundary diagnostics.
upper_bound_value = 0.0  # <-- MODIFY: upper boundary depth in meters for fixed-depth layers.
lower_bound_value = 50.0  # <-- MODIFY: lower boundary depth in meters for fixed-depth layers.

raw_data = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=search_depth_range,
)
```

### Stage 2: Optional polygon mask

```text
combined_mask = combine_masks(
    masks=[region_mask.data, isobath_mask.data],
    operation=mask_combination_operation,
)
```

### Stage 3: Optional mask application

```text
masked_data = apply_mask(
    data=raw_data.data,
    mask=combined_mask.data,
)
```

### Stage 4A: Fixed numeric depth bounds

```python
layer_mean_field = compute_layer_mean(
    data=raw_data.data,
    upper_bound_value=upper_bound_value,
    lower_bound_value=lower_bound_value,
)
```

### Stage 4B: Dynamic boundary fields - surface to thermocline

Use this branch for requests such as "above thermocline" or "between surface and thermocline".

```python
temp_field = load_dataset(
    variable='temp',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=search_depth_range,
)

thermocline_depth_field = identify_thermocline_depth(
    temp=temp_field.data,
)

layer_mean_field = compute_layer_mean(
    data=raw_data.data,
    upper_bound_value=0.0,
    lower_bound_field=thermocline_depth_field.data,
)
```

### Stage 4C: Dynamic boundary fields - MLD to thermocline

Use this branch only when the query explicitly asks for the layer between MLD and thermocline.

```python
temp_field = load_dataset(
    variable='temp',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=search_depth_range,
)

salt_field = load_dataset(
    variable='salt',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=search_depth_range,
)

thermo_dataset = assemble_dataset(
    variables={'temp': temp_field.data, 'salt': salt_field.data},
)

density_field = compute_density(
    data=thermo_dataset.data,
)

mld_field = identify_mixed_layer_depth(
    density=density_field.data,
)

thermocline_depth_field = identify_thermocline_depth(
    temp=temp_field.data,
)

layer_mean_field = compute_layer_mean(
    data=raw_data.data,
    upper_bound_field=mld_field.data,
    lower_bound_field=thermocline_depth_field.data,
)
```

### Stage 5: Output - time series (`output_mode == "timeseries"` or default)

```python
timeseries = extract_regional_mean(
    data=layer_mean_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
)
```

### Stage 6: Output - spatial map (`output_mode == "spatial"`)

```python
spatial_field_result = compute_spatial_field(
    data=layer_mean_field.data,
    time_range=time_range,
    time_aggregation='mean',
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- For ordinary fixed-depth layer means, use `raw_data.data` with numeric upper/lower bounds.
- For "upper 50 m", set `upper_bound_value = 0.0` and `lower_bound_value = 50.0`; do not pass feature-bound fields.
- For "above thermocline", use `upper_bound_value=0.0` and `lower_bound_field=thermocline_depth_field.data`.
- For "between MLD and thermocline", use `upper_bound_field=mld_field.data` and `lower_bound_field=thermocline_depth_field.data`.
- Never use unsupported parameter names such as `lower_bound_feature`, `upper_bound_feature`, or `feature`.
- Loader contract: never pass `depth_aggregation` to `load_dataset`; loader depth controls are `vertical_mode`, `depth_value`, and `depth_range`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
