---
skill_id: ocean_derived_histogram_analysis
description: Computes a derived diagnostic field and builds a histogram of that diagnostic.
input_intent: Derived diagnostic request with source variables, region, time range, vertical selection, and histogram binning intent.
output_intent: One-dimensional or two-dimensional histogram of a derived diagnostic field.
avoid_when:
- Use ocean_histogram_analysis for histograms of raw variables.
- Use map or time-series skills when distribution output is not requested.
composes_with:
- ocean_masking_workflow
---
# Ocean Derived Histogram Analysis

## Purpose

This skill computes a derived diagnostic field and then builds a histogram of that field. It supports feature-depth histograms and layer-mean histograms. Do not use it for volume transport streamfunction, transport streamfunction, depth-...

## Workflow

### Stage 1: Shared scope and source variables

```python
# IMPORTANT: Planner fills these values from the query before emitting workflow_code.
variables = ['u', 'v']  # <-- MODIFY: source variables; use ['u', 'v'] for current speed/vorticity, ['temp', 'salt'] for buoyancy diagnostics, or one variable for gradients.
field_type = 'speed'  # <-- MODIFY: speed, vorticity, horizontal_gradient, vertical_gradient, or buoyancy_frequency.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
season_filter = None  # <-- MODIFY: seasonal subset such as spring/summer when requested.
vertical_mode = None  # <-- MODIFY: surface, bottom, fixed_depth, depth_range, or unspecified.
depth_value = None  # <-- MODIFY: fixed depth in meters when requested.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
n_bins = 50  # <-- MODIFY: requested number of histogram bins.
normalize = False  # <-- MODIFY: True for normalized/probability histograms.
bin_range = None  # <-- MODIFY: histogram value range when requested.

primary_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
```

### Stage 2: Optional second source variable

```python
secondary_field = load_dataset(
    variable=variables[1],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
```

### Stage 3: Assemble and derive diagnostic

Choose exactly one derivation pattern that matches `field_type`.

For current-speed diagnostics (`field_type='speed'`), always compute the speed magnitude from u/v with `compute_speed_from_uv`. Do not use `compute_derived_field` for speed; no other speed tool is valid.

```python
derived_field = compute_speed_from_uv(
    u=primary_field.data,
    v=secondary_field.data,
)
```

For relative vorticity, assemble `{'u': primary_field.data, 'v': secondary_field.data}` and call `compute_derived_field(dataset=source_dataset.data, field_type='vorticity')`. Use `compute_derived_field` only for supported non-speed diagnostics.

For one-variable gradients such as `horizontal_gradient` or `vertical_gradient`:

```python
source_dataset = assemble_dataset(
    variables={variables[0]: primary_field.data},
)

derived_field = compute_derived_field(
    dataset=source_dataset.data,
    field_type='horizontal_gradient',
    variable=variables[0],
)
```

For `buoyancy_frequency`, compute density before deriving the field:

```python
thermo_dataset = assemble_dataset(
    variables={'temp': primary_field.data, 'salt': secondary_field.data},
)

density_field = compute_density(
    data=thermo_dataset.data,
)

derived_field = compute_derived_field(
    density=density_field.data,
    field_type='buoyancy_frequency',
)
```

### Stage 4: Histogram

```python
histogram_result = compute_histogram(
    data=derived_field.data,
    n_bins=n_bins,
    bin_range=bin_range,
    normalize=normalize,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Loader contract: never pass `depth_aggregation` to `load_dataset`; loader depth controls are `vertical_mode`, `depth_value`, and `depth_range`.
- Current speed contract: whenever the requested diagnostic is current speed or surface current speed, the derivation step must be `compute_speed_from_uv(u=..., v=...)`.
- Emit only one Stage 3 derivation branch. Do not output all alternative snippets in the same workflow.
- For single-variable gradients, pass `variable=variables[0]` to `compute_derived_field`; otherwise the gradient tool cannot identify the source variable.
- For `buoyancy_frequency`, call `compute_density` first and then call `compute_derived_field(density=density_field.data, field_type='buoyancy_frequency')`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
