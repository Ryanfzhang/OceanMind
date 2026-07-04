---
skill_id: ocean_derived_profile_analysis
description: Computes a derived diagnostic field and extracts a vertical profile.
input_intent: Derived diagnostic source variables with point or region selection, time range, and vertical profile request.
output_intent: Vertical profile of a derived diagnostic field.
avoid_when:
- Use ocean_profile_analysis for raw-variable profiles.
- Use derived_hovmoller when the user asks for time-depth diagrams.
composes_with:
- ocean_masking_workflow
---
# Ocean Derived Vertical Profile Analysis

## Purpose

This skill computes a derived diagnostic field first, then extracts a vertical profile. In v1, the vertical-feature workflow supports overlay only. Do not use it for volume transport streamfunction, transport streamfunction, depth-integr...

## Workflow

### Stage 1: Shared scope and source variables

```python
# IMPORTANT: Planner fills these values from the query before emitting workflow_code.
variables = ['u', 'v']  # <-- MODIFY: source variables; use ['u', 'v'] for speed/vorticity, ['temp', 'salt'] for density or buoyancy diagnostics.
field_type = 'speed'  # <-- MODIFY: speed, vorticity, horizontal_gradient, vertical_gradient, buoyancy_frequency, or density-related diagnostic.
lon_range = None  # <-- MODIFY: optional bounding lon range from the user query; for point profiles, use a small non-zero window around target_lon.
lat_range = None  # <-- MODIFY: optional bounding lat range from the user query; for point profiles, use a small non-zero window around target_lat.
time_range = None  # <-- MODIFY: analysis time window from the user query.
season_filter = None  # <-- MODIFY: seasonal subset such as spring/summer when requested.
target_lon = 115.0  # <-- MODIFY: profile longitude from the query.
target_lat = 20.0  # <-- MODIFY: profile latitude from the query.
method = 'nearest'  # <-- MODIFY: nearest or interpolation method requested by the user.

primary_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
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

For `buoyancy_frequency` or density-related profiles, compute density before deriving the field:

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

### Stage 4: Vertical profile

```python
derived_profile = extract_vertical_profile(
    data=derived_field.data,
    lon=target_lon,
    lat=target_lat,
    method=method,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Loader contract: never pass `depth_aggregation` to `load_dataset`; loader depth controls are `vertical_mode`, `depth_value`, and `depth_range`.
- Point profile loader contract: for a point profile, load a small non-zero lon/lat window around `target_lon`/`target_lat`; never use zero-width ranges such as `lon_range=[target_lon, target_lon]` or `lat_range=[target_lat, target_lat]`. The exact point is selected only in `extract_vertical_profile(lon=target_lon, lat=target_lat, ...)`.
- Current speed contract: whenever the requested diagnostic is current speed or surface current speed, the derivation step must be `compute_speed_from_uv(u=..., v=...)`.
- Emit only one Stage 3 derivation branch. Do not output all alternative snippets in the same workflow.
- For single-variable gradients, pass `variable=variables[0]` to `compute_derived_field`; otherwise the gradient tool cannot identify the source variable.
- For `buoyancy_frequency`, call `compute_density` first and then call `compute_derived_field(density=density_field.data, field_type='buoyancy_frequency')`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
