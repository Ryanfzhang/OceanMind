---
skill_id: ocean_dynamics_diagnostics
description: Computes map-ready ocean dynamical diagnostics from velocity fields, including relative vorticity, strain, divergence, Rossby number, kinetic energy, and Richardson number.
input_intent: Velocity fields u and v, and optionally temperature/salinity for density-dependent diagnostics, with region, time, depth, and seasonal aggregation.
output_intent: 2D map-ready dynamical diagnostic fields.
avoid_when:
- Use ocean_derived_hovmoller_analysis for time-depth or Hovmoller output.
- Use derived_profile or derived_timeseries for profile or time-series outputs.
composes_with:
- ocean_masking_workflow
---
# Ocean Dynamics Diagnostics

## Purpose

Computes ocean dynamical diagnostics from velocity fields. Use this skill for relative vorticity maps such as winter mean upper-layer vorticity, and for velocity diagnostics such as strain rate, divergence, Rossby number, kinetic energy, and Richardson number.

## Workflow

### Stage 1: Shared scope and eastward velocity

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['u', 'v']  # <-- MODIFY: velocity variables in load order for current diagnostics.
requested_diagnostic = 'relative_vorticity'  # <-- MODIFY: requested diagnostic, such as relative_vorticity.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
season_filter = None  # <-- MODIFY: seasonal subset such as DJF/winter when requested.
depth_range = None  # <-- MODIFY: depth interval from query; "upper 0-750 m" uses [0, 750].
time_aggregation = 'mean'  # <-- MODIFY: temporal reduction for map-ready diagnostics.
depth_aggregation = 'mean'  # <-- MODIFY: vertical reduction for map-ready diagnostics.

u_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    depth_range=depth_range,
)
```

### Stage 2: Load northward velocity

```python
v_field = load_dataset(
    variable=variables[1],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    depth_range=depth_range,
)
```

### Stage 3: Map-ready relative vorticity

```python
spatial_diagnostic = compute_spatial_vorticity_map(
    u=u_field.data,
    v=v_field.data,
    time_range=time_range,
    time_aggregation=time_aggregation,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 4: Other map-ready velocity diagnostics

Choose exactly one diagnostic field tool for non-vorticity map requests, then reduce it with `compute_spatial_field`.

```python
diagnostic_field = compute_strain_rate(
    u_data=u_field.data,
    v_data=v_field.data,
)

spatial_diagnostic = compute_spatial_field(
    data=diagnostic_field.data,
    time_range=time_range,
    time_aggregation=time_aggregation,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

Use `compute_kinetic_energy(u_data=..., v_data=...)` for kinetic energy, `compute_eddy_kinetic_energy(u_data=..., v_data=...)` for eddy kinetic energy, `compute_rossby_number(u_data=..., v_data=...)` for Rossby number, and `compute_divergence(u_data=..., v_data=...)` for divergence. Do not invent additional map-specific tool names.

### Stage 5: Richardson number

For Richardson number, load temperature and salinity over the same scope, compute density, derive buoyancy frequency, then call the Richardson tool with its supported parameter names.

```python
temp_field = load_dataset(
    variable='temp',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    depth_range=depth_range,
)

salt_field = load_dataset(
    variable='salt',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    depth_range=depth_range,
)

thermo_dataset = assemble_dataset(
    variables={'temp': temp_field.data, 'salt': salt_field.data},
)

density_field = compute_density(
    data=thermo_dataset.data,
)

n2_field = compute_derived_field(
    density=density_field.data,
    field_type='buoyancy_frequency',
)

richardson_field = compute_richardson_number(
    n2_data=n2_field.data,
    u_data=u_field.data,
    v_data=v_field.data,
)

spatial_diagnostic = compute_spatial_field(
    data=richardson_field.data,
    time_range=time_range,
    time_aggregation=time_aggregation,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

## Notes

- For a map request for relative vorticity, use `compute_spatial_vorticity_map` directly instead of generated Python or a generic derived-field template.
- For non-vorticity dynamical diagnostics, load the same `u_field` and `v_field`, compute the requested diagnostic tool, then reduce it with `compute_spatial_field` or `extract_regional_mean` according to the requested output.
- Loader contract: never pass `depth_aggregation` to `load_dataset`; loader depth controls are `vertical_mode`, `depth_value`, and `depth_range`.
- `compute_spatial_vorticity_map` accepts `u`, `v`, `time_range`, `time_aggregation`, `depth_range`, `depth_aggregation`, and optional `mask`; do not pass any diagnostic-name parameter to it.
- `compute_richardson_number` requires `n2_data`, `u_data`, and `v_data`; do not use `u=` or `v=` parameter names.
- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
