---
skill_id: ocean_spatial_diagnostics
description: Computes map-ready velocity-derived spatial fields from u and v, including current speed, kinetic energy, eddy kinetic energy, strain, divergence, and Rossby number.
input_intent: Velocity fields u and v with region, time range, depth selection, and current-speed, kinetic-energy, strain, divergence, Rossby, or intensity intent.
output_intent: 2D map-ready velocity-derived spatial field.
avoid_when:
- Use ocean_dynamics_diagnostics for relative vorticity or Richardson number.
- Use derived_hovmoller or derived_timeseries for non-map outputs.
composes_with:
- ocean_masking_workflow
---
# Ocean Spatial Diagnostics

## Purpose

Computes map-ready velocity-derived fields from eastward and northward velocity.

## Workflow

### Stage 1: Shared scope and eastward velocity

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['u', 'v']  # <-- MODIFY: current velocity variables in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
season_filter = None  # <-- MODIFY: seasonal subset such as DJF/winter when requested.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
time_aggregation = 'mean'  # <-- MODIFY: temporal reduction for the output map.
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.

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

### Stage 3: Compute current speed

```python
speed_field = compute_speed_from_uv(
    u=u_field.data,
    v=v_field.data,
)
```

For kinetic energy, eddy kinetic energy, strain, divergence, or Rossby number, choose the supported diagnostic tool instead of `compute_speed_from_uv`:

```python
diagnostic_field = compute_kinetic_energy(
    u_data=u_field.data,
    v_data=v_field.data,
)
```

Use `compute_eddy_kinetic_energy(u_data=..., v_data=...)` for EKE, `compute_strain_rate(u_data=..., v_data=...)` for strain rate, `compute_divergence(u_data=..., v_data=...)` for divergence, and `compute_rossby_number(u_data=..., v_data=...)` for Rossby number.

### Stage 4: Reduce to a map

```python
spatial_diagnostic = compute_spatial_field(
    data=speed_field.data,
    time_range=time_range,
    time_aggregation=time_aggregation,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Loader contract: never pass `depth_aggregation` to `load_dataset`; loader depth controls are `vertical_mode`, `depth_value`, and `depth_range`.
- Do not invent additional map-specific tool names. Compute the diagnostic field first, then pass it to `compute_spatial_field`.
- In Stage 4, use `data=diagnostic_field.data` when Stage 3 used `compute_kinetic_energy`, `compute_eddy_kinetic_energy`, `compute_strain_rate`, `compute_divergence`, or `compute_rossby_number`; use `data=speed_field.data` only for current speed.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
