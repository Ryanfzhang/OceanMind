---
skill_id: ocean_derived_hovmoller_analysis
description: Computes u/v-derived diagnostics such as current speed or relative vorticity and builds time-depth, time-lon, or time-lat Hovmoller diagrams with masks.
input_intent: Derived diagnostic request such as current speed, surface current speed, relative vorticity, shear, strain, or Rossby number; may include polygon mask, isobath mask, region, time range, and depth axis.
output_intent: Time-depth, time-lon, or time-lat Hovmoller diagram for a derived diagnostic.
variables:
- u
- v
avoid_when:
- Use ocean_dynamics_diagnostics for 2D map-ready relative-vorticity diagnostics.
- Use derived_spatial_field for map output and derived_timeseries for time-series output.
composes_with:
- ocean_masking_workflow
---
# Ocean Derived Hovmoller Analysis

## Purpose

Computes a derived diagnostic field and builds a Hovmoller diagram. Use this for current speed, surface current speed, relative vorticity, shear, strain, or other u/v-derived Hovmoller requests, including area-averaged time-depth relative vorticity over a drawn polygon.

## Workflow

### Stage 1: Shared scope and eastward velocity

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['u', 'v']  # fixed source variables for relative vorticity.
field_type = 'vorticity'  # <-- MODIFY: derived diagnostic field to compute.
lon_range = None  # <-- MODIFY: polygon/bounding lon range from the user query or workspace.
lat_range = None  # <-- MODIFY: polygon/bounding lat range from the user query or workspace.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: vertical axis subset only; do not use this for isobath/bathymetry masks.
mask_polygon = None  # <-- MODIFY: drawn polygon points [[lon, lat], ...] when requested.
mask_isobath_depth = None  # <-- MODIFY: bathymetry threshold from the query for "areas deeper than/equal to N m".
mask_isobath_comparison = 'deeper_or_equal'  # <-- MODIFY: use deeper_or_equal for "deeper than" area masks.
diagram_type = 'time_depth'  # <-- MODIFY: time_depth for time-depth diagrams.
aggregate_dim = 'mean'  # <-- MODIFY: spatial aggregation across the masked polygon.
spatial_weighting = 'area_weighted'  # <-- MODIFY: area-weighted mean for regional diagrams.
fixed_lat = None  # <-- MODIFY: required fixed latitude for time_lon diagrams; longitude remains the displayed axis.
fixed_lon = None  # <-- MODIFY: required fixed longitude for time_lat diagrams; latitude remains the displayed axis.

u_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
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
    depth_range=depth_range,
)
```

### Stage 3: Optional polygon analysis mask

```python
polygon_mask = build_polygon_mask(
    data=u_field.data,
    polygon_points=mask_polygon,
)
```

### Stage 4: Optional isobath analysis mask

```python
isobath_mask = build_isobath_mask(
    data=u_field.data,
    isobath_depth=mask_isobath_depth,
    comparison=mask_isobath_comparison,
)
```

### Stage 5: Optional combined mask

```python
analysis_mask = combine_masks(
    masks=[polygon_mask.data, isobath_mask.data],
    operation='and',
)
```

### Stage 6: Compute derived diagnostic field

For current-speed Hovmoller requests (`field_type='speed'`), always compute speed from u/v first with `compute_speed_from_uv`. Do not use `compute_derived_field` for speed; no other speed tool is valid.

```python
derived_field = compute_speed_from_uv(
    u=u_field.data,
    v=v_field.data,
)
```

For relative-vorticity Hovmoller requests (`field_type='vorticity'`), call `compute_derived_field` with the explicit `u=` and `v=` parameters. Do not use `u_data=` or `v_data=` for `compute_derived_field`.

```python
derived_field = compute_derived_field(
    u=u_field.data,
    v=v_field.data,
    field_type='vorticity',
)
```

Use `compute_derived_field` only for supported non-speed diagnostics.

For strain, kinetic energy, eddy kinetic energy, divergence, or Rossby number Hovmoller requests, use the dedicated diagnostic tool instead of `compute_derived_field`, then feed that result into the mask/Hovmoller steps:

```python
derived_field = compute_strain_rate(
    u_data=u_field.data,
    v_data=v_field.data,
)
```

Use `compute_kinetic_energy`, `compute_eddy_kinetic_energy`, `compute_divergence`, or `compute_rossby_number` with the same `u_data`/`v_data` parameter names as needed.

### Stage 7: Optional analysis mask application

```python
masked_derived_field = apply_mask(
    data=derived_field.data,
    mask=analysis_mask.data,
)
```

### Stage 8: Build Hovmoller diagram

```python
hovmoller_result = compute_hovmoller(
    data=masked_derived_field.data,
    diagram_type=diagram_type,
    fixed_lat=fixed_lat,
    fixed_lon=fixed_lon,
    fixed_lon_range=lon_range,
    fixed_lat_range=lat_range,
    aggregate_dim=aggregate_dim,
    spatial_weighting=spatial_weighting,
    depth_range=depth_range,
)
```


## Notes

- For time-depth relative vorticity, do not use `compute_spatial_vorticity_map`; it collapses time/depth into a 2D map.
- Build polygon and isobath masks before applying them to the derived field.
- "Areas deeper than/equal to N m" is a bathymetry/isobath mask, not the diagram's vertical `depth_range`.
- Loader contract: never pass `depth_aggregation` to `load_dataset`; loader depth controls are `vertical_mode`, `depth_value`, and `depth_range`.
- Current speed contract: whenever the requested diagnostic is current speed or surface current speed, the derivation step must be `compute_speed_from_uv(u=..., v=...)`, then feed that artifact into `compute_hovmoller`.
- Relative vorticity contract: for vorticity Hovmoller output, use `compute_derived_field(u=u_field.data, v=v_field.data, field_type='vorticity')`; never use `u_data` or `v_data` with `compute_derived_field`.
- For Hovmoller output, preserve the requested time/depth/longitude/latitude dimensions until `compute_hovmoller`; do not reduce with `compute_spatial_field`.
- Hovmoller axis contract: `diagram_type='time_lon'` means retain longitude on the output axis, so set `fixed_lat` or `fixed_lat_range`; `diagram_type='time_lat'` means retain latitude on the output axis, so set `fixed_lon` or `fixed_lon_range`. Mnemonic: time_lon -> fixed_lat, time_lat -> fixed_lon.
- Fixed-depth contract: `compute_hovmoller` accepts either `depth` or `depth_range`, never both. Prefer one representation consistently: for "at 50 m", either load with `depth_range=[50, 50]` and pass only `depth_range=depth_range` to `compute_hovmoller`, or load the full vertical field and pass only `depth=50`. Do not emit `depth=...` and `depth_range=...` in the same `compute_hovmoller` call.
- For `diagram_type='time_depth'`, use `fixed_lon`/`fixed_lat` for a point Hovmoller or `fixed_lon_range`/`fixed_lat_range` for a regional average.
- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
