---
skill_id: ocean_transport_analysis
description: Computes volume transport, streamfunction maps, transect transport, and layer transport diagnostics.
input_intent: Velocity fields and optional temperature/salinity with transect, basin, layer, depth range, or transport/streamfunction intent.
output_intent: Transport time series, transport maps, streamfunction maps, layer transport, or transport Hovmoller diagrams.
avoid_when:
- Use dynamics diagnostics for non-transport velocity diagnostics such as vorticity or Rossby number.
composes_with:
- ocean_masking_workflow
---
# Ocean Transport Analysis

## Purpose

This skill is the canonical skill for volume transport, volume transport streamfunction maps, transport streamfunction maps, depth-integrated transport streamfunction maps, transect-integrated transport time series, layer-by-layer transp...

## Workflow

### Stage 1: Load zonal velocity

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['u', 'v']  # Fixed velocity variables in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
season_filter = None  # <-- MODIFY: seasonal subset such as 'winter', 'summer', 'DJF', or None.
time_aggregation = "mean"  # <-- MODIFY: aggregation for streamfunction maps, usually 'mean' unless query asks otherwise.
regional_gauge = None  # <-- MODIFY: use 'gan_fig10_china_seas' for China Seas + western Pacific / Gan Fig. 10 streamfunction maps; enables Area 1/Area 2 rendering and separate colorbars.
transect_points = [[113.0, 18.0], [114.0, 19.0]]  # <-- MODIFY: ordered lon/lat transect points.
n_samples = 100  # <-- MODIFY: number of samples along the transect.
method = 'linear'  # <-- MODIFY: interpolation method.

u_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    depth_range=depth_range,
)
```

### Stage 2: Load meridional velocity

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

### Stage 3: Total volume transport time series

```python
transport_timeseries = compute_volume_transport(
    u=u_field.data,
    v=v_field.data,
    transect_points=transect_points,
    depth_range=depth_range,
    n_samples=n_samples,
    method=method,
)
```

### Stage 4: Total heat transport time series

```text
heat_transport_timeseries = compute_heat_transport(
    u=u_field.data,
    v=v_field.data,
    temp=temp_field.data,
    transect_points=transect_points,
    depth_range=depth_range,
    rho0=rho0,
    cp=cp,
    n_samples=n_samples,
    method=method,
)
```

### Stage 5: Total salt transport time series

```text
salt_transport_timeseries = compute_salt_transport(
    u=u_field.data,
    v=v_field.data,
    salt=salt_field.data,
    transect_points=transect_points,
    depth_range=depth_range,
    n_samples=n_samples,
    method=method,
)
```

### Stage 6: Total freshwater transport time series

```text
freshwater_transport_timeseries = compute_freshwater_transport(
    u=u_field.data,
    v=v_field.data,
    salt=salt_field.data,
    transect_points=transect_points,
    depth_range=depth_range,
    s_ref=s_ref,
    n_samples=n_samples,
    method=method,
)
```

### Stage 7: Transport streamfunction map

```python
transport_streamfunction_map = compute_transport_streamfunction_map(
    u=u_field.data,
    v=v_field.data,
    depth_range=depth_range,
    time_aggregation=time_aggregation,
    regional_gauge=regional_gauge,
)
```

### Stage 8: Time-depth normal flux Hovmoller

```python
transport_flux_hovmoller = compute_transect_normal_flux_hovmoller(
    u=u_field.data,
    v=v_field.data,
    transect_points=transect_points,
    depth_range=depth_range,
    n_samples=n_samples,
    method=method,
)
```

### Stage 9: Layer-by-layer transport

```text
layer_transport = compute_transport_by_layer(
    u=u_field.data,
    v=v_field.data,
    transect_points=transect_points,
    layer_bounds=layer_bounds,
    transport_type=transport_type,
    temp=temp_field.data,
    salt=salt_field.data,
    rho0=rho0,
    cp=cp,
    s_ref=s_ref,
    n_samples=n_samples,
    method=method,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- For volume transport streamfunction map requests, use the exact tool call `compute_transport_streamfunction_map(...)`. Do not rename it to `compute_streamfunction`.
- For time-depth normal volume flux / transport-flux Hovmoller requests, use the exact tool call `compute_transect_normal_flux_hovmoller(...)`. Do not use `generated_python_analysis` for this workflow branch.
- Reuse the already loaded `u_field` and `v_field`; do not load duplicate zonal/meridional current fields for the same scope.
- For seasonal streamfunction requests such as "summer mean", set `season_filter='summer'` or `season_filter='JJA'` on the `load_dataset` calls and keep `time_aggregation='mean'` on `compute_transport_streamfunction_map`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
