---
skill_id: ocean_stratification_diagnostics
description: Diagnoses stratification, density structure, and vertical stability from temperature and salinity.
input_intent: Temperature and salinity fields with region, time range, depth range, and stratification, density, MLD, thermocline, pycnocline, or stability intent.
output_intent: Stratification metrics, density-derived fields, stability time series, or vertical-structure context.
avoid_when:
- Use vertical_structure for a compact MLD/thermocline/pycnocline overview.
composes_with:
- ocean_masking_workflow
---
# Ocean Stratification Diagnostics

## Purpose

This skill diagnoses stratification, density structure, and vertical stability from the available temp and salt fields. Use it when the user wants a mechanism-ready view of whether stronger layering, shallower mixing, or vertical structu...

## Workflow

### Stage 1: Load temperature and salinity

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['temp', 'salt']  # Fixed temperature and salinity variable names in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: vertical range needed for density/stratification diagnostics.

temp_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)

salt_field = load_dataset(
    variable=variables[1],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Assemble temperature-salinity dataset

```python
thermo_dataset = assemble_dataset(
    variables={'temp': temp_field.data, 'salt': salt_field.data},
)
```

### Stage 3: Compute density

```python
density_field = compute_density(
    data=thermo_dataset.data,
)
```

### Stage 4: Regional vertical stability time series

```python
stability_timeseries = compute_vertical_stability_timeseries(
    density=density_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    weighting='area_weighted',
)
```

### Stage 5: Optional point profile

```text
density_gradient_profile = compute_density_gradient_profile(
    density=density_field.data,
    lon=profile_point_lon,
    lat=profile_point_lat,
    method='nearest',
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- `compute_density` requires an assembled temperature-salinity dataset as `data`; do not pass precomputed density, response, or stratification time series into it.
- `compute_vertical_stability_timeseries` is the default time-series output for stability, stratification, and vertical-stability requests.
- Keep the temperature and salinity load steps multi-level for stability diagnostics; do not collapse them to a single surface, bottom, or fixed-depth layer before computing density.
- Use `compute_density_gradient_profile` only when the user provides a point profile location.
- Loader contract: never pass `depth_aggregation` to `load_dataset`; loader depth controls are `vertical_mode`, `depth_value`, and `depth_range`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
