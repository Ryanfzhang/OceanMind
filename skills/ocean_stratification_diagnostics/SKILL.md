---
skill_id: ocean_stratification_diagnostics
description: Diagnoses stratification, density structure, and vertical stability from temperature and salinity, using mean N2 by default and peak N2 only for explicit strongest-layer or pycnocline-strength requests.
input_intent: Temperature and salinity fields with region, time range, depth range, and stratification, density, MLD, thermocline, pycnocline, or stability intent.
output_intent: Explicitly defined mean-N2 or peak-N2 stratification metrics, density-derived fields, stability time series, or vertical-structure context.
avoid_when:
- Use vertical_structure for a compact MLD/thermocline/pycnocline overview.
composes_with:
- ocean_masking_workflow
---
# Ocean Stratification Diagnostics

## Purpose

This skill diagnoses stratification, density structure, and vertical stability from the available temperature and salinity fields. Treat a generic `stratification index`, `stratification strength`, or `water-column stability` request as mean squared buoyancy frequency, N2, over the selected water column. Use peak N2 only when the user explicitly asks for peak stratification, the strongest density gradient, or pycnocline strength.

## Workflow

### Stage 1: Load temperature and salinity

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['temp', 'salt']  # Fixed temperature and salinity variable names in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: vertical range needed for density/stratification diagnostics.
depth_aggregation = 'mean'  # <-- MODIFY: keep 'mean' for generic stratification; use 'max' only for explicit peak/strongest-layer intent.

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

### Stage 4: Compute density-derived N2

```python
n2_field = compute_brunt_vaisala_frequency(
    density=density_field.data,
)
```

### Stage 5: Regional vertical stability time series

```python
stability_timeseries = compute_area_weighted_mean(
    data=n2_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 6: Optional point profile

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
- Define generic `stratification index`, `stratification strength`, and `water-column stability` requests as mean N2 over the selected depth interval. State the operational definition, depth interval, vertical aggregation, area weighting, and units (`s^-2`) in the result.
- Use `depth_aggregation='max'` only for explicit peak-stratification, strongest-gradient, or pycnocline-strength requests. Label that result `peak N2`, never an unspecified `Stratification Index`.
- Use `compute_stratification_index(method='surface_bottom_density_difference')` only when the user explicitly asks for a surface-to-bottom density difference. Label it `surface-to-bottom potential-density difference` with units of density, not a density gradient.
- Do not call `compute_vertical_stability_timeseries` in this skill because its current default is the surface-to-bottom density difference.
- The current `mean` reducer is an arithmetic mean across available model depth levels. Call it `mean N2 across available model depth levels` when vertical spacing is uneven; do not claim a thickness-weighted water-column mean.
- Describe the current tool output as density-derived N2. It finite-differences the potential-density field; do not claim that it is the exact TEOS-10 `gsw.Nsquared` calculation.
- Treat mean or peak N2 as stratification or ventilation-vulnerability evidence, not direct bloom, hypoxia, pollution-source, or causal evidence.
- Do not describe any result as potential energy anomaly (PEA); no current tool computes PEA.
- Keep the temperature and salinity load steps multi-level for stability diagnostics; do not collapse them to a single surface, bottom, or fixed-depth layer before computing density.
- Use `compute_density_gradient_profile` only when the user provides a point profile location.
- Loader contract: never pass `depth_aggregation` to `load_dataset`; loader depth controls are `vertical_mode`, `depth_value`, and `depth_range`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
