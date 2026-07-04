---
skill_id: ocean_watermass_analysis
description: Runs water-mass workflows including T-S diagrams, isopycnal surfaces, and isopycnal layer means.
input_intent: Temperature and salinity, and optionally another variable, with region, time range, depth selection, and water-mass or isopycnal intent.
output_intent: T-S diagram, isopycnal surface, or isopycnal layer-mean result.
avoid_when:
- Use stratification_diagnostics for generic stability/density structure without water-mass framing.
composes_with:
- ocean_masking_workflow
---
# Ocean Watermass Analysis

## Purpose

This skill provides three water-mass-oriented workflows for the single West Pacific dataset: - tsdiagram - isopycnalsurface - isopycnallayermean

## Workflow

### Stage 1: Load temperature and salinity

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = ['temp', 'salt']  # Fixed temperature and salinity variable names in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
max_points = 20000  # <-- MODIFY: maximum points sampled for the T-S diagram.
sampling = 'random'  # <-- MODIFY: random or head.

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

### Stage 2: Branch A: T-S diagram

```python
ts_diagram = compute_ts_diagram(
    temp=temp_field.data,
    salt=salt_field.data,
    max_points=max_points,
    sampling=sampling,
)
```

### Stage 3: Optional density field for isopycnal branches

```text
ts_dataset = assemble_dataset(
    variables={'temp': temp_field.data, 'salt': salt_field.data},
)

density_field = compute_density(
    data=ts_dataset.data,
)
```

### Stage 4: Branch B: Isopycnal surface

```text
isopycnal_surface = extract_isopycnal_surface(
    density=density_field.data,
    target_sigma0=target_sigma0,
)
```

### Stage 5: Branch C: Isopycnal layer mean

```text
isopycnal_layer_mean = compute_isopycnal_layer_mean(
    data=temp_field.data,
    density=density_field.data,
    sigma0_upper=sigma0_upper,
    sigma0_lower=sigma0_lower,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- T-S diagram requests need only loaded temperature and salinity fields. Isopycnal surface/layer requests must first assemble temp+salt and compute `density_field`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
