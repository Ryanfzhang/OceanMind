---
skill_id: ocean_section_analysis
description: Extracts along-transect sections for raw variables or derived diagnostics and can build section Hovmoller diagrams.
input_intent: Transect points or section path, raw or derived variable intent, time range, and depth selection.
output_intent: Section field, transect profile, or section-based Hovmoller diagram.
avoid_when:
- Use profile_analysis for single-point vertical profiles and hovmoller_analysis for box-based Hovmoller diagrams.
composes_with:
- ocean_masking_workflow
---
# Ocean Section Analysis

## Purpose

This skill extracts an along-transect section from the single West Pacific dataset. It supports both base variables and derived diagnostics, and can optionally collapse the section into a section-based Hovmoller diagram. It also supports...

## Workflow

### Stage 1: `speed` or `vorticity`

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
transect_points = [[113.0, 18.0], [114.0, 19.0]]  # <-- MODIFY: ordered lon/lat transect points.
n_samples = 100  # <-- MODIFY: number of samples along the transect.
method = 'linear'  # <-- MODIFY: interpolation method.

section_source = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Build a derived field

```text
masked_section_source = apply_mask(
    data=section_source.data,
    mask=region_mask.data,
)
```

### Stage 3: Optional field temporal analysis

```python
section_result = extract_transect_section(
    data=section_source.data,
    transect_points=transect_points,
    n_samples=n_samples,
    method=method,
)
```

### Stage 4: Optional section Hovmoller

```text
section_hovmoller = compute_section_hovmoller(
    section=section_result,
    diagram_type=diagram_type,
    fixed_depth=fixed_depth,
    depth_range=depth_range,
    fixed_distance_km=fixed_distance_km,
    distance_range_km=distance_range_km,
    aggregate_method=aggregate_method,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- For ordinary section requests, pass `section_source.data` directly. For masked section requests, pass `masked_section_source.data`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
