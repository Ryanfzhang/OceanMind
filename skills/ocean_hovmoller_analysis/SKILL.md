---
skill_id: ocean_hovmoller_analysis
description: Builds Hovmoller diagrams for raw ocean variables only.
input_intent: Raw scalar variable with region or fixed coordinate, time range, depth selection, and diagram type time-depth, time-lon, or time-lat.
output_intent: Time-depth, time-lon, or time-lat Hovmoller diagram for a raw variable.
avoid_when:
- Use ocean_derived_hovmoller_analysis for vorticity, current speed, surface current speed, shear, strain, or other derived diagnostics, even when the source variables are u and v.
- Use section_analysis for transect-based Hovmoller diagrams.
composes_with:
- ocean_masking_workflow
---
# Ocean Hovmoller Analysis

## Purpose

This skill builds a Hovmoller matrix for timelon, timelat, or timedepth analysis of raw variables. Do not use this skill for current speed, relative vorticity, shear, strain, or any diagnostic derived from u/v; route those requests to `ocean_derived_hovmoller_analysis`.

## Workflow

### Stage 1: Load the source field

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
diagram_type = 'time_depth'  # <-- MODIFY: time_depth, time_lon, or time_lat.
aggregate_dim = 'mean'  # <-- MODIFY: spatial aggregation for regional Hovmoller diagrams.
spatial_weighting = 'area_weighted'  # <-- MODIFY: weighting for regional aggregation.
fixed_lat = None  # <-- MODIFY: required fixed latitude for time_lon diagrams; longitude remains the displayed axis.
fixed_lon = None  # <-- MODIFY: required fixed longitude for time_lat diagrams; latitude remains the displayed axis.

raw_hovmoller_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Optional polygon mask

```text
masked_analysis_base = apply_mask(
    data=raw_hovmoller_field.data,
    mask=region_mask.data,
)
```

### Stage 3: Optional field temporal analysis

```python
hovmoller_result = compute_hovmoller(
    data=raw_hovmoller_field.data,
    diagram_type=diagram_type,
    fixed_lat=fixed_lat,
    fixed_lon=fixed_lon,
    fixed_lat_range=lat_range,
    fixed_lon_range=lon_range,
    aggregate_dim=aggregate_dim,
    spatial_weighting=spatial_weighting,
    depth_range=depth_range,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- For unmasked Hovmoller requests, pass `raw_hovmoller_field.data` directly. For masked requests, pass `masked_analysis_base.data`.
- Hovmoller axis contract: `diagram_type='time_lon'` means retain longitude on the output axis, so set `fixed_lat` or `fixed_lat_range`; `diagram_type='time_lat'` means retain latitude on the output axis, so set `fixed_lon` or `fixed_lon_range`. Mnemonic: time_lon -> fixed_lat, time_lat -> fixed_lon.
- Fixed-depth contract: `compute_hovmoller` accepts either `depth` or `depth_range`, never both. Prefer one representation consistently: for "at 50 m", either load with `depth_range=[50, 50]` and pass only `depth_range=depth_range` to `compute_hovmoller`, or load the full vertical field and pass only `depth=50`. Do not emit `depth=...` and `depth_range=...` in the same `compute_hovmoller` call.
- For `diagram_type='time_depth'`, use `fixed_lon`/`fixed_lat` for a point Hovmoller or `fixed_lon_range`/`fixed_lat_range` for a regional average.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
