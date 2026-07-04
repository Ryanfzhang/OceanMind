---
skill_id: ocean_partial_budget_closure
description: Assembles a compact partial tracer budget and keeps the residual visible.
input_intent: Tracer plus available u/v and optional supporting fields with region, time range, and depth selection.
output_intent: Partial budget closure card with residual and term magnitudes.
avoid_when:
- Use budget_analysis for full process-oriented tendency analysis or horizontal_advection_attribution for advection-only attribution.
composes_with:
- ocean_masking_workflow
---
# Ocean Partial Budget Closure

## Purpose

This skill assembles a compact partial budget from the currently available fields and keeps the residual visible. Use it when the user asks for a budget-style interpretation but only temp / salt / u / v / oxygen / chla are available.

## Workflow

### Stage 1: Shared scope and tracer/velocity fields

```python
# IMPORTANT: Planner fills these values from the query before emitting workflow_code.
tracer_variable = 'oxygen'  # <-- MODIFY: tracer variable such as oxygen, temp, chlorophyll, or salt.
velocity_variables = ['u', 'v']  # <-- MODIFY: velocity variables in eastward/northward order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
season_filter = None  # <-- MODIFY: seasonal subset such as spring/summer when requested.
vertical_mode = None  # <-- MODIFY: surface, bottom, fixed_depth, depth_range, or unspecified.
depth_value = None  # <-- MODIFY: fixed depth in meters when requested.
depth_range = None  # <-- MODIFY: depth interval from query; upper 50 m should use [0, 50].
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.
weighting = 'area_weighted'  # <-- MODIFY: regional weighting for budget time series.

tracer_field = load_dataset(
    variable=tracer_variable,
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)

u_field = load_dataset(
    variable=velocity_variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)

v_field = load_dataset(
    variable=velocity_variables[1],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
```

### Stage 2: Partial budget terms

```python
local_tendency_field = compute_local_tendency(
    data=tracer_field.data,
)

local_tendency_timeseries = compute_area_weighted_mean(
    data=local_tendency_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)

horizontal_advection_timeseries = compute_tracer_horizontal_advection_timeseries(
    data=tracer_field.data,
    u_data=u_field.data,
    v_data=v_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
    weighting=weighting,
)
```

### Stage 3: Residual closure

```python
budget_residual = compute_budget_residual(
    local_tendency=local_tendency_timeseries,
    horizontal_advection=horizontal_advection_timeseries,
)

budget_closure = compare_budget_term_magnitudes(
    local_tendency=local_tendency_timeseries,
    horizontal_advection=horizontal_advection_timeseries,
    residual=budget_residual,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
