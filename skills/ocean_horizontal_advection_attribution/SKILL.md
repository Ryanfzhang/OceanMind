---
skill_id: ocean_horizontal_advection_attribution
description: Attributes tracer changes to horizontal advection using tracer and velocity fields.
input_intent: Tracer field plus u and v velocity over region, time range, and depth selection.
output_intent: Horizontal advection time series, map, or attribution diagnostic.
avoid_when:
- Use budget_analysis for broader partial budgets and mechanism_ranking for multi-mechanism scoring.
composes_with:
- ocean_masking_workflow
---
# Ocean Horizontal Advection Attribution

## Purpose

This skill tests whether a tracer anomaly is more consistent with horizontal transport than with local accumulation. It is designed for six-variable workflows where the system can estimate local tendency, horizontal advection, and a resi...

## Workflow

### Stage 1: Shared scope and tracer/velocity fields

```python
# IMPORTANT: Planner fills these values from the query before emitting workflow_code.
tracer_variable = 'temp'  # <-- MODIFY: tracer variable such as temp, oxygen, chlorophyll, or salt.
velocity_variables = ['u', 'v']  # <-- MODIFY: velocity variables in eastward/northward order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
season_filter = None  # <-- MODIFY: seasonal subset such as spring/summer when requested.
vertical_mode = None  # <-- MODIFY: surface, bottom, fixed_depth, depth_range, or unspecified.
depth_value = None  # <-- MODIFY: fixed depth in meters when requested.
depth_range = None  # <-- MODIFY: depth interval from query; upper 50 m should use [0, 50].
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.
weighting = 'area_weighted'  # <-- MODIFY: regional weighting for attribution time series.

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

### Stage 2: Local tendency and horizontal advection

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

### Stage 3: Residual and term ranking

```python
budget_residual = compute_budget_residual(
    local_tendency=local_tendency_timeseries,
    horizontal_advection=horizontal_advection_timeseries,
)

advection_attribution = compare_budget_term_magnitudes(
    local_tendency=local_tendency_timeseries,
    horizontal_advection=horizontal_advection_timeseries,
    residual=budget_residual,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
