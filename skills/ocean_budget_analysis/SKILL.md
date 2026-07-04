---
skill_id: ocean_budget_analysis
description: Diagnoses process-oriented tracer budget terms from tracer and velocity fields.
input_intent: Tracer variable plus u and v velocity over a region, time range, and depth selection.
output_intent: Partial budget diagnostics comparing local tendency, advection, residuals, or ranked process terms.
avoid_when:
- Use simple trend, anomaly, or spatial field skills when no process budget or tendency attribution is requested.
composes_with:
- ocean_masking_workflow
---
# Ocean Budget Analysis

## Purpose

This skill diagnoses process-oriented tracer tendencies from the same single dataset. Its default workflow is a partial budget that uses one tracer plus horizontal velocity (u, v) to compare local tendency against horizontal advection. T...

## Workflow

### Stage 1: Load tracer and horizontal velocity fields

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested tracer variable only; velocity variables are fixed u and v.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: numeric depth interval from query; upper-50 m means [0, 50].
depth_aggregation = 'mean'  # <-- MODIFY: use mean for layer-budget comparison.
weighting = 'area_weighted'  # <-- MODIFY: area_weighted or volume_weighted.

tracer_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)

u_field = load_dataset(
    variable='u',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)

v_field = load_dataset(
    variable='v',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
```

### Stage 2: Local tendency as regional time series

```python
local_tendency_field = compute_local_tendency(
    data=tracer_field.data,
)

local_tendency_ts = extract_regional_mean(
    data=local_tendency_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 3: Horizontal advection as regional time series

```python
horizontal_advection_ts = compute_tracer_horizontal_advection_timeseries(
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

### Stage 4: Residual and mechanism comparison

```python
residual_ts = compute_budget_residual(
    local_tendency=local_tendency_ts,
    horizontal_advection=horizontal_advection_ts,
)

comparison_result = compare_budget_term_magnitudes(
    local_tendency=local_tendency_ts,
    horizontal_advection=horizontal_advection_ts,
    residual=residual_ts,
)
```

## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Budget comparison tools require reduced `timeseries_result` inputs. Do not pass raw 3D/4D fields from `compute_local_tendency` or `compute_horizontal_advection` directly into `compare_budget_term_magnitudes`.
- `compute_local_tendency` and `compute_horizontal_advection` use comparable tracer-per-day tendency units; reduce both with the same `lon_range`, `lat_range`, `depth_range`, and `depth_aggregation` before comparing magnitudes.
- Include `compute_budget_residual` when the user asks about residual uncertainty.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
