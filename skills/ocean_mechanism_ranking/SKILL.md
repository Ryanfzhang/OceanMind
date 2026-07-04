---
skill_id: ocean_mechanism_ranking
description: Computes proxy diagnostics and ranks candidate mechanisms into an ordered mechanism card.
input_intent: Mechanism attribution question with event, tracer, or proxy diagnostics and candidate drivers.
output_intent: Ranked mechanism support card.
avoid_when:
- Use evidence_synthesis for graded claim support and specific diagnostic skills for single computations.
composes_with:
- ocean_masking_workflow
---
# Ocean Mechanism Ranking

## Purpose

This skill computes only the candidate mechanism branches requested by the user, preserves their requested order, and ranks the resulting evidence items. It is not a free-form explanation skill: every branch must end in a concrete score-like artifact before `rank_mechanism_support`.

## Workflow

### Stage 1: Shared scope

```python
# IMPORTANT: Planner fills these values from the query before emitting workflow_code.
lon_range = None  # <-- MODIFY: west/east bounds.
lat_range = None  # <-- MODIFY: south/north bounds.
time_range = None  # <-- MODIFY: analysis time window.
season_filter = None  # <-- MODIFY: spring/summer/winter/autumn subset when requested.
depth_aggregation = 'mean'  # Use only in reduction/diagnostic tools, never in load_dataset.
```

### Stage 2: Branch A - stratification or density-gradient control

Use this branch for `stratification`, `density_gradient`, `upper-100 m`, or `upper-200 m` mechanism scores.

```python
strat_depth_range = [0, 200]  # <-- MODIFY: [0, 100] for upper-100 m requests.

temp_field = load_dataset(
    variable='temp',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    depth_range=strat_depth_range,
)

salt_field = load_dataset(
    variable='salt',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    depth_range=strat_depth_range,
)

thermo_dataset = assemble_dataset(
    variables={'temp': temp_field.data, 'salt': salt_field.data},
)

density_field = compute_density(
    data=thermo_dataset.data,
)

stratification_field = compute_stratification_index(
    density=density_field.data,
    depth_range=strat_depth_range,
    method='density_gradient',
)

stratification_timeseries = compute_area_weighted_mean(
    data=stratification_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=strat_depth_range,
    depth_aggregation=depth_aggregation,
)

stratification_score = compute_stratification_response_index(
    stratification=stratification_timeseries,
)
```

### Stage 3: Branch B - horizontal advection or partial budget

Use this branch for tracer advection scores. For bottom oxygen, use `vertical_mode='bottom'` and no depth-range load aggregation.

```python
tracer_variable = 'oxygen'  # <-- MODIFY: oxygen, temp, salt, or chlorophyll.
tracer_vertical_mode = 'bottom'  # <-- MODIFY: 'bottom' for bottom-layer oxygen, 'surface' for surface tracers, or None.
tracer_depth_range = None  # <-- MODIFY: [0, 0] for surface fields or [0, 50] for upper-layer requests.

tracer_field = load_dataset(
    variable=tracer_variable,
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode=tracer_vertical_mode,
    depth_range=tracer_depth_range,
)

u_field = load_dataset(
    variable='u',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode=tracer_vertical_mode,
    depth_range=tracer_depth_range,
)

v_field = load_dataset(
    variable='v',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode=tracer_vertical_mode,
    depth_range=tracer_depth_range,
)

local_tendency_field = compute_local_tendency(
    data=tracer_field.data,
)

local_tendency_timeseries = compute_area_weighted_mean(
    data=local_tendency_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=tracer_depth_range,
    depth_aggregation=depth_aggregation,
)

horizontal_advection_timeseries = compute_tracer_horizontal_advection_timeseries(
    data=tracer_field.data,
    u_data=u_field.data,
    v_data=v_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=tracer_depth_range,
    depth_aggregation=depth_aggregation,
    weighting='area_weighted',
)

budget_residual = compute_budget_residual(
    local_tendency=local_tendency_timeseries,
    horizontal_advection=horizontal_advection_timeseries,
)

advection_score = compare_budget_term_magnitudes(
    local_tendency=local_tendency_timeseries,
    horizontal_advection=horizontal_advection_timeseries,
    residual=budget_residual,
)
```

### Stage 4: Branch C - oxygen-chlorophyll coupling

Use `chlorophyll` as the dataset variable for user terms like `chla`.

```python
oxygen_field = load_dataset(
    variable='oxygen',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode='bottom',
)

chlorophyll_field = load_dataset(
    variable='chlorophyll',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode='surface',
    depth_range=[0, 0],
)

oxygen_timeseries = extract_regional_mean(
    data=oxygen_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_aggregation=depth_aggregation,
)

chlorophyll_timeseries = extract_regional_mean(
    data=chlorophyll_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=[0, 0],
    depth_aggregation=depth_aggregation,
)

coupling_score = compute_oxygen_chla_coupling_metrics(
    oxygen_timeseries=oxygen_timeseries,
    chla_timeseries=chlorophyll_timeseries,
)
```

### Stage 5: Branch D - mesoscale organization

Use this branch for mesoscale organization, eddy/front/flow alignment, or bloom hotspot organization scores.

```python
mesoscale_variable = 'chlorophyll'  # <-- MODIFY: chlorophyll for bloom hotspots, temp for warm anomalies.

mesoscale_field = load_dataset(
    variable=mesoscale_variable,
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode='surface',
    depth_range=[0, 0],
)

u_surface = load_dataset(
    variable='u',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode='surface',
    depth_range=[0, 0],
)

v_surface = load_dataset(
    variable='v',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode='surface',
    depth_range=[0, 0],
)

front_proximity = compute_front_proximity_index(
    data=mesoscale_field.data,
    percentile=90.0,
)

eddy_influence = compute_eddy_influence_mask(
    u_data=u_surface.data,
    v_data=v_surface.data,
    percentile=90.0,
)

gradient_alignment = compute_tracer_gradient_alignment(
    data=mesoscale_field.data,
    u_data=u_surface.data,
    v_data=v_surface.data,
)

flow_context = compute_flow_structure_context(
    u_data=u_surface.data,
    v_data=v_surface.data,
)

mesoscale_score = grade_evidence_strength(
    evidence_items=[front_proximity, eddy_influence, gradient_alignment, flow_context],
)
```

### Stage 6: Final ranking

```python
mechanism_ranking = rank_mechanism_support(
    evidence_items=[stratification_score, advection_score, coupling_score, mesoscale_score],
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Emit only the branches requested by the user. If the user says "exactly these diagnostics", do not add optional diagnostics.
- Preserve requested branch order when appending to `evidence_items`.
- Loader contract for all branches: never pass `depth_aggregation` or `var` to `load_dataset`; use `variable`, `vertical_mode`, `depth_value`, and `depth_range`.
- For velocity-derived evidence, use `u_data` and `v_data` parameter names. Do not use `u=`, `v=`, `data_u=`, or `data_v=`.
- For chlorophyll, prefer the dataset variable name `chlorophyll` even when the query says `chla`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
