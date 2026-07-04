---
skill_id: ocean_evidence_synthesis
description: Runs proxy diagnostics and counterfactual experiments to produce graded evidence statements.
input_intent: Mechanism, causal support, or evidence-strength question with available tracer, event, or diagnostic fields.
output_intent: Evidence synthesis separating supported, limited, and untestable claims.
avoid_when:
- Use mechanism_ranking for a ranked mechanism list and specific analysis skills for direct computations.
composes_with:
- ocean_masking_workflow
---
# Ocean Evidence Synthesis

## Purpose

This skill runs only the evidence branches requested by the user, then grades support as supported, limited, or not testable. It must not invent extra diagnostics. For multi-branch requests, preserve the branch order in the query.

## Workflow

### Stage 1: Shared scope

```python
lon_range = None  # <-- MODIFY: west/east bounds.
lat_range = None  # <-- MODIFY: south/north bounds.
time_range = None  # <-- MODIFY: analysis time window.
season_filter = None  # <-- MODIFY: spring/summer/winter/autumn subset when requested.
depth_aggregation = 'mean'  # Use only in reduction/diagnostic tools, never in load_dataset.
claim = 'The requested mechanism claim is supported by the requested diagnostics.'
requested_strength = 'supported'
```

### Stage 2: Branch A - stratification evidence

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

stratification_timeseries = extract_regional_mean(
    data=stratification_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=strat_depth_range,
    depth_aggregation=depth_aggregation,
)

stratification_score = compute_stratification_response_index(
    stratification=stratification_timeseries,
)

stratification_evidence = grade_evidence_strength(
    evidence_items=[stratification_score],
)
```

### Stage 3: Branch B - horizontal-advection evidence

```python
tracer_variable = 'oxygen'  # <-- MODIFY: oxygen, temp, salt, or chlorophyll.
tracer_vertical_mode = 'bottom'  # <-- MODIFY: 'bottom' for bottom oxygen, 'surface' for surface tracers, or None.
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

local_tendency_timeseries = extract_regional_mean(
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

advection_evidence = grade_evidence_strength(
    evidence_items=[advection_score],
)
```

### Stage 4: Branch C - proxy-counterfactual evidence

```python
counterfactual_variable = 'oxygen'  # <-- MODIFY: target variable.
counterfactual_vertical_mode = 'bottom'  # <-- MODIFY: bottom, surface, or None.
counterfactual_depth_range = None  # <-- MODIFY: [0, 0] for surface or a requested depth range.

baseline_field = load_dataset(
    variable=counterfactual_variable,
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode=counterfactual_vertical_mode,
    depth_range=counterfactual_depth_range,
)

counterfactual_field = remove_field_anomaly_component(
    data=baseline_field.data,
    period='seasonal',
)

baseline_timeseries = extract_regional_mean(
    data=baseline_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=counterfactual_depth_range,
    depth_aggregation=depth_aggregation,
)

counterfactual_outcome = run_proxy_counterfactual_experiment(
    baseline=baseline_field.data,
    counterfactual=counterfactual_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=counterfactual_depth_range,
    depth_aggregation=depth_aggregation,
)

counterfactual_comparison = compare_counterfactual_outcome(
    baseline=baseline_timeseries,
    counterfactual=counterfactual_outcome,
    mechanism_name='proxy_counterfactual',
)

counterfactual_evidence = grade_evidence_strength(
    evidence_items=[counterfactual_comparison],
)
```

### Stage 5: Branch D - oxygen-chlorophyll coupling evidence

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

oxygen_chla_score = compute_oxygen_chla_coupling_metrics(
    oxygen_timeseries=oxygen_timeseries,
    chla_timeseries=chlorophyll_timeseries,
)

oxygen_chla_evidence = grade_evidence_strength(
    evidence_items=[oxygen_chla_score],
)
```

### Stage 6: Branch E - mesoscale organization or bloom linkage evidence

```python
chlorophyll_surface = load_dataset(
    variable='chlorophyll',
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
    data=chlorophyll_surface.data,
    percentile=90.0,
)

eddy_influence = compute_eddy_influence_mask(
    u_data=u_surface.data,
    v_data=v_surface.data,
    percentile=90.0,
)

gradient_alignment = compute_tracer_gradient_alignment(
    data=chlorophyll_surface.data,
    u_data=u_surface.data,
    v_data=v_surface.data,
)

flow_context = compute_flow_structure_context(
    u_data=u_surface.data,
    v_data=v_surface.data,
)

bloom_detection = detect_algal_blooms(
    chlorophyll=chlorophyll_surface.data,
    threshold=1.0,
    min_duration_days=5,
    min_area_km2=500,
    bloom_type='auto',
    vertical_mode='surface',
    depth_range=[0, 0],
    depth_aggregation=depth_aggregation,
)

bloom_linkage = compute_event_lead_lag_regression(
    field=chlorophyll_surface.data,
    events=bloom_detection.events,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=[0, 0],
    depth_aggregation=depth_aggregation,
    max_lag=30,
)

mesoscale_bloom_evidence = grade_evidence_strength(
    evidence_items=[front_proximity, eddy_influence, gradient_alignment, flow_context, bloom_linkage],
)
```

### Stage 7: Final evidence report

```python
evidence_report = assemble_mechanism_evidence_report(
    mechanism_scores=[
        stratification_evidence,
        advection_evidence,
        counterfactual_evidence,
        oxygen_chla_evidence,
        mesoscale_bloom_evidence,
    ],
    context_note='Evidence report assembled from the requested branches only.',
)

claim_check = check_claim_support_level(
    claim=claim,
    evidence=evidence_report,
    requested_strength=requested_strength,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Emit only the evidence branches requested by the user. If the query says "exactly these branches", do not add optional branches.
- Preserve requested branch order in `evidence_items`.
- Loader contract for all branches: never pass `depth_aggregation` or `var` to `load_dataset`; use `variable`, `vertical_mode`, `depth_value`, and `depth_range`.
- Put `depth_aggregation` only on reducers and diagnostic tools such as `extract_regional_mean`, `compute_tracer_horizontal_advection_timeseries`, and event diagnostics.
- For velocity-derived evidence, use `u_data` and `v_data` parameter names. Do not use `u=`, `v=`, `data_u=`, or `data_v=`.
- For chlorophyll, prefer the dataset variable name `chlorophyll` even when the query says `chla`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
