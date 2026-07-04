---
skill_id: ocean_environment_health_assessment
description: Composes an evidence package for environmental health, suitability, risk, or management assessment.
input_intent: Broad environmental health, suitability, risk, policy, or management question needing multiple indicators and evidence branches.
output_intent: Integrated environment-health assessment card or evidence package.
avoid_when:
- Use specific analysis skills for a single requested map, time series, event detection, or diagnostic.
composes_with:
- ocean_masking_workflow
---
# Ocean Environment Health Template Composer

## Purpose

This skill is a planner-facing diagnostic composer. It combines existing ocean
analysis tools into the smallest evidence package that can answer broad
environmental-health, suitability, risk, policy, or management questions.

It collects evidence only. The final suitability, policy, or management
judgement is produced later by the summary/result synthesizer from completed
tool results. Do not jump directly to `assemble_policy_recommendation_report`
unless the user explicitly asks for a standalone fixed policy report.

Use this skill when a question needs multiple indicators, for example bottom
oxygen stress, hypoxic days, oxygen-deficit burden, warming, stratification, or
bloom pressure.

## Allowed Tools

Only use real tool names. Core evidence tools include:

`load_dataset`, `compute_area_weighted_mean`, `extract_regional_mean`,
`extract_timeseries`, `compute_trend`, `compute_field_trend`,
`compute_spatial_field`, `detect_hypoxia`, `detect_algal_blooms`,
`detect_heatwaves`, `detect_upwelling`, `compute_event_statistics`,
`compute_event_summary_map`, `compute_event_frequency_map`,
`compute_event_spatial_distribution`, `compare_event_periods`,
`assemble_dataset`, `compute_density`,
`compute_vertical_stability_timeseries`, `compute_lag_correlation`,
`compute_regression_map`, `compute_composite_field`, `apply_mask`,
`build_threshold_mask`, `build_condition_mask`, `build_polygon_mask`,
`build_isobath_mask`, `combine_masks`.

Legacy fixed report tools are explicit-only:

`assemble_environment_health_report`, `assemble_policy_recommendation_report`.

## Required Scope

- Resolve `lon_range`, `lat_range`, and `time_range` before any
  `load_dataset` step.
- Treat named sea regions as explicit spatial intent. Use approximate known
  bounds for common regions rather than full dataset extent or workspace bounds.
- Use workspace bounds only for phrases like current region, selected box,
  drawn area, or drawn polygon.
- If a single year is named, use January 1 through December 31 of that year.

## Workflow

### Stage 1: Shared scope

```python
# IMPORTANT: Planner fills these values from the query before emitting workflow_code.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: assessment time window from the user query.
season_filter = None  # <-- MODIFY: seasonal subset such as spring/summer when requested.
confidence_level = 0.95  # <-- MODIFY: trend confidence level.
context_note = 'environment health evidence package'  # <-- MODIFY: short scope note for the assessment.
```

### Stage 2: SST warming branch

```python
sst_field = load_dataset(
    variable='temp',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode='surface',
    depth_range=[0, 0],
)

sst_timeseries = compute_area_weighted_mean(
    data=sst_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=[0, 0],
    depth_aggregation='mean',
)

sst_trend = compute_trend(
    timeseries=sst_timeseries,
    method='linear',
    confidence_level=confidence_level,
)
```

### Stage 3: Bloom pressure branch

```python
bloom_field = load_dataset(
    variable='chlorophyll',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode='surface',
    depth_range=[0, 0],
)

bloom_detection = detect_algal_blooms(
    chlorophyll=bloom_field.data,
    percentile_threshold=90.0,
    min_duration_days=3,
    min_area_km2=0.0,
    vertical_mode='surface',
    depth_range=[0, 0],
)

bloom_statistics = compute_event_statistics(
    events=bloom_detection.events,
    group_by='year',
)
```

### Stage 4: Bottom oxygen branch

```python
bottom_oxygen_field = load_dataset(
    variable='oxygen',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode='bottom',
)

bottom_oxygen_timeseries = compute_area_weighted_mean(
    data=bottom_oxygen_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_aggregation='mean',
)

bottom_oxygen_trend = compute_trend(
    timeseries=bottom_oxygen_timeseries,
    method='linear',
    confidence_level=confidence_level,
)
```

### Stage 5: Stratification branch

```python
temp_field = load_dataset(
    variable='temp',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
)

salt_field = load_dataset(
    variable='salt',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
)

thermo_dataset = assemble_dataset(
    variables={
        'temp': temp_field.data,
        'salt': salt_field.data,
    },
)

density_field = compute_density(
    data=thermo_dataset.data,
)

stability_timeseries = compute_vertical_stability_timeseries(
    density=density_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    weighting='area_weighted',
)

stratification_trend = compute_trend(
    timeseries=stability_timeseries,
    method='linear',
    confidence_level=confidence_level,
)
```

### Stage 6: Integrated assessment

```python
environment_health_assessment = assemble_environment_health_report(
    branches=[
        sst_trend,
        bloom_statistics,
        bottom_oxygen_trend,
        stratification_trend,
    ],
    context_note=context_note,
)
```

Identify the requested evidence branches and emit only the smallest executable diagnostic chain that covers them. For queries that say "do not use bloom frequency", omit the bloom branch from workflow_code and from `branches`.

## Diagnostic Templates

### Bottom Oxygen And Hypoxia

Use for bottom hypoxic days, low-oxygen area, oxygen-deficit burden, hypoxia
hotspots, or oxygen-risk endpoints.

1. `load_dataset(variable="oxygen", vertical_mode="bottom")` ->
   `bottom_oxygen_field`
2. `compute_area_weighted_mean(data="$ref:bottom_oxygen_field.data")` ->
   `bottom_oxygen_timeseries`
3. `compute_trend(timeseries="$ref:bottom_oxygen_timeseries")` ->
   `bottom_oxygen_trend`
4. `detect_hypoxia(oxygen="$ref:bottom_oxygen_field.data",
   vertical_mode="bottom")` -> `hypoxia_detection`
5. `compute_event_statistics(events="$ref:hypoxia_detection.events",
   group_by="year")` -> `hypoxia_statistics`
6. `compute_event_summary_map(event_detection="$ref:hypoxia_detection",
   data="$ref:bottom_oxygen_field.data", summary_mode="event_days")` ->
   `hypoxic_days`
7. `compute_event_summary_map(event_detection="$ref:hypoxia_detection",
   data="$ref:bottom_oxygen_field.data", summary_mode="burden")` ->
   `hypoxia_oxygen_deficit_burden`

### Warming And Heatwave Pressure

Use for warming overlap, heat stress, or heatwave exposure.

1. `load_dataset(variable="temp", vertical_mode="surface")` -> `sst_field`
2. `compute_area_weighted_mean(data="$ref:sst_field.data")` ->
   `sst_timeseries`
3. `compute_trend(timeseries="$ref:sst_timeseries")` -> `sst_trend`
4. For heatwave exposure, load surface temperature as `heatwave_field`, run
   `detect_heatwaves(temp="$ref:heatwave_field.data")`, then summarize with
   `compute_event_summary_map` using `event_days` and/or `burden`.

### Stratification

Use for overlap with stratification or ventilation-risk context.

1. `load_dataset(variable="temp")` -> `temp_field`
2. `load_dataset(variable="salt")` -> `salt_field`
3. `assemble_dataset(variables={"temp": "$ref:temp_field.data",
   "salt": "$ref:salt_field.data"})` -> `thermo_dataset`
4. `compute_density(data="$ref:thermo_dataset.data")` -> `density_field`
5. `compute_vertical_stability_timeseries(density="$ref:density_field.data")`
   -> `stability_timeseries`
6. `compute_trend(timeseries="$ref:stability_timeseries")` ->
   `stratification_trend`

### Bloom Or Chlorophyll Pressure

Use for bloom pressure, chlorophyll screening, eutrophication context, or
red-tide/HAB pressure.

1. `load_dataset(variable="chlorophyll", vertical_mode="surface")` ->
   `bloom_field`
2. `detect_algal_blooms(chlorophyll="$ref:bloom_field.data")` ->
   `bloom_detection`
3. `compute_event_statistics(events="$ref:bloom_detection.events",
   group_by="year")` -> `bloom_statistics`
4. `compute_event_summary_map(event_detection="$ref:bloom_detection",
   data="$ref:bloom_field.data", summary_mode="event_days")` ->
   `bloom_event_days`
5. `compute_event_summary_map(event_detection="$ref:bloom_detection",
   data="$ref:bloom_field.data", summary_mode="burden")` ->
   `bloom_chlorophyll_burden`

Treat bloom/chlorophyll as ecological-pressure screening unless nutrient,
source, or discharge data are explicitly present.

## Management And Policy Questions

- Policy, zoning, monitoring, seasonal-management, aquaculture, and marine
  ranching questions still need executable diagnostics first.
- Do not answer "revise management" from the policy skill alone when the query
  asks for bottom hypoxic days, low-oxygen area, oxygen-deficit burden, hotspot
  expansion, warming, stratification, or bloom overlap.
- Build the evidence package here; the summary layer should translate completed
  evidence into priority zones, evidence strength, limitations, and suggested
  monitoring or seasonal actions.

## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or
  mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Bottom oxygen must use `vertical_mode="bottom"`, not surface or full-depth
  averaging.
- SST, heatwave, bloom, and chlorophyll surface evidence must use
  `vertical_mode="surface"` unless the user explicitly asks otherwise.
- Stratification temperature and salinity loads should retain the water column.
- Event statistics must reference the matching detection result.
