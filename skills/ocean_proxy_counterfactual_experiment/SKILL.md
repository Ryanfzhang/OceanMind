---
skill_id: ocean_proxy_counterfactual_experiment
description: Runs data-space proxy counterfactual experiments by removing anomalies, climatology departures, or mesoscale components.
input_intent: Counterfactual question about removing anomalies, climatological departures, mesoscale structure, or proxy drivers from available fields.
output_intent: Proxy counterfactual outcome and comparison evidence.
avoid_when:
- Use preprocessing when only a transformed field is requested, and evidence_synthesis when the final ask is graded claim support.
composes_with:
- ocean_masking_workflow
---
# Ocean Proxy Counterfactual Experiment

## Purpose

This skill runs a data-space proxy counterfactual rather than a full dynamical intervention. Use it when the user asks what happens if anomalies, climatological departures, or mesoscale structure are removed from the currently available...

## Workflow

### Stage 1: Shared scope and baseline field

```python
# IMPORTANT: Planner fills these values from the query before emitting workflow_code.
variable = 'temp'  # <-- MODIFY: target variable such as temp, oxygen, chlorophyll, or salt.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
season_filter = None  # <-- MODIFY: seasonal subset such as spring/summer when requested.
vertical_mode = 'surface'  # <-- MODIFY: surface, bottom, fixed_depth, depth_range, or unspecified.
depth_value = None  # <-- MODIFY: fixed depth in meters when requested.
depth_range = None  # <-- MODIFY: depth interval from query.
depth_aggregation = 'mean'  # <-- MODIFY: how to reduce a depth range.
period = 'seasonal'  # <-- MODIFY: 'monthly' or 'seasonal' for anomaly/climatology removal.
cutoff_period = None  # <-- MODIFY: numeric cutoff only for mesoscale-component requests.
component = 'high_pass'  # <-- MODIFY: mesoscale component selection when cutoff_period is used.
mechanism_name = 'proxy_counterfactual'  # <-- MODIFY: short name of the tested mechanism.

baseline_field = load_dataset(
    variable=variable,
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    season_filter=season_filter,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
```

### Stage 2: Construct proxy counterfactual field

```python
counterfactual_field = remove_field_anomaly_component(
    data=baseline_field.data,
    period=period,
)
```

Use `replace_field_with_climatology(data=baseline_field.data, period=period)` when the request explicitly says to replace the field with climatology.

Use `filter_mesoscale_component` instead when the request asks to keep/remove a mesoscale component or gives a numeric cutoff period:

```python
counterfactual_field = filter_mesoscale_component(
    data=baseline_field.data,
    cutoff_period=cutoff_period,
    component=component,
)
```

### Stage 3: Regional baseline and counterfactual outcomes

```python
baseline_timeseries = extract_regional_mean(
    data=baseline_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)

counterfactual_timeseries = extract_regional_mean(
    data=counterfactual_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)

counterfactual_outcome = run_proxy_counterfactual_experiment(
    baseline=baseline_field.data,
    counterfactual=counterfactual_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)

counterfactual_comparison = compare_counterfactual_outcome(
    baseline=baseline_timeseries,
    counterfactual=counterfactual_outcome,
    mechanism_name=mechanism_name,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Loader contract: never pass `depth_aggregation` to `load_dataset`; loader depth controls are `vertical_mode`, `depth_value`, and `depth_range`.
- `remove_field_anomaly_component` and `replace_field_with_climatology` support `period='monthly'` or `period='seasonal'` only. Never put a climatology suffix value or numeric cutoff such as `2.0` in `period`.
- For mesoscale-only or cutoff-period counterfactuals, use `filter_mesoscale_component(data=..., cutoff_period=..., component=...)` before the regional comparison.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
