---
skill_id: ocean_lag_correlation
description: Computes lag correlation between two ocean variables over a region and time period.
input_intent: Two variables or indices with region, time range, depth selections, lag range, and optional detrending/anomaly intent.
output_intent: Lag-correlation curve or lag relationship summary.
avoid_when:
- Use regression_analysis for spatial regression maps and coupling-specific skills for dedicated oxygen-chlorophyll coupling.
composes_with:
- ocean_masking_workflow
---
# Ocean Lag Correlation Analysis

## Purpose

This skill computes the lag correlation between two ocean variables over a region and time period. It extracts a regional mean time series for each variable, then computes the cross-correlation at each lag on either the raw series, the d...

## Workflow

### Stage 1: Load first variable

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
variables = [variable1, variable2]  # <-- MODIFY: requested variable list in load order.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
max_lag = 12  # <-- MODIFY: maximum lag in time steps; default is 12.
confidence_level = 0.95  # <-- MODIFY: confidence level for lag-correlation significance intervals; default is 0.95.

raw_data1 = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range1,
)
```

### Stage 2: Load second variable

```python
raw_data2 = load_dataset(
    variable=variables[1],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range2,
)
```

### Stage 3: Extract regional mean time series for variable 1

```python
timeseries1 = extract_regional_mean(
    data=raw_data1.data,
    lon_range=lon_range,
    lat_range=lat_range,
)
```

### Stage 4: Extract regional mean time series for variable 2

```python
timeseries2 = extract_regional_mean(
    data=raw_data2.data,
    lon_range=lon_range,
    lat_range=lat_range,
)
```

### Stage 5: Raw lag correlation

```python
lag_correlation_raw = compute_lag_correlation(
    timeseries1=timeseries1,
    timeseries2=timeseries2,
    max_lag=max_lag,
    confidence_level=confidence_level,
)
```

### Stage 6: Step 5B (optional): Remove seasonal cycle

```python
lag_correlation_deseasoned = remove_seasonal_cycle(
    timeseries1=timeseries1,
    timeseries2=timeseries2,
    max_lag=max_lag,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
