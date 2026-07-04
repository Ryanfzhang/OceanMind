---
skill_id: ocean_histogram_analysis
description: Computes one-dimensional or two-dimensional histograms from raw ocean variables.
input_intent: Raw variable with region, time range, depth or vertical mode, and histogram binning or variable-pair intent.
output_intent: 1D or 2D histogram of raw variable values.
avoid_when:
- Use derived_histogram_analysis for histograms of derived diagnostics.
composes_with:
- ocean_masking_workflow
---
# Ocean Histogram Analysis

## Purpose

This skill computes one-dimensional or two-dimensional histograms from ocean variables. The new vertical-feature workflow is supported only for the 1D branch.

## Workflow

### Stage 1: Shared scope and raw variable

```python
# IMPORTANT: Planner fills these values from the query before emitting workflow_code.
variable = 'temp'  # <-- MODIFY: requested raw variable such as temp, salt, oxygen, or chlorophyll.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or selected workspace region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or selected workspace region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
season_filter = None  # <-- MODIFY: seasonal subset such as spring/summer when requested.
vertical_mode = None  # <-- MODIFY: surface, bottom, fixed_depth, depth_range, or unspecified.
depth_value = None  # <-- MODIFY: fixed depth in meters when requested.
depth_range = None  # <-- MODIFY: depth interval from query; surface requests may use [0, 0].
n_bins = 50  # <-- MODIFY: requested number of histogram bins.
normalize = False  # <-- MODIFY: True for normalized/probability histograms.
bin_range = None  # <-- MODIFY: histogram value range when requested.

raw_field = load_dataset(
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

### Stage 2: Histogram

```python
histogram_result = compute_histogram(
    data=raw_field.data,
    n_bins=n_bins,
    bin_range=bin_range,
    normalize=normalize,
)
```


## Notes

- Analysis masks are accepted as first-class artifacts; use `apply_mask` or mask-aware tools when a downstream tool does not consume masks directly.
- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
