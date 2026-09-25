---
skill_id: ocean_spatial_field_analysis
description: Converts a raw ocean variable into a map-ready two-dimensional spatial field.
input_intent: Raw variable with region, time range, depth selection, optional polygon mask, temporal aggregation, or feature-depth controls.
output_intent: 2D map-ready field for a raw variable.
avoid_when:
- Use ocean_dynamics_diagnostics or derived_spatial_field_analysis for derived velocity diagnostics.
- Use timeseries, profile, histogram, or Hovmoller skills for non-map outputs.
composes_with:
- ocean_masking_workflow
---
# Ocean Spatial Field Analysis

Use this skill when the requested product is a horizontal map of an observed or modelled variable. Read the source metadata and a bounded sample with `inspect_data` before writing code: identify the actual variable name, units, horizontal coordinates, available dates, depth coordinate, and coverage. A data path or variable unknown to `load_dataset` can be read with xarray inside the analysis script after inspecting it; do not silently substitute another variable. Inspect `find_tools("load_dataset")` and `find_tools("compute_spatial_field")` if their signatures are uncertain.

Choose the spatial, time, and vertical scope from the request and available coordinates. `load_dataset` requires `variable`, `lon_range`, and `lat_range`; its optional controls are `time_range`, `season_filter`, `vertical_mode`, `depth_value`, and `depth_range`. Use `vertical_mode="surface"` for surface, `"bottom"` for each column's deepest valid level, `"fixed_depth"` with `depth_value` for a named depth, and `"depth_range"` for an explicitly requested numeric layer. A bottom map is **not** a fixed deepest coordinate or a column mean. Do not invent a depth range for an unspecified vertical scope; ask or inspect the source if the choice changes the answer.

`compute_spatial_field` reduces any remaining time and depth dimensions to a 2D `lat`/`lon` result with `lon`, `lat`, `values`, and `metadata`. Time choices are `mean`, `max`, `min`, `std`, and `median`; depth choices are `mean`, `max`, `min`, `integral`, and `surface`. The defaults are `mean` for each. Use `integral` only when a depth integral, with its resulting units, is scientifically requested. For a layer mean, subset the requested layer and use `depth_aggregation="mean"`. For bottom or single-depth selection, leave `depth_range=None` at this step and keep a supported depth aggregation such as `"mean"`; no full-column depth reduction is then needed. State the period, vertical choice, aggregations, units, and any missing cells when reporting the map.

The following is ordinary Python for one unmasked map. The analysis runner supplies `tools`; its stage manager records both tool calls, their results, and progress. Fill the arguments from the actual request and source metadata, then run it as one saved analysis script:

```python
from packages.analysis_runtime.stages import stage


def run_spatial_map(tools, *, variable, lon_range, lat_range, time_range,
                    vertical_mode, depth_value=None, depth_range=None,
                    time_aggregation="mean", depth_aggregation="mean"):
    with stage("Load source field"):
        field = tools.load_dataset(
            variable=variable, lon_range=lon_range, lat_range=lat_range,
            time_range=time_range, vertical_mode=vertical_mode,
            depth_value=depth_value, depth_range=depth_range,
        )

    with stage("Compute spatial field"):
        spatial_map = tools.compute_spatial_field(
            data=field, time_range=time_range,
            time_aggregation=time_aggregation,
            depth_range=depth_range if vertical_mode == "depth_range" else None,
            depth_aggregation=depth_aggregation,
            input_refs=[tools.ref(field)],
        )
    return spatial_map
```

The loader and map tool consume `DataArray` values directly; do not append `.data` to their results. The map tool does **not** accept `lon_range`, `lat_range`, `vertical_mode`, or `depth_value`. If the source has nonstandard coordinate names, rename them to `lat` and `lon` in the analysis code before calling the map tool. If a loader cannot resolve the variable, use the inspected xarray field, apply the same requested subsets, publish a bounded input artifact if lineage is needed, and pass the resulting `DataArray` directly to `tools.compute_spatial_field`.

For a polygon or isobath request, read `ocean_masking_workflow` and choose only the necessary mask builders. `tools.build_polygon_mask(data=field, polygon_points=...)` and `tools.build_isobath_mask(data=field, isobath_depth=..., comparison=...)` return boolean `DataArray` masks. Combine multiple masks with `tools.combine_masks(masks=[...], operation="and")` when the question requires their intersection. Pass the resulting mask as `mask=...` to `tools.compute_spatial_field`; that tool applies it before reductions. `tools.apply_mask(data=field, mask=...)` is useful when another downstream tool lacks a mask argument, but it is not an obligatory extra step here. Record the mask artifact IDs as input references for the final call.

If the request names a feature depth, such as mixed-layer or thermocline depth, inspect the corresponding feature tool and its source requirements, compute that feature field first, and map the feature field. Do not replace a feature-depth calculation with a fixed depth, a layer mean, or the first available level. If the request asks for several independent maps, select the actual slices and use one stage with progress over the requested set; retain each complete result and its time/depth label instead of collapsing them into one unlabeled field.
