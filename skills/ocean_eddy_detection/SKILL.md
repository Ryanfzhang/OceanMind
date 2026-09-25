---
skill_id: ocean_eddy_detection
description: Detect ocean eddies from eastward and northward velocity fields at task-selected times and depths.
input_intent: Aligned u and v velocity fields over a specified region, time, and depth scope.
output_intent: Per-slice eddy events, statistics, masks, Okubo–Weiss and vorticity fields.
avoid_when:
- Use ocean_eddy_tracking for trajectories through time.
- Use dynamics diagnostics for a vorticity map without eddy detection.
composes_with:
- ocean_masking_workflow
---
# Ocean Eddy Detection

`detect_eddies(u, v)` applies an Okubo–Weiss threshold to one aligned, two-dimensional `lat`/`lon` pair of xarray `DataArray` fields. It separates cyclonic and anticyclonic connected components using the vorticity sign. The defaults are `ow_threshold=-2e-12`, `min_radius_km=30`, `max_radius_km=300`, and `min_pixels=10`. A more negative threshold is stricter. Change these only for a stated scientific reason, and report the values used.

Before writing code, inspect the actual source metadata and a bounded sample with `inspect_data`. Identify the eastward and northward velocity variables, units, coordinate names, available times and depths, and spatial coverage. Use `find_tools("load_dataset")` and `find_tools("detect_eddies")` when their signatures or return values are uncertain. Variable names need not be `u` and `v`: pass the actual names if the loader resolves them. If it does not, read the actual variables with xarray in the analysis script, subset them to the requested region/time/depth, and pass the resulting `DataArray` objects to the detector. Do not guess an alias or silently use the first time or depth. Normalize coordinate names to `lat` and `lon` if needed; the detector requires exactly those two dimensions after selection.

Choose the time/depth slices from the user's question and the coordinates actually present. For a single requested slice, make one detection call. For multiple slices, use one detection stage outside the loop; each call produces its own saved result and stage index entry. The following ordinary Python pattern assumes the analysis runner has activated its stage manager and supplied `tools`; `scope` contains real `load_dataset` arguments and `slices` is a list of requested `(time, depth)` coordinate values:

```python
from packages.analysis_runtime.stages import stage


def run_eddy_detection(tools, *, u_name, v_name, scope, slices,
                       ow_threshold=-2e-12, min_radius_km=30,
                       max_radius_km=300, min_pixels=10):
    with stage("读取流速"):
        u = tools.load_dataset(variable=u_name, **scope)
        v = tools.load_dataset(variable=v_name, **scope)

    summaries = []
    with stage("涡旋检测", total=len(slices), unit="切片") as progress:
        for time, depth in slices:
            progress.set_current(time=time, depth=depth)
            selected = []
            for field in (u, v):
                coordinates = {name: value for name, value in
                               (("time", time), ("depth", depth))
                               if name in field.dims and value is not None}
                horizontal = field.sel(coordinates) if coordinates else field
                if set(horizontal.dims) != {"lat", "lon"}:
                    raise ValueError("Select one time and depth, leaving only lat/lon")
                selected.append(horizontal)
            result = tools.detect_eddies(
                u=selected[0], v=selected[1],
                input_refs=[tools.ref(u), tools.ref(v)],
                ow_threshold=ow_threshold, min_radius_km=min_radius_km,
                max_radius_km=max_radius_km, min_pixels=min_pixels,
            )
            summaries.append({"time": time, "depth": depth,
                              "count": result["statistics"]["total_count"]})
            progress.advance()
    return summaries
```

This example uses the registered loader. For a source it cannot resolve, adapt the loading lines to the inspected data and keep the explicit two-dimensional selection and recorded detection calls. `tools.ref` works only for results returned by wrapped tools; for direct xarray reads, save the bounded input slices with `tools.publish` and pass their artifact IDs as `input_refs` instead. Do not snapshot an entire large source file merely to create a reference. `tools.detect_eddies` saves each complete return value, while `progress.set_current` associates that result with its actual time and depth.

The returned `mask` marks cells satisfying the raw Okubo–Weiss threshold. The `events` list contains only components that also pass the pixel count and radius filters; use `statistics` for accepted event counts. A nonempty mask with zero accepted events is possible, and zero events alone does not justify changing thresholds. The tool does not create a figure. Independent depth slices are not a three-dimensional eddy, and independent dates are not a trajectory; read the tracking or visualization guidance only if the task calls for those products.
