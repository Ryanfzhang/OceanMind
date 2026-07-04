---
skill_id: ocean_watermass_event_association
description: Diagnoses whether event hotspots are organized by named water masses.
input_intent: Event type, detection variables, temperature/salinity for water-mass classification, region, time range, and tiling intent.
output_intent: Tile-level water-mass association summary for event hotspots, hotspot tile map, dominant-watermass tile map, and T-S diagram.
avoid_when:
- Use watermass_analysis for water-mass diagnostics without events and event_frequency_map for hotspot maps without water-mass association.
composes_with:
- ocean_masking_workflow
---
# Ocean Watermass Event Association

## Purpose

This skill diagnoses whether event hotspots are organized by named water masses when the target region is partitioned into equal lon-lat tiles. It combines surface event detection, temperature-salinity density context, tile-level hotspot aggregation, dominant water-mass classification, and hotspot-versus-background distribution comparison.

## Workflow

### Stage 1: Define analysis scope and tiling

```python
# IMPORTANT: Planner fills shared analysis scope here before instantiating tool calls.
lon_range = None  # <-- MODIFY: west/east bounds from the user query or named region.
lat_range = None  # <-- MODIFY: south/north bounds from the user query or named region.
time_range = None  # <-- MODIFY: analysis time window from the user query.
vertical_mode = 'surface'  # Fixed surface-event and surface-watermass analysis.
depth_value = None
depth_range = None
depth_aggregation = 'mean'
threshold = 1.0  # <-- MODIFY: absolute chlorophyll bloom threshold; keep 1.0 unless the query gives another threshold.
percentile_threshold = None  # <-- MODIFY: use only when the query asks for percentile or anomalously high blooms.
min_duration_days = 5  # <-- MODIFY: minimum bloom duration in days.
min_area_km2 = 500  # <-- MODIFY: minimum connected bloom area in km2.
bloom_type = 'auto'
subregion_grid = [30, 30]  # <-- MODIFY: equal lon-lat tile grid as [nx, ny]; a 30x30 grid means [30, 30].
hotspot_quantile = 0.75  # <-- MODIFY: event-score quantile used to flag hotspot tiles.
max_ts_points = 2500  # <-- MODIFY: maximum sampled points for the T-S diagram.
sampling = 'random'
watermass_config_path = None
```

### Stage 2: Load event and water-mass fields

```python
chlorophyll_field = load_dataset(
    variable='chlorophyll',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)

temp_field = load_dataset(
    variable='temp',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)

salt_field = load_dataset(
    variable='salt',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
```

### Stage 3: Detect surface chlorophyll blooms

```python
bloom_detection = detect_algal_blooms(
    chlorophyll=chlorophyll_field.data,
    threshold=threshold,
    percentile_threshold=percentile_threshold,
    min_duration_days=min_duration_days,
    min_area_km2=min_area_km2,
    bloom_type=bloom_type,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
```

### Stage 4: Compute density for water-mass classification

```python
thermo_dataset = assemble_dataset(
    variables={'temp': temp_field.data, 'salt': salt_field.data},
)

density_field = compute_density(
    data=thermo_dataset.data,
)
```

### Stage 5: Classify water masses and associate event hotspots

```python
watermass_association = compute_watermass_event_association(
    event_field=chlorophyll_field.data,
    temp=temp_field.data,
    salt=salt_field.data,
    density=density_field.data,
    event_detection=bloom_detection,
    lon_range=lon_range,
    lat_range=lat_range,
    subregion_grid=subregion_grid,
    hotspot_quantile=hotspot_quantile,
    max_ts_points=max_ts_points,
    sampling=sampling,
    watermass_config_path=watermass_config_path,
)
```

### Stage 6: Build tile maps and T-S diagram

```python
hotspot_tile_map = build_watermass_tile_map(
    association_result=watermass_association,
    map_kind='event_hotspot',
)

dominant_watermass_tile_map = build_watermass_tile_map(
    association_result=watermass_association,
    map_kind='dominant_watermass',
)

watermass_ts_diagram = build_watermass_ts_diagram(
    association_result=watermass_association,
)
```


## Notes

- Named regions are valid region inputs. For `East China Sea`, use the East China Sea shelf bounds when explicit lon/lat bounds are not supplied.
- A tile-grid request such as `30x30`, `30×30`, or `30 by 30` must become `subregion_grid=[30, 30]`.
- Surface chlorophyll bloom hotspot requests should keep `vertical_mode='surface'` for chlorophyll, temperature, salinity, and density inputs.
- `compute_watermass_event_association` requires the bloom `event_detection` result plus the original event field and surface temperature, salinity, and density fields.
- The expected final artifacts are `watermass_association`, `hotspot_tile_map`, `dominant_watermass_tile_map`, and `watermass_ts_diagram`.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
