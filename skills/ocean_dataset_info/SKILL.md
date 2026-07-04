---
skill_id: ocean_dataset_info
description: Answers questions about the active dataset metadata, variables, coverage, and configuration.
input_intent: Dataset metadata questions about variables, aliases, spatial extent, temporal coverage, depth range, backend, chunks, or paths.
output_intent: Dataset information response without running scientific analysis tools.
avoid_when:
- Do not use for compute, plot, detect, trend, map, profile, or time-series analysis requests.
composes_with:
- ocean_masking_workflow
---
# Ocean Dataset Info

## Purpose

This is the single skill for questions about the active dataset's configuration and metadata. Use it for dataset introductions, variable lists, variable aliases, spatial/temporal/depth coverage, resolution, backend, chunking, and Zarr st...

## Workflow

### Stage 1: Read active dataset config and metadata

```python
dataset_info = get_dataset_info(
    dataset='current',
)
```


## Notes

- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
