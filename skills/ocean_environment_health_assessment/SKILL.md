---
skill_id: ocean_environment_health_assessment
description: Selects relevant ocean diagnostics and explains the strength and limits of environmental-health evidence.
input_intent: Environmental health, suitability, risk, or management question that may need one or more indicators.
output_intent: An assessment grounded in the requested diagnostics, with missing evidence stated explicitly.
avoid_when:
- Use a specific analysis skill directly for a single requested map, time series, or event detection.
composes_with:
- ocean_masking_workflow
---
# Ocean Environment Health Assessment

## Decide what to measure

Use the user's endpoint to choose diagnostics. This is a decision guide, not a
four-branch recipe. A question about bottom low-oxygen risk may need only bottom
oxygen and hypoxia; it does not trigger SST, bloom, and stratification analyses.
Add another indicator only when the question asks about it or the first result
leaves a specific explanatory gap. A policy question can use the completed
diagnostics here, then read `ocean_policy_recommendation` for bounded actions.

| Question or endpoint | Direct evidence to consider | Read next when needed |
| --- | --- | --- |
| Bottom oxygen status, hypoxic days, deficit burden, hotspots | Bottom oxygen field; `detect_hypoxia`; event statistics or summary maps only for the requested measure | `ocean_hypoxia_detection`, `ocean_event_statistics`, `ocean_event_frequency_map` |
| Warming or heatwave exposure | Surface temperature trend or heatwave detection; distinguish gradual change from events | `ocean_trend_analysis`, `ocean_heatwave_detection` |
| Bloom or chlorophyll pressure | Surface chlorophyll and, if event exposure is asked, bloom detection and statistics | `ocean_bloom_detection`, `ocean_event_statistics` |
| Stratification or weak ventilation context | Multilevel temperature and salinity, density-derived N², requested trend or profile | `ocean_stratification_diagnostics`, `ocean_trend_analysis` |
| A spatial mask, named management zone, or overlapping indicators | Apply a common region, time window, and relevant mask before comparing results | `ocean_masking_workflow` |

Read only the relevant calculation skill(s). Their algorithm and parameter
notes are useful, but some older snippets still use planner wrappers or
reference strings. Check the live tool signature and pass ordinary Python results
directly. If no skill covers a requested indicator, discover the available
tools and compose a small Python calculation; state how the indicator was
defined and save the derived result.

## Scope and execution

- Resolve explicit longitude, latitude, time, depth, units, and threshold
  before loading. A named sea region needs explicit bounds; use workspace
  bounds only when the user refers to the selected or drawn region. A named
  year covers January 1 through December 31 unless the user specifies a season.
- Check that the dataset actually contains the requested variables and period.
  Report missing inputs rather than replacing the requested endpoint with a
  convenient proxy without disclosure.
- Use `tools.<name>(...)` inside the analysis script so each call and result is
  recorded. Group related calls with `stage(...)`; for many dates, depths, or
  regions, place the stage outside the loop and advance its units inside it.
- `load_dataset` returns an `xarray.DataArray`, not a wrapper with a `.data`
  field. `compute_area_weighted_mean` returns a timeseries dictionary for
  `compute_trend`; event detectors return dictionaries whose `events` list can
  be passed to `compute_event_statistics`.

For example, if the user asks only for bottom hypoxic days, the relevant chain
is the following. The scope and threshold must be filled from the actual task
and checked against the oxygen units. These statements run in an active
analysis script with `tools` and `stage` available:

```python
with stage("Load bottom oxygen"):
    oxygen = tools.load_dataset(
        variable="oxygen", lon_range=lon_range, lat_range=lat_range,
        time_range=time_range, vertical_mode="bottom",
    )

with stage("Diagnose hypoxic days"):
    hypoxia = tools.detect_hypoxia(
        oxygen=oxygen, vertical_mode="bottom",
        oxygen_threshold=oxygen_threshold,
    )
    hypoxic_days = tools.compute_event_summary_map(
        event_detection=hypoxia, data=oxygen, summary_mode="event_days",
        input_refs=[tools.ref(hypoxia), tools.ref(oxygen)],
    )
```

For bottom oxygen trend instead, use the same bottom field, then
`compute_area_weighted_mean(data=oxygen)` and `compute_trend(timeseries=mean)`.
For oxygen-deficit burden, request `summary_mode="burden"` on the matching
detector and field. Do not compute these outputs merely because they are
available. Use `compute_event_statistics(events=hypoxia["events"],
group_by="year")` only when event counts or their annual distribution matter.

## Interpret the evidence

- Bottom oxygen and hypoxia are direct oxygen-risk endpoints. Surface SST and
  heatwaves are heat pressure; chlorophyll and blooms screen ecological
  pressure. Bloom/chlorophyll alone cannot identify a nutrient or discharge
  source. Stratification is physical vulnerability or timing context, not
  proof of hypoxia, bloom occurrence, or causation.
- For hypoxia, use bottom water unless a different depth is explicitly asked.
  The detector defaults to oxygen below 60 mmol/m³, severe below 20 mmol/m³,
  minimum 100 km² and three days. Confirm units, cadence, area resolution,
  and whether those thresholds fit the question before interpreting events.
- For SST, heatwave and bloom evidence, use a surface field unless the user
  asks for another layer. For stratification, keep multiple depths of both
  temperature and salinity; do not reduce them to one layer before density.
- The current N² tool finite-differences a potential-density field between
  levels. Mean N² across available model levels (s⁻²) is the generic index;
  peak N² is for explicit strongest-gradient or pycnocline questions. Report
  depth range and aggregation; do not call the result exact TEOS-10
  `gsw.Nsquared` or potential energy anomaly. Keep negative N² and missing
  pairs visible; do not interpolate them into a stronger claim.
- A trend from fewer than five years is a short-record change, not a
  long-term trend. Check `is_significant`, sampling, missing values, and
  seasonality before claiming a direction. Co-occurrence or correlated
  trends do not establish a mechanism.
- Name each finding's region, period, depth, units, method, and supporting
  saved result. Separate direct measurements, proxy context, and unavailable
  evidence. If the requested direct endpoint is missing, say it is untested.

## Optional fixed report

Normally synthesize the requested findings directly from saved artifacts. Use
`assemble_environment_health_report(branches=..., context_note=...)` only when
the user asks for its fixed report format and there are relevant, completed,
compatible evidence items. Its branches are dictionaries with `name`, `result`,
`role`, and optional `evidence_kind`, `metric`, and `worse_when`; raw detector
or trend dictionaries generally lack the normalized `output_type` that this
legacy report expects. Verify that each item is accepted before using it.
Never pass an empty branch list and describe the resulting default verdict as
observed environmental stability. Do not invoke a policy report to stand in
for absent diagnostics.
