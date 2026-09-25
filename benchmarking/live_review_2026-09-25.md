# LangGraph live benchmark, 2026-09-25

The [69-question live suite](live_queries.json) contains one query per selected task type, with shorter data windows. Ten parallel shards sent every question once through the Next.js `/api/query/stream` route against the server test worktree at commit `63c311f`. The frozen 240-question `queries.json` was not changed.

## Baseline outcome

| Measure | Result |
| --- | ---: |
| HTTP 200 with valid NDJSON/final event | 69 / 69 |
| Completed | 6 / 69 (Q001, Q003, Q035, Q064, Q066, Q069) |
| Failed with `max_rounds_exceeded` | 63 / 69 |
| No visible execution step | 28 / 69 |
| Saved result cards | 161; all had `renderer=summary` and no interactive workspace data |

The HTTP proxy and streaming envelope worked. Most failures occurred inside agent routing, code execution, or final synthesis. A failed final response may still have saved intermediate artifacts.

## Reproduced issues

1. **Lost interactive results.** Q035 completed its SST time-series calculation but rendered only a summary card. Q005 saved heatwave event-day and burden maps, Q010 saved temperature-front fields, and Q039 saved a regression map; none reached the existing interactive chart or Leaflet map. The UI showed `{"type":"dict"}` instead of scientific data. The LangGraph progress adapter did not project saved NetCDF/JSON artifacts into `workspaceData`.
2. **Hidden intermediate artifacts.** The adapter kept only two cards per stage. `Browse all results` was the only UI path to the rest, contrary to the desired step-by-step view.
3. **Agent loop exhaustion.** 63 queries ended at `max_rounds_exceeded`, including queries that had completed several stages and saved results. Q027 ran 14 stages and saved 14 results before failing. This is the dominant release blocker.
4. **Generated code and runtime mismatch.** Repeated scripts called `publish()` without required `inputs` (for example Q008, Q017, Q018, Q027, Q034, Q044, Q046). Q006 used an invalid `PngFigure(name=...)` argument. Q026 attempted to persist nested `DataArray` objects as JSON. Q057 and Q067 tried to reference derived objects with no artifact ID. Q014 hit an xarray indexing error.
5. **Routing errors.** Data-analysis requests Q020, Q038, Q043, Q050, Q053, Q059, Q061, and Q062 entered `general_answer`. The general-knowledge Q065 entered `dataset_analysis` and failed after web sources were collected.
6. **Frontend/backend contract.** The result artifact endpoint returns truncated content by default. The frontend proxy does not forward pagination parameters, so the browser cannot reconstruct a map or time-series from that endpoint. The server must provide bounded visualization payloads as part of the result protocol.

## Source change under validation

The Mac branch now restores size-bounded interactive payloads for saved spatial fields, time series, profiles, Hovmöller plots, sections, histograms, T–S plots, EOFs, and composite maps. Every intermediate result remains in its stage; the stream sends one result at a time. It keeps the main Leaflet map and chart components, adds inline PNG display, and removes `Browse all results`. Nested `DataArray` values are saved as NetCDF sidecars. This fixes presentation and one persistence error; the agent loop and routing failures above remain separate work.
