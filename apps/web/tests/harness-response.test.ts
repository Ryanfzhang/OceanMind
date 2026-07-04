import assert from "node:assert/strict";
import test from "node:test";

import { mergeOrAppendStepCard } from "../lib/step-card-state";
import { hydrateResultCards, normalizeWorkspaceDataByResult } from "../lib/workspace-results";
import type { ResultCardSummary, StepCard } from "../lib/types";

test("hydrates harness result cards with existing workspace_data_by_result contract", () => {
  const workspaceByResult = normalizeWorkspaceDataByResult({
    hypoxia_timeseries: {
      resultSeries: [
        { label: "2020-01-01", value: 0.2 },
        { label: "2020-02-01", value: 0.4 },
      ],
      seriesLabels: { result: "Hypoxia area fraction" },
    },
  });

  const cards: ResultCardSummary[] = [
    {
      id: "hypoxia_timeseries",
      title: "Hypoxia time series",
      type: "timeseries_result",
      headline: "2 points",
      description: "Harness-projected time series",
      renderer: "timeseries",
      metrics: [],
      surface: "inline",
    },
  ];

  const hydrated = hydrateResultCards(cards, workspaceByResult);
  assert.equal(hydrated[0].workspaceData?.resultSeries?.length, 2);
  assert.equal(hydrated[0].workspaceData?.seriesLabels?.result, "Hypoxia area fraction");
});

test("step card merge allows recovery after transient reflection failure", () => {
  const failedCard: StepCard = {
    step_id: "generated_analysis",
    human_label: "Generated analysis",
    technical_label: "Run generated spatial analysis",
    status: "failed",
    results_hidden_by_default: true,
    results: [],
    interpretation: "",
    actions: [],
    is_map_bound: false,
    is_expanded: true,
    error: "Generated code returned the wrong shape.",
  };

  const runningCard: StepCard = {
    ...failedCard,
    status: "running",
    is_expanded: false,
    progress: {
      phase: "reflection",
      message: "Updating workflow",
    },
    error: undefined,
  };

  const recovered = mergeOrAppendStepCard([failedCard], runningCard);
  assert.equal(recovered[0].status, "running");
  assert.equal(recovered[0].error, undefined);
  assert.equal(recovered[0].progress?.phase, "reflection");

  const completedCard: StepCard = {
    ...runningCard,
    status: "completed",
    progress: undefined,
    results: [
      {
        id: "generated_analysis_retry",
        title: "Recovered analysis",
        type: "spatial_field_result",
        headline: "Map ready",
        description: "Recovered result",
        renderer: "summary",
        metrics: [],
        surface: "map",
      },
    ],
  };

  const completed = mergeOrAppendStepCard(recovered, completedCard);
  assert.equal(completed[0].status, "completed");
  assert.equal(completed[0].error, undefined);
  assert.equal(completed[0].results[0].id, "generated_analysis_retry");
});
