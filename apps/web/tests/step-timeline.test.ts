import assert from "node:assert/strict";
import test from "node:test";
import { displayLooseResults, displayStepDetails, displayStepTimeline, formatStepProgressText, latestFigureResultIds } from "../lib/step-card-state";
import type { ResultCardSummary, StepCard } from "../lib/types";

function step(id: string, label: string, status: StepCard["status"], attempt_id: string): StepCard {
  return {
    step_id: id,
    attempt_id,
    human_label: label,
    technical_label: "analysis stage",
    status,
    results_hidden_by_default: false,
    results: [],
    actions: [],
    is_map_bound: false,
    is_expanded: false,
  };
}

function withVisual(card: StepCard): StepCard {
  return { ...card, results: [{
    id: `${card.step_id}_map`, title: card.human_label, type: "image_png",
    headline: card.human_label, description: "", renderer: "summary", metrics: [],
  }] };
}

function withSummary(card: StepCard): StepCard {
  return { ...card, results: [{
    id: `${card.step_id}_stats`, title: card.human_label, type: "json",
    headline: card.human_label, description: "", renderer: "summary", metrics: [],
  }] };
}

test("workflow shows deliverables and collapses successful internal stages", () => {
  const cards = [
    step("load_1", "Load SST", "completed", "attempt_1"),
    step("detect_1", "Detect heatwaves", "completed", "attempt_1"),
    step("summary_failed", "Summarize events", "failed", "attempt_1"),
    withVisual(step("detect_2", "Detect heatwaves", "completed", "attempt_2")),
    step("summary_2", "Summarize events", "completed", "attempt_2"),
    step("verify_2", "Verify statistics", "completed", "attempt_2"),
  ];
  const visible = displayStepTimeline(cards, true);
  assert.deepEqual(visible.map((item) => item.step_id), ["detect_2"]);
  assert.deepEqual(displayStepDetails(cards, visible).map((item) => item.step_id),
    ["load_1", "detect_1", "summary_2", "verify_2"]);
});

test("repeated execution keeps details but does not fill the main view", () => {
  const cards = Array.from({ length: 20 }, (_, index) =>
    step(`stage_${index}`, `Step ${index % 5}`, "completed",
      `attempt_${Math.floor(index / 5)}`));
  assert.equal(displayStepTimeline(cards, false).length, 0);
  assert.equal(displayStepTimeline(cards, true).length, 0);
  assert.equal(displayStepDetails(cards, []).length, 20);
});

test("a retry stays in workflow progress until it completes", () => {
  const cards = [
    step("load_1", "Load data", "completed", "attempt_1"),
    withVisual(step("detect_1", "Detect events", "completed", "attempt_1")),
    step("load_2", "Load data", "running", "attempt_2"),
  ];
  assert.deepEqual(displayStepTimeline(cards, false).map((item) => item.step_id),
    ["detect_1"]);
  assert.deepEqual(displayStepTimeline(cards, true).map((item) => item.step_id),
    ["detect_1"]);
  cards[2] = { ...cards[2], status: "completed" };
  assert.deepEqual(displayStepTimeline(cards, false).map((item) => item.step_id),
    ["detect_1"]);
});

test("two genuine same-name stages in one attempt remain visible", () => {
  const visible = displayStepTimeline([
    withVisual(step("map_day_1", "Compute map", "completed", "attempt_1")),
    withVisual(step("map_day_2", "Compute map", "completed", "attempt_1")),
  ], true);
  assert.deepEqual(visible.map((item) => item.step_id), ["map_day_1", "map_day_2"]);
});

test("later revisions replace the same figure but keep distinct scopes", () => {
  const figure = (id: string, headline: string, attemptIndex: number): ResultCardSummary => ({
    id, title: "Daily mean line", type: "image_png", headline,
    description: "", renderer: "summary", metrics: [], attemptIndex,
  });
  const old = figure("old", "2011-01-01 to 2011-01-07", 0);
  const revised = figure("revised", old.headline, 1);
  const final = figure("final", old.headline, 2);
  const other = figure("other_region", "2011-01-01 to 2011-01-07, South China Sea", 2);
  const steps = [
    { ...step("render_old", "Render chart", "completed", "attempt_1"), results: [old] },
    { ...step("render_final", "Fix labels", "completed", "attempt_3"), results: [final] },
  ];
  assert.equal(latestFigureResultIds([old, revised, final, other], steps, false).size, 0);
  const current = latestFigureResultIds([old, revised, final, other], steps, true);
  assert.deepEqual([...current].sort(), ["final", "other_region"]);
  assert.deepEqual(displayStepTimeline(steps.map((item) => ({
    ...item, results: item.results.filter((result) => current.has(result.id)),
  })), true).map((item) => item.step_id), ["render_final"]);
});

test("failed and unfinished stages never appear as cards", () => {
  const cards = [
    withSummary(step("load", "Load data", "completed", "attempt_1")),
    step("analysis", "Compute field", "running", "attempt_1"),
    step("retry", "Compute field", "failed", "attempt_2"),
  ];
  assert.deepEqual(displayStepTimeline(cards, false).map((item) => item.step_id),
    ["load"]);
  assert.deepEqual(displayStepTimeline(cards, true).map((item) => item.step_id),
    ["load"]);
});

test("a failed step with a partial result is still hidden", () => {
  const delivered = step("partial", "Compute field", "failed", "attempt_1");
  delivered.results = [{
    id: "field", title: "Computed field", type: "json", headline: "Computed field",
    description: "", renderer: "summary", metrics: [],
  }];
  assert.deepEqual(displayStepTimeline([delivered], false), []);
  assert.deepEqual(displayStepTimeline([delivered], true), []);
});

test("a nonvisual analysis keeps only its latest saved outcome in the main view", () => {
  const cards = [
    withSummary(step("inspect", "Inspect store", "completed", "attempt_1")),
    withSummary(step("calculate", "Calculate sea depth", "completed", "attempt_1")),
    step("probe", "Probe serializer", "completed", "attempt_1"),
  ];
  assert.deepEqual(displayStepTimeline(cards, true).map((item) => item.step_id), ["calculate"]);
  assert.deepEqual(displayStepDetails(cards, displayStepTimeline(cards, true)).map((item) => item.step_id),
    ["inspect", "probe"]);
});

test("loose results reuse the latest version of each saved product", () => {
  const card = (id: string, title: string, attemptId: string): ResultCardSummary => ({
    id, title, attemptId, type: "json", headline: title, description: "", renderer: "summary", metrics: [],
  });
  assert.deepEqual(displayLooseResults([
    card("map_old", "Event footprint", "attempt_1"),
    { ...card("owned", "Step-owned result", "attempt_1"), ownerStepId: "detect_1" },
    card("map_new", "Event footprint", "attempt_2"),
    card("stats", "Statistics", "attempt_2"),
  ], [step("detect_1", "Detect", "completed", "attempt_1")]).map((item) => item.id),
  ["map_new", "stats"]);
});

test("a result remains visible when its owning step is hidden", () => {
  const card: ResultCardSummary = {
    id: "saved", ownerStepId: "failed_step", title: "Saved figure", type: "image_png",
    headline: "Saved figure", description: "", renderer: "summary", metrics: [],
  };
  assert.deepEqual(displayLooseResults([card], []).map((item) => item.id), ["saved"]);
});

test("a superseded step does not duplicate its result as a loose card", () => {
  const oldStep = step("old", "Detect events", "completed", "attempt_1");
  const card: ResultCardSummary = {
    id: "old_result", ownerStepId: "old", title: "Old event map", type: "json",
    headline: "Old event map", description: "", renderer: "event", metrics: [],
  };
  assert.deepEqual(displayLooseResults([card], [oldStep]), []);
});

test("an interim compute failure stays out of the visible progress text", () => {
  assert.equal(formatStepProgressText({ phase: "compute_failed", message: "Computation failed: retrying" }), "Continuing analysis");
  assert.equal(formatStepProgressText({ phase: "computing", message: "1 source failed" }), "Continuing analysis");
  assert.equal(formatStepProgressText({ phase: "computing", message: "Computing 1 source" }), "Computing · Computing 1 source");
});
