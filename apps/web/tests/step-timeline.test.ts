import assert from "node:assert/strict";
import test from "node:test";
import { displayLooseResults, displayStepTimeline } from "../lib/step-card-state";
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

test("completed workflow keeps all distinct successful steps without retry groups", () => {
  const visible = displayStepTimeline([
    step("load_1", "Load SST", "completed", "attempt_1"),
    step("detect_1", "Detect heatwaves", "completed", "attempt_1"),
    step("summary_failed", "Summarize events", "failed", "attempt_1"),
    step("detect_2", "Detect heatwaves", "completed", "attempt_2"),
    step("summary_2", "Summarize events", "completed", "attempt_2"),
    step("verify_2", "Verify statistics", "completed", "attempt_2"),
  ], true);
  assert.deepEqual(visible.map((item) => item.step_id),
    ["load_1", "detect_2", "summary_2", "verify_2"]);
});

test("two genuine same-name stages in one attempt remain visible", () => {
  const visible = displayStepTimeline([
    step("map_day_1", "Compute map", "completed", "attempt_1"),
    step("map_day_2", "Compute map", "completed", "attempt_1"),
  ], true);
  assert.deepEqual(visible.map((item) => item.step_id), ["map_day_1", "map_day_2"]);
});

test("cards remain visible during execution and empty failed cards disappear at the end", () => {
  const cards = [
    step("load", "Load data", "completed", "attempt_1"),
    step("analysis", "Compute field", "running", "attempt_1"),
    step("retry", "Compute field", "failed", "attempt_2"),
  ];
  assert.deepEqual(displayStepTimeline(cards, false).map((item) => item.step_id),
    ["load", "analysis", "retry"]);
  assert.deepEqual(displayStepTimeline(cards, true).map((item) => item.step_id), ["load"]);
});

test("a failed step with a delivered result stays in the final timeline", () => {
  const delivered = step("partial", "Compute field", "failed", "attempt_1");
  delivered.results = [{
    id: "field", title: "Computed field", type: "json", headline: "Computed field",
    description: "", renderer: "summary", metrics: [],
  }];
  assert.deepEqual(displayStepTimeline([delivered], true).map((item) => item.step_id), ["partial"]);
});

test("loose results keep the latest version of each saved product", () => {
  const card = (id: string, title: string, attemptId: string): ResultCardSummary => ({
    id, title, attemptId, type: "json", headline: title, description: "", renderer: "summary", metrics: [],
  });
  assert.deepEqual(displayLooseResults([
    card("map_old", "Event footprint", "attempt_1"),
    { ...card("owned", "Step-owned result", "attempt_1"), ownerStepId: "detect_1" },
    card("map_new", "Event footprint", "attempt_2"),
    card("stats", "Statistics", "attempt_2"),
  ], [step("detect_1", "Detect", "completed", "attempt_1")], true).map((item) => item.id), ["map_new", "stats"]);
});

test("a result remains visible when its owning step is hidden", () => {
  const card: ResultCardSummary = {
    id: "saved", ownerStepId: "failed_step", title: "Saved figure", type: "image_png",
    headline: "Saved figure", description: "", renderer: "summary", metrics: [],
  };
  assert.deepEqual(displayLooseResults([card], [], true).map((item) => item.id), ["saved"]);
});

test("a superseded step does not duplicate its result as a loose card", () => {
  const oldStep = step("old", "Detect events", "completed", "attempt_1");
  const card: ResultCardSummary = {
    id: "old_result", ownerStepId: "old", title: "Old event map", type: "json",
    headline: "Old event map", description: "", renderer: "event", metrics: [],
  };
  assert.deepEqual(displayLooseResults([card], [oldStep], true), []);
});
