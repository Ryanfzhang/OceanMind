import assert from "node:assert/strict";
import test from "node:test";

import { selectResponseFigures } from "../lib/response-figures";
import type { ResultCardSummary } from "../lib/types";

test("response embeds an LLM-selected interactive card and keeps its citation", () => {
  const card: ResultCardSummary = {
    id: "artifact_123", title: "Hypoxic Event Days By Year", type: "dataarray_netcdf",
    headline: "Hypoxic Event Days By Year", description: "", renderer: "summary", metrics: [],
    workspaceData: { mapField: {
      lat: [32, 33], lon: [120, 121], values: [[1, 2], [3, 4]],
      label: "2020 hypoxic days", variable: "hypoxic_days",
    } },
  };
  const unusedCard = { ...card, id: "artifact_456", title: "Other result" };
  const text = "The hotspot expanded [(Fig. 1)](#figure-artifact_123).\n\n" +
    "![Fig. 1. Bottom hypoxia in 2020](#figure-artifact_123)";
  const selected = selectResponseFigures(text, [card, unusedCard]);
  assert.equal(selected.figures.length, 1);
  assert.equal(selected.figures[0]?.card.id, card.id);
  assert.match(selected.figures[0]?.caption ?? "", /bottom hypoxia/i);
  assert.match(selected.prose, /\[\(Fig\. 1\)\]\(#figure-artifact_123\)/);
  assert.doesNotMatch(selected.prose, /!\[Fig\. 1/);
});

test("response does not embed an unknown or nonvisual artifact", () => {
  const text = "![Figure 1. Unsupported](#figure-artifact_unknown)";
  assert.equal(selectResponseFigures(text, []).prose, "Figure 1. Unsupported");
});
