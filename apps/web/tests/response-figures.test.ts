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
  const text = "The hotspot expanded [Figure 1](#figure-artifact_123).\n\n" +
    "![Figure 1. Days with bottom hypoxia in 2020; the map locates the highest burden.](#figure-artifact_123)";
  const selected = selectResponseFigures(text, [card]);
  assert.equal(selected.figures[0]?.card.id, card.id);
  assert.match(selected.figures[0]?.caption ?? "", /bottom hypoxia/);
  assert.match(selected.prose, /\[Figure 1\]\(#figure-artifact_123\)/);
  assert.doesNotMatch(selected.prose, /!\[Figure 1/);
});

test("response does not embed an unknown or nonvisual artifact", () => {
  const text = "![Figure 1. Unsupported](#figure-artifact_unknown)";
  assert.equal(selectResponseFigures(text, []).prose, "Figure 1. Unsupported");
});
