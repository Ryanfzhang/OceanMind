import assert from "node:assert/strict";
import test from "node:test";

import { mapFieldRasterSize } from "../lib/map-field-preview";
import type { MapFieldData } from "../lib/types";

test("map raster retains every latitude row after Mercator projection", () => {
  const lat = Array.from({ length: 601 }, (_, index) => index * 52 / 600);
  const lon = Array.from({ length: 517 }, (_, index) => 100 + index * 50 / 516);
  const field: MapFieldData = { lat, lon, values: [], label: "Depth", variable: "depth" };

  const size = mapFieldRasterSize(field);
  assert.ok(size.width >= lon.length);
  assert.ok(size.height > lat.length);
});
