import assert from "node:assert/strict";
import test from "node:test";

import { mapFieldRasterSize } from "../lib/map-field-preview";
import { sampleNearestMapFieldValue } from "../lib/map-hover";
import type { MapFieldData } from "../lib/types";

test("map raster retains every latitude row after Mercator projection", () => {
  const lat = Array.from({ length: 601 }, (_, index) => index * 52 / 600);
  const lon = Array.from({ length: 517 }, (_, index) => 100 + index * 50 / 516);
  const field: MapFieldData = { lat, lon, values: [], label: "Depth", variable: "depth" };

  const size = mapFieldRasterSize(field);
  assert.ok(size.width >= lon.length);
  assert.ok(size.height > lat.length);
});

test("map hover does not report the int64 missing sentinel as a measurement", () => {
  const field: MapFieldData = {
    lat: [32, 33], lon: [120, 121],
    values: [[-9.223372036854776e18, 1], [2, 3]],
    label: "Hypoxic days", variable: "hypoxic_days",
  };
  const bounds = { latMin: 32, latMax: 33, lonMin: 120, lonMax: 121 };
  assert.equal(sampleNearestMapFieldValue(field, 32, 120, bounds), null);
  assert.equal(sampleNearestMapFieldValue(field, 32, 121, bounds)?.value, 1);
});
