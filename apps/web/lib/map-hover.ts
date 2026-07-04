import type { ManualViewState, MapFieldData } from "./types";

export type HoverSample = {
  lat: number;
  lon: number;
  value: number;
};

type SampleBounds = {
  latMin: number;
  latMax: number;
  lonMin: number;
  lonMax: number;
};

function nearestIndex(values: number[], target: number) {
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;

  values.forEach((value, index) => {
    const distance = Math.abs(value - target);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });

  return bestIndex;
}

function coordinateTolerance(values: number[]) {
  if (values.length < 2) {
    return 1e-9;
  }

  const spacings = values
    .slice(1)
    .map((value, index) => Math.abs(value - values[index]))
    .filter((spacing) => Number.isFinite(spacing) && spacing > 0);

  if (spacings.length === 0) {
    return 1e-9;
  }

  return Math.min(...spacings) / 2 + 1e-9;
}

function mapFieldBounds(mapField: MapFieldData): SampleBounds | null {
  if (!mapField.bounds) {
    return null;
  }

  const [[latA, lonA], [latB, lonB]] = mapField.bounds;
  return {
    latMin: Math.min(latA, latB),
    latMax: Math.max(latA, latB),
    lonMin: Math.min(lonA, lonB),
    lonMax: Math.max(lonA, lonB),
  };
}

function sampleBoundsForField(
  mapField: MapFieldData,
  fallbackRegionBounds: ManualViewState["regionBounds"],
): SampleBounds {
  return mapFieldBounds(mapField) ?? fallbackRegionBounds;
}

export function sampleNearestMapFieldValue(
  mapField: MapFieldData | null | undefined,
  lat: number,
  lon: number,
  fallbackRegionBounds: ManualViewState["regionBounds"],
): HoverSample | null {
  if (!mapField || mapField.lat.length === 0 || mapField.lon.length === 0 || mapField.values.length === 0) {
    return null;
  }

  const bounds = sampleBoundsForField(mapField, fallbackRegionBounds);
  const latTolerance = coordinateTolerance(mapField.lat);
  const lonTolerance = coordinateTolerance(mapField.lon);
  const insideField =
    lon >= bounds.lonMin - lonTolerance &&
    lon <= bounds.lonMax + lonTolerance &&
    lat >= bounds.latMin - latTolerance &&
    lat <= bounds.latMax + latTolerance;
  if (!insideField) {
    return null;
  }

  const latIndex = nearestIndex(mapField.lat, lat);
  const lonIndex = nearestIndex(mapField.lon, lon);
  const row = mapField.values[latIndex];
  const value = row?.[lonIndex] as unknown;

  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }

  return {
    lat: mapField.lat[latIndex],
    lon: mapField.lon[lonIndex],
    value,
  };
}
