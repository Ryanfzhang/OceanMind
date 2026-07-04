import { formatRegionLabel } from "./manual-state";
import type { DatasetInfo, GeoPoint, ManualViewState, RegionBounds } from "./types";

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function normalizeDepthLevels(datasetInfo?: DatasetInfo | null): number[] {
  if (!Array.isArray(datasetInfo?.depth_levels)) {
    return [];
  }
  return datasetInfo.depth_levels.filter(isFiniteNumber);
}

function normalizeVariables(datasetInfo?: DatasetInfo | null): string[] {
  if (!Array.isArray(datasetInfo?.variables)) {
    return [];
  }
  return datasetInfo.variables.filter(
    (value): value is string => typeof value === "string" && value.trim().length > 0,
  );
}

export function datasetHasStoreAvailability(datasetInfo?: DatasetInfo | null) {
  return Boolean(datasetInfo?.backend === "zarr" && datasetInfo.data_stores);
}

export function isDatasetVariableAvailable(datasetInfo: DatasetInfo | null | undefined, variable: string) {
  if (!variable.trim()) {
    return false;
  }
  if (!datasetHasStoreAvailability(datasetInfo)) {
    return true;
  }
  return datasetInfo?.data_stores?.[variable]?.exists === true;
}

export function availableDatasetVariables(datasetInfo?: DatasetInfo | null): string[] {
  const variables = normalizeVariables(datasetInfo);
  if (!datasetHasStoreAvailability(datasetInfo)) {
    return variables;
  }
  return variables.filter((variable) => isDatasetVariableAvailable(datasetInfo, variable));
}

function nearestDepth(depths: number[], target: number) {
  if (depths.length === 0) {
    return target;
  }
  let best = depths[0];
  let bestDistance = Math.abs(best - target);
  for (const depth of depths.slice(1)) {
    const distance = Math.abs(depth - target);
    if (distance < bestDistance) {
      best = depth;
      bestDistance = distance;
    }
  }
  return best;
}

function normalizeNumberPair(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length < 2 || !isFiniteNumber(value[0]) || !isFiniteNumber(value[1])) {
    return null;
  }
  return [Math.min(value[0], value[1]), Math.max(value[0], value[1])];
}

function normalizeSpatialExtent(datasetInfo?: DatasetInfo | null): RegionBounds | null {
  const spatialExtent =
    datasetInfo?.spatial_extent && typeof datasetInfo.spatial_extent === "object"
      ? (datasetInfo.spatial_extent as Record<string, unknown>)
      : null;
  const lon = normalizeNumberPair(spatialExtent?.lon);
  const lat = normalizeNumberPair(spatialExtent?.lat);
  if (!lon || !lat) {
    return null;
  }
  return {
    lonMin: lon[0],
    lonMax: lon[1],
    latMin: lat[0],
    latMax: lat[1],
  };
}

function normalizeIsoDay(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const match = value.match(/^\d{4}-\d{2}-\d{2}/);
  return match ? match[0] : null;
}

function pointInsideExtent(point: GeoPoint, extent: RegionBounds) {
  return point.lon >= extent.lonMin &&
    point.lon <= extent.lonMax &&
    point.lat >= extent.latMin &&
    point.lat <= extent.latMax;
}

function extentCenter(extent: RegionBounds): [number, number] {
  return [
    (extent.latMin + extent.latMax) / 2,
    (extent.lonMin + extent.lonMax) / 2,
  ];
}

export function hydrateManualViewStateFromDataset(
  state: ManualViewState,
  datasetInfo?: DatasetInfo | null,
): ManualViewState {
  if (!datasetInfo) {
    return state;
  }

  const availableDepths = normalizeDepthLevels(datasetInfo);
  const availableVariables = availableDatasetVariables(datasetInfo);
  const nextVariable = availableVariables.includes(state.variable) ? state.variable : (availableVariables[0] ?? "");

  const temporalExtent =
    datasetInfo.temporal_extent && typeof datasetInfo.temporal_extent === "object"
      ? (datasetInfo.temporal_extent as Record<string, unknown>)
      : null;
  const temporalStart =
    temporalExtent ? normalizeIsoDay(temporalExtent.start) : null;

  const nextTimeRange: [string, string] = temporalStart ? [temporalStart, temporalStart] : state.timeRange;

  const selectedDepth = availableDepths.length > 0 ? nearestDepth(availableDepths, state.depthRange[0]) : state.depthRange[0];
  const spatialExtent = normalizeSpatialExtent(datasetInfo);
  const nextRegionBounds = spatialExtent ?? state.regionBounds;
  const nextSelectedPoint = spatialExtent ? extentCenter(spatialExtent) : state.selectedPoint;
  const nextTransectPoints =
    spatialExtent && state.transectPoints.some((point) => !pointInsideExtent(point, spatialExtent))
      ? []
      : state.transectPoints;
  const nextPolygonPoints =
    spatialExtent && state.polygonPoints.some((point) => !pointInsideExtent(point, spatialExtent))
      ? []
      : state.polygonPoints;

  return {
    ...state,
    dataset: datasetInfo.name || state.dataset,
    variable: nextVariable,
    availableDepths,
    depthRange: availableDepths.length > 0 ? [selectedDepth, selectedDepth] : state.depthRange,
    timeRange: nextTimeRange,
    timeLabel: `${nextTimeRange[0]} ~ ${nextTimeRange[1]}`,
    regionBounds: nextRegionBounds,
    regionLabel: formatRegionLabel(nextRegionBounds),
    selectedPoint: nextSelectedPoint,
    transectPoints: nextTransectPoints,
    polygonPoints: nextPolygonPoints,
  };
}
