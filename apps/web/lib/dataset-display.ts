import type { DatasetInfo } from "./types";
import { availableDatasetVariables, datasetHasStoreAvailability } from "./dataset-state";

function formatNumericRange(values: unknown, suffix = ""): string | null {
  if (!Array.isArray(values) || values.length < 2) {
    return null;
  }
  const [start, end] = values;
  if (typeof start !== "number" || typeof end !== "number") {
    return null;
  }
  return `${start} to ${end}${suffix}`;
}

function formatCoveragePair(
  values: unknown,
  positiveSuffix: string,
  negativeSuffix: string,
): string | null {
  if (!Array.isArray(values) || values.length < 2) {
    return null;
  }
  const [start, end] = values;
  if (typeof start !== "number" || typeof end !== "number") {
    return null;
  }
  const startSuffix = start < 0 ? negativeSuffix : positiveSuffix;
  const endSuffix = end < 0 ? negativeSuffix : positiveSuffix;
  return `${Math.abs(start)}${startSuffix} to ${Math.abs(end)}${endSuffix}`;
}

export function formatDatasetVariables(datasetInfo?: DatasetInfo | null): string {
  const variables = availableDatasetVariables(datasetInfo);
  if (variables.length === 0) {
    return "Not specified";
  }
  if (!datasetHasStoreAvailability(datasetInfo)) {
    return variables.join(", ");
  }
  const total = datasetInfo?.variables?.length ?? variables.length;
  return `${variables.join(", ")} (${variables.length}/${total} ready)`;
}

export function formatSpatialCoverage(datasetInfo?: DatasetInfo | null): string {
  const spatialExtent = datasetInfo?.spatial_extent;
  if (!spatialExtent || typeof spatialExtent !== "object") {
    return "Not specified";
  }
  const lon = formatCoveragePair((spatialExtent as Record<string, unknown>).lon, "E", "W");
  const lat = formatCoveragePair((spatialExtent as Record<string, unknown>).lat, "N", "S");
  if (lon && lat) {
    return `${lon} · ${lat}`;
  }
  return lon ?? lat ?? "Not specified";
}

export function formatTemporalCoverage(datasetInfo?: DatasetInfo | null): string {
  const temporalExtent = datasetInfo?.temporal_extent;
  if (!temporalExtent || typeof temporalExtent !== "object") {
    return "Not specified";
  }
  const start = (temporalExtent as Record<string, unknown>).start;
  const end = (temporalExtent as Record<string, unknown>).end;
  if (typeof start === "string" && typeof end === "string") {
    return `${start} to ${end}`;
  }
  return "Not specified";
}

export function formatDepthCoverage(datasetInfo?: DatasetInfo | null): string {
  const explicitRange = formatNumericRange(datasetInfo?.depth_range, " m");
  if (explicitRange) {
    return explicitRange;
  }
  if (Array.isArray(datasetInfo?.depth_levels) && datasetInfo.depth_levels.length >= 2) {
    const first = datasetInfo.depth_levels[0];
    const last = datasetInfo.depth_levels[datasetInfo.depth_levels.length - 1];
    if (typeof first === "number" && typeof last === "number") {
      return `${first} to ${last} m`;
    }
  }
  return "Not specified";
}

export function formatResolution(datasetInfo?: DatasetInfo | null): string {
  const resolution = datasetInfo?.resolution;
  if (!resolution) {
    return "Not specified";
  }
  if (typeof resolution === "string") {
    return resolution;
  }
  if (typeof resolution !== "object") {
    return "Not specified";
  }
  const spatial = (resolution as Record<string, unknown>).spatial;
  const temporal = (resolution as Record<string, unknown>).temporal;
  if (typeof spatial === "string" && typeof temporal === "string") {
    return `${spatial}, ${temporal}`;
  }
  if (typeof spatial === "string") {
    return spatial;
  }
  if (typeof temporal === "string") {
    return temporal;
  }
  return "Not specified";
}
