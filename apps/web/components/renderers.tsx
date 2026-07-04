"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { colormapRGB, useMapFieldImageUrl } from "../lib/map-field-preview";
import { MapColorbar } from "./map-colorbar";
import {
  gridCellBadge,
  gridCellDisplayName,
  gridCellStrokeColor,
  gridCellSupportText,
  gridCellTint,
  sortSubregionCells,
  subregionOverlayBox,
} from "../lib/subregion-grid";
import type {
  AnalysisSeries,
  CompositeFieldPreview,
  EventOverlay,
  EofModePreview,
  EofPoint,
  HistogramBin,
  MapFieldData,
  ProfilePoint,
  ResultCardSummary,
  TsDiagramPoint,
  TsDiagramWatermassBin,
  WorkspaceData
} from "../lib/types";

type LineSeriesGroup = {
  id: string;
  label: string;
  color: string;
  fill?: string;
  dashed?: boolean;
  data: AnalysisSeries[];
};

type SubregionCell = NonNullable<MapFieldData["subregionGrid"]>["cells"][number];

const CHART_FRAME = {
  width: 640,
  height: 280,
  paddingTop: 20,
  paddingRight: 18,
  paddingBottom: 38,
  paddingLeft: 72
};

const DETAIL_FRAME = {
  width: 640,
  height: 320,
  paddingTop: 24,
  paddingRight: 22,
  paddingBottom: 42,
  paddingLeft: 56
};

function EmptyState({ message }: { message: string }) {
  return <div className="empty-state">{message}</div>;
}

const NO_FINITE_TIMESERIES_MESSAGE = "No finite time-series values were available for this selection.";

function hasNoFiniteTimeseriesValues(card: ResultCardSummary, data: WorkspaceData) {
  return card.renderer === "timeseries" && (card.hasFiniteValues === false || data.timeseriesDisplayInfo?.hasFiniteValues === false);
}

function formatValue(value: number) {
  if (!Number.isFinite(value)) {
    return "NaN";
  }
  if (Math.abs(value) >= 100 || Math.abs(value) < 0.01) {
    return value.toExponential(2);
  }
  return value.toFixed(3).replace(/\.?0+$/, "");
}

function formatCompactTimeLabel(label: string) {
  const text = label.trim();
  if (!text) return label;
  const epochLabel = formatEpochLikeTimeLabel(text);
  if (epochLabel) return epochLabel;
  const datePart = text.includes("T") ? text.split("T", 1)[0] : text;
  const isoMatch = /^(\d{4})-(\d{2})-(\d{2})$/.exec(datePart);
  if (isoMatch) {
    const [, year, month, day] = isoMatch;
    return `${month}-${day}`;
  }
  return datePart;
}

function formatEpochLikeTimeLabel(text: string) {
  if (!/^-?\d+(\.\d+)?$/.test(text)) {
    return null;
  }
  const raw = Number(text);
  if (!Number.isFinite(raw)) {
    return null;
  }
  const absRaw = Math.abs(raw);
  let milliseconds: number | null = null;
  if (absRaw >= 100_000_000_000_000_000) {
    milliseconds = raw / 1_000_000;
  } else if (absRaw >= 100_000_000_000_000) {
    milliseconds = raw / 1_000;
  } else if (absRaw >= 100_000_000_000) {
    milliseconds = raw;
  } else if (absRaw >= 1_000_000_000) {
    milliseconds = raw * 1_000;
  }
  if (milliseconds === null) {
    return null;
  }

  const date = new Date(milliseconds);
  if (!Number.isFinite(date.getTime())) {
    return null;
  }
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  return `${month}-${day}`;
}

function sameSeries(a: AnalysisSeries[], b: AnalysisSeries[]) {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((point, index) => point.label === b[index]?.label && point.value === b[index]?.value);
}

function sampleIndices(length: number, count = 5) {
  if (length <= count) {
    return Array.from({ length }, (_, index) => index);
  }

  const indices = new Set<number>([0, length - 1]);
  for (let step = 1; step < count - 1; step += 1) {
    indices.add(Math.round((step / (count - 1)) * (length - 1)));
  }
  return Array.from(indices).sort((a, b) => a - b);
}

function finiteExtentFromValues(values: Iterable<number>) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let count = 0;

  for (const value of values) {
    if (!Number.isFinite(value)) {
      continue;
    }
    min = Math.min(min, value);
    max = Math.max(max, value);
    count += 1;
  }

  return count > 0 ? { min, max, count } : null;
}

function finiteExtentFromRows(rows: { values?: number[] }[]) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let count = 0;

  rows.forEach((row) => {
    if (!Array.isArray(row.values)) {
      return;
    }
    row.values.forEach((value) => {
      if (!Number.isFinite(value)) {
        return;
      }
      min = Math.min(min, value);
      max = Math.max(max, value);
      count += 1;
    });
  });

  return count > 0 ? { min, max, count } : null;
}

function finiteValuesFromRows(rows: { values?: Array<number | null> }[]) {
  const values: number[] = [];
  rows.forEach((row) => {
    if (!Array.isArray(row.values)) {
      return;
    }
    row.values.forEach((value) => {
      if (typeof value === "number" && Number.isFinite(value)) {
        values.push(value);
      }
    });
  });
  return values;
}

function quantileSorted(values: number[], q: number) {
  if (values.length === 0) {
    return Number.NaN;
  }
  if (values.length === 1) {
    return values[0];
  }
  const position = clamp(q, 0, 1) * (values.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) {
    return values[lower];
  }
  const factor = position - lower;
  return values[lower] + (values[upper] - values[lower]) * factor;
}

const HOVMOLLER_COLORMAP_STOPS: [number, [number, number, number]][] = [
  [0.0, [32, 85, 191]],
  [0.25, [112, 166, 228]],
  [0.5, [255, 255, 255]],
  [0.75, [248, 154, 146]],
  [1.0, [213, 36, 42]],
];
const HOVMOLLER_CONTOUR_LABEL_MARGIN_X = 42;
const HOVMOLLER_CONTOUR_LABEL_MARGIN_Y = 12;

function interpolateStops(stops: [number, [number, number, number]][], t: number): [number, number, number] {
  const clamped = clamp(t, 0, 1);
  for (let index = 0; index < stops.length - 1; index += 1) {
    const [t0, rgb0] = stops[index];
    const [t1, rgb1] = stops[index + 1];
    if (clamped <= t1) {
      const factor = (clamped - t0) / Math.max(t1 - t0, 1e-9);
      return [
        Math.round(rgb0[0] + (rgb1[0] - rgb0[0]) * factor),
        Math.round(rgb0[1] + (rgb1[1] - rgb0[1]) * factor),
        Math.round(rgb0[2] + (rgb1[2] - rgb0[2]) * factor),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

function hovmollerColorRGB(value: number, limit: number): [number, number, number] {
  if (!Number.isFinite(value)) {
    return [226, 232, 240];
  }
  const normalized = (clamp(value, -limit, limit) + limit) / Math.max(2 * limit, 1e-12);
  return interpolateStops(HOVMOLLER_COLORMAP_STOPS, normalized);
}

function robustSymmetricLimit(values: number[]) {
  if (values.length === 0) {
    return 1;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const q02 = quantileSorted(sorted, 0.02);
  const q98 = quantileSorted(sorted, 0.98);
  const limit = Math.max(Math.abs(q02), Math.abs(q98), 1e-12);
  return Number.isFinite(limit) ? limit : 1;
}

function safeMatrixValue(rows: { values?: Array<number | null> }[], row: number, col: number) {
  const value = rows[row]?.values?.[col];
  return typeof value === "number" && Number.isFinite(value) ? value : Number.NaN;
}

function hovmollerRowDepth(row: { depthValue?: number | null; depthLabel?: string }) {
  const raw = typeof row.depthValue === "number" ? row.depthValue : Number.parseFloat(row.depthLabel ?? "");
  return Number.isFinite(raw) ? Math.abs(raw) : Number.NaN;
}

function hovmollerRowWithConfiguredDepth(
  row: WorkspaceData["hovmollerRows"][number],
  index: number,
  configuredDepthLevels?: number[],
) {
  const configuredDepth = configuredDepthLevels?.[index];
  if (typeof configuredDepth !== "number" || !Number.isFinite(configuredDepth)) {
    return row;
  }
  const rawDepth = typeof row.depthValue === "number" ? row.depthValue : Number.parseFloat(row.depthLabel ?? "");
  const looksLikeIndex =
    Number.isFinite(rawDepth) &&
    Math.abs(rawDepth - index) < 1e-9 &&
    Math.abs(rawDepth - configuredDepth) > 1e-9;
  if (Number.isFinite(rawDepth) && !looksLikeIndex) {
    return row;
  }
  return {
    ...row,
    depthValue: configuredDepth,
    depthLabel: `${configuredDepth} m`,
  };
}

function hovmollerRowYPositions(rows: WorkspaceData["hovmollerRows"], plotHeight: number) {
  const depths = rows.map(hovmollerRowDepth);
  const finiteDepths = depths.filter((depth) => Number.isFinite(depth));
  if (finiteDepths.length === 0) {
    return rows.map((_, index) => ((index + 0.5) / Math.max(rows.length, 1)) * plotHeight);
  }

  const top = Math.min(...finiteDepths);
  const bottom = Math.max(...finiteDepths);
  if (bottom <= top) {
    return rows.map(() => plotHeight / 2);
  }

  return depths.map((depth, index) =>
    Number.isFinite(depth)
      ? ((depth - top) / (bottom - top)) * plotHeight
      : ((index + 0.5) / Math.max(rows.length, 1)) * plotHeight
  );
}

function depthBracket(depths: number[], targetDepth: number) {
  if (depths.length === 0) {
    return { lower: 0, upper: 0, fraction: 0 };
  }
  if (depths.length === 1 || !Number.isFinite(targetDepth)) {
    return { lower: 0, upper: 0, fraction: 0 };
  }
  if (targetDepth <= depths[0]) {
    return { lower: 0, upper: 0, fraction: 0 };
  }
  const lastIndex = depths.length - 1;
  if (targetDepth >= depths[lastIndex]) {
    return { lower: lastIndex, upper: lastIndex, fraction: 0 };
  }

  for (let index = 0; index < lastIndex; index += 1) {
    const shallow = depths[index];
    const deep = depths[index + 1];
    if (targetDepth >= shallow && targetDepth <= deep) {
      const span = Math.max(deep - shallow, 1e-12);
      return { lower: index, upper: index + 1, fraction: clamp((targetDepth - shallow) / span, 0, 1) };
    }
  }

  return { lower: lastIndex, upper: lastIndex, fraction: 0 };
}

function nearestDepthIndex(depths: number[], targetDepth: number) {
  if (depths.length === 0 || !Number.isFinite(targetDepth)) {
    return 0;
  }
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  depths.forEach((depth, index) => {
    const distance = Math.abs(depth - targetDepth);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function interpolateHovmollerValue(
  rows: WorkspaceData["hovmollerRows"],
  depths: number[],
  targetDepth: number,
  fractionalColumn: number,
  nColumns: number,
) {
  if (rows.length === 0 || depths.length === 0 || nColumns === 0) {
    return Number.NaN;
  }
  const gc = fractionalColumn - 0.5;
  const c0 = clamp(Math.floor(gc), 0, nColumns - 1);
  const c1 = clamp(c0 + 1, 0, nColumns - 1);
  const dc = clamp(gc - c0, 0, 1);
  const { lower, upper, fraction } = depthBracket(depths, targetDepth);

  return interpolateFinite([
    { value: safeMatrixValue(rows, lower, c0), weight: (1 - fraction) * (1 - dc) },
    { value: safeMatrixValue(rows, lower, c1), weight: (1 - fraction) * dc },
    { value: safeMatrixValue(rows, upper, c0), weight: fraction * (1 - dc) },
    { value: safeMatrixValue(rows, upper, c1), weight: fraction * dc },
  ]);
}

function rowHasFiniteValue(row: { values?: Array<number | null> }) {
  return Array.isArray(row.values) && row.values.some((value) => typeof value === "number" && Number.isFinite(value));
}

function interpolateFinite(values: Array<{ value: number; weight: number }>) {
  let weighted = 0;
  let weightSum = 0;
  values.forEach(({ value, weight }) => {
    if (!Number.isFinite(value) || weight <= 0) {
      return;
    }
    weighted += value * weight;
    weightSum += weight;
  });
  return weightSum > 0 ? weighted / weightSum : Number.NaN;
}

function buildHovmollerContourLevels(limit: number) {
  if (!Number.isFinite(limit) || limit <= 0) {
    return [];
  }
  return [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75].map((factor) => factor * limit);
}

function contourEdgePoint(
  a: { x: number; y: number; value: number },
  b: { x: number; y: number; value: number },
  level: number,
) {
  const delta = b.value - a.value;
  if (!Number.isFinite(a.value) || !Number.isFinite(b.value) || Math.abs(delta) < 1e-12) {
    return null;
  }
  const t = (level - a.value) / delta;
  if (t < 0 || t > 1) {
    return null;
  }
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
  };
}

function forEachHovmollerContourSegment(
  rows: WorkspaceData["hovmollerRows"],
  nColumns: number,
  plotWidth: number,
  plotHeight: number,
  levels: number[],
  visit: (level: number, start: { x: number; y: number }, end: { x: number; y: number }) => void,
) {
  const nRows = rows.length;
  if (nRows < 2 || nColumns < 2) {
    return;
  }
  const cellWidth = plotWidth / nColumns;
  const rowY = hovmollerRowYPositions(rows, plotHeight);
  for (let row = 0; row < nRows - 1; row += 1) {
    for (let col = 0; col < nColumns - 1; col += 1) {
      const corners = [
        { x: (col + 0.5) * cellWidth, y: rowY[row], value: safeMatrixValue(rows, row, col) },
        { x: (col + 1.5) * cellWidth, y: rowY[row], value: safeMatrixValue(rows, row, col + 1) },
        { x: (col + 1.5) * cellWidth, y: rowY[row + 1], value: safeMatrixValue(rows, row + 1, col + 1) },
        { x: (col + 0.5) * cellWidth, y: rowY[row + 1], value: safeMatrixValue(rows, row + 1, col) },
      ];
      if (!corners.every((corner) => Number.isFinite(corner.value))) {
        continue;
      }
      levels.forEach((level) => {
        const crossings = [
          contourEdgePoint(corners[0], corners[1], level),
          contourEdgePoint(corners[1], corners[2], level),
          contourEdgePoint(corners[2], corners[3], level),
          contourEdgePoint(corners[3], corners[0], level),
        ].filter((point): point is { x: number; y: number } => Boolean(point));
        if (crossings.length === 2) {
          visit(level, crossings[0], crossings[1]);
        } else if (crossings.length === 4) {
          visit(level, crossings[0], crossings[1]);
          visit(level, crossings[2], crossings[3]);
        }
      });
    }
  }
}

function drawHovmollerContours(
  context: CanvasRenderingContext2D,
  rows: WorkspaceData["hovmollerRows"],
  nColumns: number,
  plotWidth: number,
  plotHeight: number,
  levels: number[],
) {
  levels.forEach((level) => {
    context.beginPath();
    context.strokeStyle = Math.abs(level) < 1e-12 ? "rgba(15, 23, 42, 0.9)" : "rgba(15, 23, 42, 0.72)";
    context.lineWidth = Math.abs(level) < 1e-12 ? 1.25 : 0.9;
    let hasSegment = false;
    forEachHovmollerContourSegment(rows, nColumns, plotWidth, plotHeight, [level], (_level, start, end) => {
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      hasSegment = true;
    });
    if (hasSegment) {
      context.stroke();
    }
  });
}

function buildHovmollerContourLabels(
  rows: WorkspaceData["hovmollerRows"],
  nColumns: number,
  plotWidth: number,
  plotHeight: number,
  levels: number[],
) {
  const labels: Array<{ level: number; x: number; y: number; text: string }> = [];
  const centerX = plotWidth / 2;
  const centerY = plotHeight / 2;
  levels.forEach((targetLevel) => {
    const bestRef: { value: { x: number; y: number; score: number } | null } = { value: null };
    forEachHovmollerContourSegment(rows, nColumns, plotWidth, plotHeight, [targetLevel], (_level, start, end) => {
      const x = (start.x + end.x) / 2;
      const y = (start.y + end.y) / 2;
      const score = Math.hypot(x - centerX, y - centerY);
      if (!bestRef.value || score < bestRef.value.score) {
        bestRef.value = { x, y, score };
      }
    });
    const best = bestRef.value;
    if (best) {
      labels.push({
        level: targetLevel,
        x: clamp(best.x, HOVMOLLER_CONTOUR_LABEL_MARGIN_X, Math.max(HOVMOLLER_CONTOUR_LABEL_MARGIN_X, plotWidth - HOVMOLLER_CONTOUR_LABEL_MARGIN_X)),
        y: clamp(best.y, HOVMOLLER_CONTOUR_LABEL_MARGIN_Y, Math.max(HOVMOLLER_CONTOUR_LABEL_MARGIN_Y, plotHeight - HOVMOLLER_CONTOUR_LABEL_MARGIN_Y)),
        text: formatValue(targetLevel),
      });
    }
  });
  return labels;
}

function formatDepthTickMeters(value: number) {
  if (!Number.isFinite(value)) {
    return "";
  }
  const depth = Math.abs(value);
  return `${Math.round(depth)}`;
}

export const __HOVMOLLER_TEST_UTILS = {
  buildHovmollerContourLabels,
  hovmollerColorRGB,
  hovmollerRowDepth,
  hovmollerRowYPositions,
  interpolateHovmollerValue,
};

function buildPath(points: string[]) {
  return points.join(" ");
}

/**
 * Convert an array of [x,y] coordinate pairs into a smooth cubic-bezier SVG path
 * using Catmull-Rom to Bezier conversion for natural-looking curves.
 */
function buildSmoothPath(coords: [number, number][]): string {
  if (coords.length < 2) return "";
  if (coords.length === 2) {
    return `M ${coords[0][0]},${coords[0][1]} L ${coords[1][0]},${coords[1][1]}`;
  }

  let d = `M ${coords[0][0]},${coords[0][1]}`;
  for (let i = 0; i < coords.length - 1; i++) {
    const p0 = coords[Math.max(i - 1, 0)];
    const p1 = coords[i];
    const p2 = coords[i + 1];
    const p3 = coords[Math.min(i + 2, coords.length - 1)];

    const tension = 6;
    const cp1x = p1[0] + (p2[0] - p0[0]) / tension;
    const cp1y = p1[1] + (p2[1] - p0[1]) / tension;
    const cp2x = p2[0] - (p3[0] - p1[0]) / tension;
    const cp2y = p2[1] - (p3[1] - p1[1]) / tension;

    d += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${p2[0]},${p2[1]}`;
  }
  return d;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function blendColor(hexA: string, hexB: string, factor: number) {
  const normalized = clamp(factor, 0, 1);
  const a = hexA.replace("#", "");
  const b = hexB.replace("#", "");
  const aRgb = [0, 2, 4].map((index) => Number.parseInt(a.slice(index, index + 2), 16));
  const bRgb = [0, 2, 4].map((index) => Number.parseInt(b.slice(index, index + 2), 16));
  const mixed = aRgb.map((value, index) => Math.round(value + (bRgb[index] - value) * normalized));
  return `rgb(${mixed[0]}, ${mixed[1]}, ${mixed[2]})`;
}

function formatMapBounds(field: MapFieldData) {
  if (field.lon.length === 0 || field.lat.length === 0) {
    return null;
  }
  return `${formatValue(field.lon[0])}–${formatValue(field.lon[field.lon.length - 1])}E · ${formatValue(field.lat[0])}–${formatValue(field.lat[field.lat.length - 1])}N`;
}

function subregionStatusText(status: string) {
  switch (String(status).trim().toLowerCase()) {
    case "ok":
      return "Valid";
    case "skipped_no_valid_ocean":
      return "No valid ocean";
    case "skipped_no_valid_samples":
      return "No valid samples";
    default:
      return status || "Skipped";
  }
}

function subregionScoreValue(cell: SubregionCell) {
  if (typeof cell.value === "number" && Number.isFinite(cell.value)) {
    return cell.value;
  }
  if (typeof cell.dominantScore === "number" && Number.isFinite(cell.dominantScore)) {
    return cell.dominantScore;
  }
  return Number.NEGATIVE_INFINITY;
}

function buildSubregionCategoryCounts(cells: SubregionCell[]) {
  const counts = new Map<
    string,
    {
      key: string;
      label: string;
      badge: string;
      color: string;
      count: number;
    }
  >();

  cells.forEach((cell) => {
    if (cell.status !== "ok") {
      return;
    }
    const key = cell.category ?? gridCellDisplayName(cell);
    const current = counts.get(key);
    if (current) {
      current.count += 1;
      return;
    }
    counts.set(key, {
      key,
      label: gridCellDisplayName(cell),
      badge: gridCellBadge(cell),
      color: gridCellStrokeColor(cell),
      count: 1,
    });
  });

  return Array.from(counts.values()).sort((left, right) => {
    if (left.count !== right.count) {
      return right.count - left.count;
    }
    return left.label.localeCompare(right.label);
  });
}

function SubregionGridOverlay({ field }: { field: MapFieldData }) {
  const subregionGrid = field.subregionGrid;
  if (!subregionGrid || subregionGrid.cells.length === 0) {
    return null;
  }
  const showLabels = subregionGrid.cells.length <= 25;

  return (
    <div className="subregion-overlay-layer" aria-hidden="true">
      {sortSubregionCells(subregionGrid.cells).map((cell) => {
        const box = subregionOverlayBox(field, cell);
        if (!box) {
          return null;
        }
        const stroke = gridCellStrokeColor(cell);
        const score =
          typeof cell.value === "number"
            ? formatValue(cell.value)
            : typeof cell.dominantScore === "number"
              ? formatValue(cell.dominantScore)
              : "NA";
        const label = cell.shortLabel ?? cell.label;
        return (
          <div
            key={cell.subregionId}
            className={`subregion-overlay-box ${cell.status !== "ok" ? "is-muted" : ""} ${showLabels ? "" : "is-dense"}`}
            style={{
              left: `${box.left}%`,
              top: `${box.top}%`,
              width: `${box.width}%`,
              height: `${box.height}%`,
              borderColor: stroke,
              background: showLabels ? gridCellTint(cell) : "transparent",
              borderStyle: cell.status === "ok" ? "solid" : "dashed",
            }}
            title={`${label}: ${gridCellDisplayName(cell)} (${score})`}
          >
            {showLabels ? (
              <>
                <span>{label}</span>
                <strong>{gridCellBadge(cell)}</strong>
              </>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function SubregionDiagnosisPanel({
  field,
  compact = false,
}: {
  field: MapFieldData;
  compact?: boolean;
}) {
  const subregionGrid = field.subregionGrid;
  const cells = useMemo(() => (subregionGrid ? sortSubregionCells(subregionGrid.cells) : []), [subregionGrid]);
  const [expanded, setExpanded] = useState(false);
  if (!subregionGrid || cells.length === 0) {
    return null;
  }

  const validCells = cells.filter((cell) => cell.status === "ok");
  const summaryCounts = buildSubregionCategoryCounts(cells);
  const visibleCount = compact ? 8 : 12;
  const rankedCells = useMemo(
    () =>
      [...cells].sort((left, right) => {
        const leftOk = left.status === "ok" ? 0 : 1;
        const rightOk = right.status === "ok" ? 0 : 1;
        if (leftOk !== rightOk) {
          return leftOk - rightOk;
        }
        const scoreDelta = subregionScoreValue(right) - subregionScoreValue(left);
        if (Number.isFinite(scoreDelta) && scoreDelta !== 0) {
          return scoreDelta;
        }
        return String(left.shortLabel ?? left.label).localeCompare(String(right.shortLabel ?? right.label));
      }),
    [cells],
  );
  const visibleCells = expanded ? rankedCells : rankedCells.slice(0, visibleCount);

  return (
    <div className={`subregion-diagnosis-panel ${compact ? "is-compact" : ""}`}>
      <div className="subregion-diagnosis-header">
        <span className="mini-label">Subregion Diagnosis</span>
        <strong>{subregionGrid.gridShape[0]}x{subregionGrid.gridShape[1]} tiles</strong>
      </div>
      <div className="subregion-diagnosis-summary">
        <span>{subregionGrid.validCount ?? validCells.length} valid</span>
        <span>{subregionGrid.skippedCount ?? cells.length - validCells.length} skipped</span>
        <span>{summaryCounts.length} dominant classes</span>
      </div>
      {summaryCounts.length > 0 ? (
        <div className="subregion-diagnosis-counts">
          {summaryCounts.map((item) => (
            <div key={item.key} className="subregion-diagnosis-count-pill">
              <span className="subregion-diagnosis-count-dot" style={{ background: item.color }} />
              <strong>{item.badge}</strong>
              <span>{item.label}</span>
              <em>{item.count}</em>
            </div>
          ))}
        </div>
      ) : null}
      <div className="subregion-diagnosis-list">
        {visibleCells.map((cell) => (
          <div
            key={cell.subregionId}
            className={`subregion-diagnosis-row ${cell.status !== "ok" ? "is-muted" : ""}`}
            style={{
              borderColor: gridCellStrokeColor(cell),
              background: gridCellTint(cell),
            }}
          >
            <div className="subregion-diagnosis-row-id">
              <strong>{cell.shortLabel ?? cell.label}</strong>
              <span>{gridCellBadge(cell)}</span>
            </div>
            <div className="subregion-diagnosis-row-main">
              <strong>{gridCellDisplayName(cell)}</strong>
              <span>{cell.status === "ok" ? gridCellSupportText(cell) : subregionStatusText(cell.status)}</span>
            </div>
            <div className="subregion-diagnosis-row-score">
              {typeof cell.value === "number"
                ? formatValue(cell.value)
                : typeof cell.dominantScore === "number"
                  ? formatValue(cell.dominantScore)
                  : "NA"}
            </div>
          </div>
        ))}
      </div>
      {rankedCells.length > visibleCount ? (
        <div className="subregion-diagnosis-actions">
          <button className="result-expand-btn" onClick={() => setExpanded((current) => !current)} type="button">
            {expanded ? "Show fewer tiles" : `Show all ${rankedCells.length} tiles`}
          </button>
        </div>
      ) : null}
    </div>
  );
}

function SpatialFieldPreview({
  field,
  compact = false,
  onShowOnMap,
}: {
  field: MapFieldData;
  compact?: boolean;
  onShowOnMap?: (field: MapFieldData) => void;
}) {
  const imageUrl = useMapFieldImageUrl(field, compact ? 280 : 420, compact ? 180 : 240);
  const stats = field.statistics ?? {};
  const footerItems = [
    typeof stats.mean === "number" ? `Mean ${formatValue(stats.mean)}` : null,
    typeof stats.min === "number" ? `Min ${formatValue(stats.min)}` : null,
    typeof stats.max === "number" ? `Max ${formatValue(stats.max)}` : null,
    field.timeLabel ? field.timeLabel : null,
    field.depthLabel ? field.depthLabel : null,
    formatMapBounds(field),
  ].filter((item): item is string => Boolean(item));
  const discreteLegend = Array.isArray(field.discreteLegend) ? field.discreteLegend : [];

  return (
    <div className={`map-preview-card ${compact ? "is-compact" : ""}`}>
      <div className="map-preview-header">
        <div>
          <p className="eyebrow">{field.subregionGrid ? "Subregion Diagnosis" : "Spatial Field"}</p>
          <h4>{field.label}</h4>
        </div>
        {field.units ? <span className="map-preview-unit">{field.units}</span> : null}
      </div>
      <div className="map-preview-visual">
        <div className="map-preview-visual-frame">
          {imageUrl ? (
            <img alt={field.label} src={imageUrl} />
          ) : (
            <div className="map-preview-empty">
              <EmptyState message="Map preview is not available for this field yet." />
            </div>
          )}
          <SubregionGridOverlay field={field} />
        </div>
      </div>
      <MapColorbar field={field} compact={compact} />
      <SubregionDiagnosisPanel field={field} compact={compact} />
      {discreteLegend.length > 0 ? (
        <div className="map-preview-meta">
          {discreteLegend.map((item) => (
            <span key={`${item.value}-${item.label}`} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: 999,
                  background: item.color,
                  border: "1px solid rgba(15, 23, 42, 0.14)",
                  display: "inline-block",
                }}
              />
              {item.label}
            </span>
          ))}
        </div>
      ) : null}
      {footerItems.length > 0 ? (
        <div className="map-preview-meta">
          {footerItems.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      ) : null}
      {onShowOnMap ? (
        <div className="map-preview-actions">
          <button className="result-expand-btn" onClick={() => onShowOnMap(field)} type="button">
            Show on main map
          </button>
        </div>
      ) : null}
    </div>
  );
}

function CompositeFieldPanel({
  fields,
  compact = false,
  onShowOnMap,
}: {
  fields: CompositeFieldPreview[];
  compact?: boolean;
  onShowOnMap?: (field: MapFieldData) => void;
}) {
  if (fields.length === 0) {
    return <EmptyState message="No composite maps are attached to this result yet." />;
  }

  return (
    <div className={`composite-field-panel ${compact ? "is-compact" : ""}`}>
      {!compact ? (
        <div className="detail-chart-header">
          <div>
            <p className="eyebrow">Composite Fields</p>
            <h4>Positive, negative, and difference maps</h4>
          </div>
          <div className="chart-readout">
            <strong>{fields.length} spatial fields</strong>
            <div className="chart-readout-values">
              <span>The main map shows the difference field by default.</span>
            </div>
          </div>
        </div>
      ) : null}
      <div className="composite-field-grid">
        {fields.map((item) => (
          <SpatialFieldPreview
            key={item.id}
            field={item.mapField}
            compact={compact}
            onShowOnMap={onShowOnMap}
          />
        ))}
      </div>
    </div>
  );
}

function buildLineSeries(workspaceData: WorkspaceData): LineSeriesGroup[] {
  const groups: LineSeriesGroup[] = [];
  const labels = workspaceData.seriesLabels ?? {};

  if (workspaceData.referenceSeries.length > 0) {
    groups.push({
      id: "reference",
      label: labels.reference ?? "Reference",
      color: "#7ea6d6",
      dashed: true,
      data: workspaceData.referenceSeries
    });
  }

  if (workspaceData.resultSeries.length > 0) {
    groups.push({
      id: "result",
      label: labels.result ?? "Result",
      color: "#4f7df3",
      fill: "rgba(79, 125, 243, 0.12)",
      data: workspaceData.resultSeries
    });
  }

  if (
    workspaceData.anomalySeries.length > 0 &&
    !sameSeries(workspaceData.anomalySeries, workspaceData.resultSeries) &&
    !sameSeries(workspaceData.anomalySeries, workspaceData.referenceSeries)
  ) {
    groups.push({
      id: "compare",
      label: labels.compare ?? "Compare",
      color: "#34c88a",
      data: workspaceData.anomalySeries
    });
  }

  return groups.filter((group) => group.data.length > 0);
}

function timeseriesDisplaySubtitle(displayInfo?: WorkspaceData["timeseriesDisplayInfo"]): string | undefined {
  if (!displayInfo?.aggregation || displayInfo.aggregation === "none") {
    return undefined;
  }
  return `${displayInfo.aggregationLabel} (${displayInfo.originalPoints} to ${displayInfo.displayPoints} points)`;
}

function InteractiveLineChart({
  emptyMessage,
  groups,
  subtitle,
  title
}: {
  emptyMessage: string;
  groups: LineSeriesGroup[];
  subtitle?: string;
  title: string;
}) {
  const chartRef = useRef<HTMLDivElement | null>(null);
  const idsKey = useMemo(() => groups.map((group) => group.id).join("|"), [groups]);
  const [visibleIds, setVisibleIds] = useState<string[]>(groups.map((group) => group.id));
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  useEffect(() => {
    setVisibleIds(groups.map((group) => group.id));
    setHoverIndex(null);
  }, [idsKey, groups]);

  const baseGroup = groups.find((group) => group.id === "result") ?? groups[0];
  const labels = baseGroup?.data.map((point) => point.label) ?? [];
  const visibleGroups = groups.filter((group) => visibleIds.includes(group.id));
  const activeGroups = visibleGroups.length > 0 ? visibleGroups : (baseGroup ? [baseGroup] : []);
  const valueExtent = finiteExtentFromValues(activeGroups.flatMap((group) => group.data.map((point) => point.value)));

  if (!baseGroup || labels.length === 0 || !valueExtent) {
    return <EmptyState message={emptyMessage} />;
  }

  const minValue = valueExtent.min;
  const maxValue = valueExtent.max;
  const valueSpan = maxValue - minValue;
  const yPadding = valueSpan === 0 ? Math.max(Math.abs(maxValue) * 0.15, 1) : valueSpan * 0.14;
  const yMin = minValue - yPadding;
  const yMax = maxValue + yPadding;
  const plotWidth = CHART_FRAME.width - CHART_FRAME.paddingLeft - CHART_FRAME.paddingRight;
  const plotHeight = CHART_FRAME.height - CHART_FRAME.paddingTop - CHART_FRAME.paddingBottom;
  const xForIndex = (index: number) =>
    CHART_FRAME.paddingLeft + (index / Math.max(labels.length - 1, 1)) * plotWidth;
  const yForValue = (value: number) =>
    CHART_FRAME.paddingTop + ((yMax - value) / Math.max(yMax - yMin, 1e-6)) * plotHeight;

  const tickValues = Array.from({ length: 5 }, (_, index) => yMax - (index / 4) * (yMax - yMin));
  const labelIndices = sampleIndices(labels.length, 5);
  const hoveredLabel = hoverIndex !== null ? labels[hoverIndex] : labels[labels.length - 1];
  const hoveredDisplayLabel = formatCompactTimeLabel(hoveredLabel ?? "");
  const hoveredValues = activeGroups
    .map((group) => {
      const point = group.data[hoverIndex ?? group.data.length - 1];
      if (!point) {
        return null;
      }
      return { color: group.color, label: group.label, value: point.value };
    })
    .filter((item): item is { color: string; label: string; value: number } => item !== null);

  return (
    <div className="interactive-chart-card">
      <div className="interactive-chart-header">
        <div>
          <p className="eyebrow">Interactive Series</p>
          <h4>{title}</h4>
          {subtitle ? <p className="chart-subtitle">{subtitle}</p> : null}
        </div>
        <div className="chart-readout">
          <strong>{hoveredDisplayLabel}</strong>
          <div className="chart-readout-values">
            {hoveredValues.map((item) => (
              <span key={item.label} style={{ color: item.color }}>
                {item.label}: {formatValue(item.value)}
              </span>
            ))}
          </div>
        </div>
      </div>

      {groups.length > 1 ? (
        <div className="series-toggle-row">
          {groups.map((group) => {
            const isVisible = visibleIds.includes(group.id);
            return (
              <button
                key={group.id}
                className={`series-toggle ${isVisible ? "is-active" : ""}`}
                onClick={() => {
                  setVisibleIds((previous) => {
                    if (previous.includes(group.id)) {
                      return previous.length === 1 ? previous : previous.filter((id) => id !== group.id);
                    }
                    return [...previous, group.id];
                  });
                }}
                type="button"
              >
                <span className="series-dot" style={{ background: group.color }} />
                {group.label}
              </button>
            );
          })}
        </div>
      ) : null}

      <div
        ref={chartRef}
        className="interactive-chart-shell"
        onMouseLeave={() => setHoverIndex(null)}
        onMouseMove={(event) => {
          const rect = chartRef.current?.getBoundingClientRect();
          if (!rect) {
            return;
          }

          const relativeX =
            ((event.clientX - rect.left) / Math.max(rect.width, 1)) * CHART_FRAME.width - CHART_FRAME.paddingLeft;
          const ratio = Math.max(0, Math.min(1, relativeX / Math.max(plotWidth, 1)));
          setHoverIndex(Math.round(ratio * Math.max(labels.length - 1, 0)));
        }}
      >
        <svg viewBox={`0 0 ${CHART_FRAME.width} ${CHART_FRAME.height}`} aria-hidden="true">
          <defs>
            {groups
              .filter((group) => group.fill)
              .map((group) => (
                <linearGradient key={group.id} id={`gradient-${group.id}`} x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor={group.color} stopOpacity="0.22" />
                  <stop offset="60%" stopColor={group.color} stopOpacity="0.06" />
                  <stop offset="100%" stopColor={group.color} stopOpacity="0" />
                </linearGradient>
              ))}
            <filter id="focus-glow">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {tickValues.map((value) => {
            const y = yForValue(value);
            return (
              <g key={value}>
                <line className="chart-grid-line" x1={CHART_FRAME.paddingLeft} x2={CHART_FRAME.width - CHART_FRAME.paddingRight} y1={y} y2={y} />
                <text className="chart-axis-label" x={CHART_FRAME.paddingLeft - 10} y={y + 4} textAnchor="end">
                  {formatValue(value)}
                </text>
              </g>
            );
          })}

          {labelIndices.map((index) => {
            const x = xForIndex(index);
            return (
              <text
                key={labels[index]}
                className="chart-axis-label"
                x={x}
                y={CHART_FRAME.height - 10}
                textAnchor={index === 0 ? "start" : index === labels.length - 1 ? "end" : "middle"}
              >
                {formatCompactTimeLabel(labels[index])}
              </text>
            );
          })}

          {activeGroups.map((group) => {
            const coords: [number, number][] = group.data.map((point, index) => [xForIndex(index), yForValue(point.value)]);
            const smoothD = buildSmoothPath(coords);
            const bottomY = CHART_FRAME.height - CHART_FRAME.paddingBottom;
            const areaD =
              group.fill && coords.length > 0
                ? `${smoothD} L ${coords[coords.length - 1][0]},${bottomY} L ${coords[0][0]},${bottomY} Z`
                : null;

            return (
              <g key={group.id}>
                {areaD ? <path d={areaD} fill={`url(#gradient-${group.id})`} /> : null}
                <path
                  className={`chart-series-line ${group.dashed ? "is-dashed" : ""}`}
                  d={smoothD}
                  fill="none"
                  stroke={group.color}
                  strokeWidth={group.id === "result" ? 2.5 : 2}
                />
              </g>
            );
          })}

          {hoverIndex !== null ? (
            <>
              <line
                className="chart-focus-line"
                x1={xForIndex(hoverIndex)}
                x2={xForIndex(hoverIndex)}
                y1={CHART_FRAME.paddingTop}
                y2={CHART_FRAME.height - CHART_FRAME.paddingBottom}
              />
              {activeGroups.map((group) => {
                const point = group.data[hoverIndex];
                if (!point) {
                  return null;
                }
                return (
                  <circle
                    key={`${group.id}-${hoverIndex}`}
                    className="chart-focus-point"
                    cx={xForIndex(hoverIndex)}
                    cy={yForValue(point.value)}
                    fill={group.color}
                    r={5}
                    filter="url(#focus-glow)"
                  />
                );
              })}
            </>
          ) : null}
        </svg>
      </div>
    </div>
  );
}

function ProfileChart({
  series,
  markers
}: {
  series: ProfilePoint[];
  markers: { label: string; depth: number }[];
}) {
  const shellRef = useRef<HTMLDivElement | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const filteredSeries = useMemo(
    () => series.filter((item) => Number.isFinite(item.depth) && Number.isFinite(item.value) && Math.abs(item.depth) < 9000),
    [series]
  );
  const filteredMarkers = useMemo(
    () => markers.filter((marker) => Number.isFinite(marker.depth) && Math.abs(marker.depth) < 9000),
    [markers]
  );

  if (filteredSeries.length === 0) {
    return <EmptyState message="No profile payload is available for this result yet." />;
  }

  const values = filteredSeries.map((item) => item.value);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const valueSpan = maxValue - minValue;
  const valuePadding = valueSpan === 0 ? Math.max(Math.abs(maxValue) * 0.15, 1) : valueSpan * 0.12;
  const xMin = minValue - valuePadding;
  const xMax = maxValue + valuePadding;
  const depths = filteredSeries.map((item) => Math.abs(item.depth));
  const markerDepths = filteredMarkers.map((marker) => Math.abs(marker.depth));
  const maxDepth = Math.max(...depths, ...(markerDepths.length > 0 ? markerDepths : [0]), 1);
  const plotWidth = DETAIL_FRAME.width - DETAIL_FRAME.paddingLeft - DETAIL_FRAME.paddingRight;
  const plotHeight = DETAIL_FRAME.height - DETAIL_FRAME.paddingTop - DETAIL_FRAME.paddingBottom;
  const xForValue = (value: number) =>
    DETAIL_FRAME.paddingLeft + ((value - xMin) / Math.max(xMax - xMin, 1e-6)) * plotWidth;
  const yForDepth = (depth: number) =>
    DETAIL_FRAME.paddingTop + (Math.abs(depth) / maxDepth) * plotHeight;

  const curveCoords: [number, number][] = filteredSeries.map((item) => [xForValue(item.value), yForDepth(item.depth)]);
  const smoothProfileD = buildSmoothPath(curveCoords);
  const plotBottom = DETAIL_FRAME.paddingTop + plotHeight;

  // Fill polygon: left axis edge -> smooth curve -> back to left
  const fillAreaD = curveCoords.length > 0
    ? `M ${DETAIL_FRAME.paddingLeft},${DETAIL_FRAME.paddingTop} L ${curveCoords[0][0]},${curveCoords[0][1]} ${smoothProfileD.indexOf("C") >= 0 ? smoothProfileD.slice(smoothProfileD.indexOf("C")) : ""} L ${curveCoords[curveCoords.length - 1][0]},${plotBottom} L ${DETAIL_FRAME.paddingLeft},${plotBottom} Z`
    : "";

  const hoveredPoint = hoverIndex !== null ? filteredSeries[hoverIndex] : null;
  const displayPoint = hoveredPoint ?? filteredSeries[filteredSeries.length - 1];
  const valueTicks = Array.from({ length: 5 }, (_, i) => xMin + (i / 4) * (xMax - xMin));
  const depthTicks = Array.from({ length: 5 }, (_, i) => (i / 4) * maxDepth);

  return (
    <div className="detail-chart-card">
      <div className="detail-chart-header">
        <div>
          <p className="eyebrow">Vertical Structure</p>
          <h4>Profile</h4>
        </div>
        <div className="chart-readout">
          <strong>
            {displayPoint
              ? `${formatValue(displayPoint.value)} at ${formatValue(Math.abs(displayPoint.depth))} m`
              : "Profile"}
          </strong>
          <div className="chart-readout-values">
            {filteredMarkers.map((marker) => (
              <span key={marker.label}>
                {marker.label}: {formatValue(Math.abs(marker.depth))} m
              </span>
            ))}
          </div>
        </div>
      </div>

      <div
        ref={shellRef}
        className="detail-chart-shell"
        onMouseLeave={() => setHoverIndex(null)}
        onMouseMove={(event) => {
          const rect = shellRef.current?.getBoundingClientRect();
          if (!rect) return;
          const relativeY =
            ((event.clientY - rect.top) / Math.max(rect.height, 1)) * DETAIL_FRAME.height - DETAIL_FRAME.paddingTop;
          const depthRatio = clamp(relativeY / Math.max(plotHeight, 1), 0, 1);
          const targetDepth = depthRatio * maxDepth;
          let closestIndex = 0;
          let closestDistance = Number.POSITIVE_INFINITY;
          filteredSeries.forEach((point, index) => {
            const distance = Math.abs(Math.abs(point.depth) - targetDepth);
            if (distance < closestDistance) {
              closestDistance = distance;
              closestIndex = index;
            }
          });
          setHoverIndex(closestIndex);
        }}
      >
        <svg viewBox={`0 0 ${DETAIL_FRAME.width} ${DETAIL_FRAME.height}`} aria-hidden="true">
          <defs>
            <linearGradient id="profile-fill-grad" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%"   stopColor="#74c0e0" stopOpacity="0.55" />
              <stop offset="45%"  stopColor="#2a73c5" stopOpacity="0.48" />
              <stop offset="100%" stopColor="#0c2a52" stopOpacity="0.38" />
            </linearGradient>
            <clipPath id="profile-plot-clip">
              <rect x={DETAIL_FRAME.paddingLeft} y={DETAIL_FRAME.paddingTop} width={plotWidth} height={plotHeight} />
            </clipPath>
            <filter id="focus-glow">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* X grid + labels (values) */}
          {valueTicks.map((value) => {
            const x = xForValue(value);
            return (
              <g key={`xv-${value}`}>
                <line className="chart-grid-line" x1={x} x2={x} y1={DETAIL_FRAME.paddingTop} y2={DETAIL_FRAME.height - DETAIL_FRAME.paddingBottom} />
                <text className="chart-axis-label" x={x} y={DETAIL_FRAME.height - 12} textAnchor="middle">
                  {formatValue(value)}
                </text>
              </g>
            );
          })}

          {/* Y grid + labels (depth) */}
          {depthTicks.map((depth) => {
            const y = yForDepth(depth);
            return (
              <g key={`yd-${depth}`}>
                <line className="chart-grid-line" x1={DETAIL_FRAME.paddingLeft} x2={DETAIL_FRAME.width - DETAIL_FRAME.paddingRight} y1={y} y2={y} />
                <text className="chart-axis-label" x={DETAIL_FRAME.paddingLeft - 10} y={y + 4} textAnchor="end">
                  {formatValue(depth)}
                </text>
              </g>
            );
          })}

          {/* Depth-gradient fill area */}
          <path
            d={fillAreaD}
            fill="url(#profile-fill-grad)"
            clipPath="url(#profile-plot-clip)"
          />

          {/* Profile curve */}
          <path
            className="chart-series-line"
            d={smoothProfileD}
            fill="none"
            stroke="#4f7df3"
            strokeWidth={2.5}
            clipPath="url(#profile-plot-clip)"
          />

          {/* Feature markers */}
          {filteredMarkers.map((marker) => {
            const y = yForDepth(marker.depth);
            return (
              <g key={marker.label}>
                <line
                  x1={DETAIL_FRAME.paddingLeft}
                  x2={DETAIL_FRAME.width - DETAIL_FRAME.paddingRight}
                  y1={y}
                  y2={y}
                  stroke="#dd8f15"
                  strokeWidth={1.4}
                  strokeDasharray="6 3"
                />
                <text className="chart-marker-label" x={DETAIL_FRAME.width - DETAIL_FRAME.paddingRight} y={y - 6} textAnchor="end">
                  {marker.label}
                </text>
              </g>
            );
          })}

          {/* Hover crosshair */}
          {hoveredPoint ? (
            <>
              <line
                x1={DETAIL_FRAME.paddingLeft}
                x2={DETAIL_FRAME.width - DETAIL_FRAME.paddingRight}
                y1={yForDepth(hoveredPoint.depth)}
                y2={yForDepth(hoveredPoint.depth)}
                stroke="rgba(79,125,243,0.35)"
                strokeWidth={1}
                strokeDasharray="4 3"
              />
              <circle
                className="chart-focus-point"
                cx={xForValue(hoveredPoint.value)}
                cy={yForDepth(hoveredPoint.depth)}
                fill="#4f7df3"
                r={5}
                filter="url(#focus-glow)"
              />
            </>
          ) : null}

          {/* Axis titles */}
          <text className="chart-axis-title" x={DETAIL_FRAME.paddingLeft + plotWidth / 2} y={DETAIL_FRAME.height - 2} textAnchor="middle">
            Value
          </text>
          <text
            className="chart-axis-title"
            x={14}
            y={DETAIL_FRAME.paddingTop + plotHeight / 2}
            textAnchor="middle"
            transform={`rotate(-90 14 ${DETAIL_FRAME.paddingTop + plotHeight / 2})`}
          >
            Depth (m)
          </text>
        </svg>
      </div>
    </div>
  );
}

function HovmollerChart({
  rows,
  displayInfo,
  overlay = [],
  timeLabels,
  depthIntegratedSeries = [],
}: {
  rows: WorkspaceData["hovmollerRows"];
  displayInfo?: WorkspaceData["hovmollerDisplayInfo"];
  overlay: WorkspaceData["overlaySeries"];
  timeLabels?: WorkspaceData["hovmollerTimeLabels"];
  depthIntegratedSeries?: WorkspaceData["hovmollerDepthIntegratedSeries"];
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hoverState, setHoverState] = useState<{
    row: number;
    col: number;
    plotX: number;
    plotY: number;
    value: number;
  } | null>(null);
  const [heatmapUrl, setHeatmapUrl] = useState<string>("");
  const frame = {
    width: 640,
    height: 440,
    paddingTop: 18,
    paddingRight: 84,
    paddingBottom: 14,
    paddingLeft: 78
  };
  const plotWidth = frame.width - frame.paddingLeft - frame.paddingRight;
  const plotHeight = 246;
  const curveTop = frame.paddingTop + plotHeight + 34;
  const curveHeight = 82;
  const timeTickY = curveTop + curveHeight + 24;
  const timeAxisY = frame.height - 4;

  const filteredRows = useMemo(
    () => {
      const configuredDepthLevels = displayInfo?.depthLevels?.filter(
        (value): value is number => typeof value === "number" && Number.isFinite(value)
      );
      const finiteRows = rows.map((row, index) => hovmollerRowWithConfiguredDepth(row, index, configuredDepthLevels)).filter((row) => {
        if (!rowHasFiniteValue(row)) {
          return false;
        }
        const depthValue = hovmollerRowDepth(row);
        return Number.isFinite(depthValue) && Math.abs(depthValue) < 9000;
      });
      return [...finiteRows].sort((a, b) => hovmollerRowDepth(a) - hovmollerRowDepth(b));
    },
    [displayInfo?.depthLevels, rows]
  );

  const nColumns = useMemo(
    () => (filteredRows.length > 0 ? Math.max(...filteredRows.map((row) => row.values.length), 1) : 0),
    [filteredRows]
  );
  const finiteValues = useMemo(() => finiteValuesFromRows(filteredRows), [filteredRows]);
  const valueLimit = useMemo(() => robustSymmetricLimit(finiteValues), [finiteValues]);
  const contourLevels = useMemo(() => buildHovmollerContourLevels(valueLimit), [valueLimit]);
  const displayDepthValues = useMemo(() => filteredRows.map(hovmollerRowDepth), [filteredRows]);
  const finiteDepthValues = useMemo(
    () => displayDepthValues.filter((depth) => Number.isFinite(depth)),
    [displayDepthValues]
  );
  const depthTop = finiteDepthValues.length > 0 ? Math.min(...finiteDepthValues) : 0;
  const depthBottom = finiteDepthValues.length > 0 ? Math.max(...finiteDepthValues) : depthTop;
  const contourLabels = useMemo(
    () => buildHovmollerContourLabels(filteredRows, nColumns, plotWidth, plotHeight, contourLevels),
    [contourLevels, filteredRows, nColumns, plotHeight, plotWidth]
  );
  const valueMin = -valueLimit;
  const valueMax = valueLimit;
  const svgViewWidth = frame.width + 64;
  const svgViewHeight = frame.height;

  useEffect(() => {
    if (filteredRows.length === 0 || finiteValues.length === 0 || nColumns === 0) return;

    const canvas = document.createElement("canvas");
    canvas.width = plotWidth;
    canvas.height = plotHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(plotWidth, plotHeight);

    for (let py = 0; py < plotHeight; py++) {
      for (let px = 0; px < plotWidth; px++) {
        const fracCol = ((px + 0.5) / plotWidth) * nColumns;
        const targetDepth =
          depthBottom > depthTop
            ? depthTop + ((py + 0.5) / plotHeight) * (depthBottom - depthTop)
            : depthTop;
        const value = interpolateHovmollerValue(filteredRows, displayDepthValues, targetDepth, fracCol, nColumns);

        const [red, green, blue] = hovmollerColorRGB(value, valueLimit);
        const idx = (py * plotWidth + px) * 4;
        imageData.data[idx] = red;
        imageData.data[idx + 1] = green;
        imageData.data[idx + 2] = blue;
        imageData.data[idx + 3] = Number.isFinite(value) ? 255 : 130;
      }
    }

    ctx.putImageData(imageData, 0, 0);
    drawHovmollerContours(ctx, filteredRows, nColumns, plotWidth, plotHeight, contourLevels);
    setHeatmapUrl(canvas.toDataURL());
  }, [contourLevels, depthBottom, depthTop, displayDepthValues, filteredRows, finiteValues.length, nColumns, plotHeight, plotWidth, valueLimit]);

  if (rows.length === 0) {
    return <EmptyState message="No Hovmoller payload is available for this result yet." />;
  }

  if (filteredRows.length === 0 || finiteValues.length === 0) {
    return <EmptyState message="The Hovmoller payload contains no finite values. Try rerunning with a wider transect region or check that the transect crosses valid ocean cells." />;
  }

  const cellWidth = plotWidth / nColumns;
  const resolvedTimeLabels =
    timeLabels && timeLabels.length >= nColumns
      ? timeLabels.slice(0, nColumns)
      : overlay.length >= nColumns
        ? overlay.slice(0, nColumns).map((point) => point.day)
        : Array.from({ length: nColumns }, (_, index) => `t${index + 1}`);
  const timeTickIndices = sampleIndices(nColumns, 6);
  const colorbarStops = Array.from({ length: 9 }, (_, index) => {
    const t = index / 8;
    const [red, green, blue] = interpolateStops(HOVMOLLER_COLORMAP_STOPS, t);
    return { t, color: `rgb(${red},${green},${blue})` };
  });
  const colorbarX = frame.width - frame.paddingRight + 14;
  const colorbarW = 10;
  const colorbarH = plotHeight;
  const colorbarLabelX = colorbarX + colorbarW / 2;
  const colorbarUnitX = colorbarX + colorbarW + 18;
  const hovered = hoverState && Number.isFinite(hoverState.value) ? hoverState.value : null;
  const hoveredDepth = hoverState ? filteredRows[hoverState.row]?.depthLabel : null;
  const hoveredTime = hoverState ? resolvedTimeLabels[hoverState.col] : null;
  const hoverOverlayX = hoverState ? frame.paddingLeft + hoverState.plotX : null;
  const hoverOverlayY = hoverState ? frame.paddingTop + hoverState.plotY : null;
  const displaySubtitle =
    displayInfo?.aggregation && displayInfo.aggregation !== "none"
      ? `${displayInfo.aggregationLabel} (${displayInfo.originalColumns} to ${displayInfo.displayColumns} steps)`
      : "Original time steps";
  const units = displayInfo?.units || "s^-1";
  const integratedUnits = displayInfo?.depthIntegratedUnits || `${units} m`;
  const integratedPoints = depthIntegratedSeries.slice(0, nColumns).map((point, index) => ({
    label: point.label || resolvedTimeLabels[index] || `t${index + 1}`,
    value: point.value,
  }));
  const integratedFinite = integratedPoints.map((point) => point.value).filter((value) => Number.isFinite(value));
  const integratedExtent = finiteExtentFromValues(integratedFinite);
  const integratedPadding = integratedExtent ? Math.max((integratedExtent.max - integratedExtent.min) * 0.12, Math.abs(integratedExtent.max) * 0.02, 1e-12) : 1;
  const integratedMin = integratedExtent ? integratedExtent.min - integratedPadding : -1;
  const integratedMax = integratedExtent ? integratedExtent.max + integratedPadding : 1;
  const integratedMean =
    integratedFinite.length > 0 ? integratedFinite.reduce((sum, value) => sum + value, 0) / integratedFinite.length : Number.NaN;
  const integratedStd =
    integratedFinite.length > 1
      ? Math.sqrt(integratedFinite.reduce((sum, value) => sum + (value - integratedMean) ** 2, 0) / (integratedFinite.length - 1))
      : Number.NaN;
  const integratedCoords: [number, number][] = integratedPoints
    .map((point, index): [number, number] | null => {
      if (!Number.isFinite(point.value) || !integratedExtent) {
        return null;
      }
      const x = frame.paddingLeft + (index + 0.5) * cellWidth;
      const normalized = (point.value - integratedMin) / Math.max(integratedMax - integratedMin, 1e-12);
      const y = curveTop + curveHeight - clamp(normalized, 0, 1) * curveHeight;
      return [x, y];
    })
    .filter((coord): coord is [number, number] => Boolean(coord));
  const integratedPath = buildSmoothPath(integratedCoords);
  const hasIntegratedCurve = integratedCoords.length > 0 && integratedExtent !== null;

  return (
    <div className="detail-chart-card hovmoller-chart-card">
      <div className="detail-chart-header">
        <div>
          <p className="eyebrow">Time-Depth Evolution</p>
          <h4>Hovmöller</h4>
          <p className="chart-subtitle">{displaySubtitle}</p>
        </div>
        <div className="chart-readout">
          <strong>
            {hovered !== null && hoveredTime && hoveredDepth
              ? `${hoveredTime} · ${hoveredDepth} · ${formatValue(hovered)}`
              : "Hover to inspect"}
          </strong>
          <div className="chart-readout-values">
            <span>Min: {formatValue(valueMin)}</span>
            <span>Max: {formatValue(valueMax)}</span>
            <span>{units}</span>
            {Number.isFinite(integratedMean) ? <span>Integral mean: {formatValue(integratedMean)}</span> : null}
          </div>
        </div>
      </div>

      <div
        className="detail-chart-shell"
        style={{ minHeight: 440 }}
        onMouseLeave={() => setHoverState(null)}
        onMouseMove={(event) => {
          const svg = svgRef.current;
          const ctm = svg?.getScreenCTM();
          if (!svg || !ctm) return;
          const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
          const rawPlotX = point.x - frame.paddingLeft;
          const rawPlotY = point.y - frame.paddingTop;
          if (rawPlotX < 0 || rawPlotX > plotWidth || rawPlotY < 0 || rawPlotY > plotHeight) {
            setHoverState(null);
            return;
          }
          const plotX = clamp(rawPlotX, 0, Math.max(plotWidth - 1e-6, 0));
          const plotY = clamp(rawPlotY, 0, Math.max(plotHeight - 1e-6, 0));
          const fracCol = (plotX / Math.max(plotWidth, 1e-6)) * nColumns;
          const targetDepth =
            depthBottom > depthTop
              ? depthTop + (plotY / Math.max(plotHeight, 1e-6)) * (depthBottom - depthTop)
              : depthTop;
          const col = clamp(Math.round(fracCol - 0.5), 0, nColumns - 1);
          const row = nearestDepthIndex(displayDepthValues, targetDepth);
          const value = interpolateHovmollerValue(filteredRows, displayDepthValues, targetDepth, fracCol, nColumns);
          setHoverState({ row, col, plotX, plotY, value });
        }}
      >
        <svg ref={svgRef} viewBox={`0 0 ${svgViewWidth} ${svgViewHeight}`} aria-hidden="true">
          <defs>
            <linearGradient id="hov-colorbar-grad" x1="0" x2="0" y1="1" y2="0">
              {colorbarStops.map(({ t, color }) => (
                <stop key={t} offset={`${t * 100}%`} stopColor={color} />
              ))}
            </linearGradient>
            <clipPath id="hov-plot-clip">
              <rect x={frame.paddingLeft} y={frame.paddingTop} width={plotWidth} height={plotHeight} />
            </clipPath>
          </defs>

          <rect x={frame.paddingLeft} y={frame.paddingTop} width={plotWidth} height={plotHeight} fill="#e2e8f0" />
          {heatmapUrl ? (
            <image
              href={heatmapUrl}
              x={frame.paddingLeft}
              y={frame.paddingTop}
              width={plotWidth}
              height={plotHeight}
              preserveAspectRatio="none"
            />
          ) : null}

          <g clipPath="url(#hov-plot-clip)" pointerEvents="none">
            {contourLabels.map((label, index) => (
              <text
                key={`${label.level}-${index}`}
                x={frame.paddingLeft + label.x}
                y={frame.paddingTop + label.y}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={9}
                fontWeight={600}
                fill="#0f172a"
                stroke="rgba(255,255,255,0.82)"
                strokeWidth={3}
                paintOrder="stroke fill"
              >
                {label.text}
              </text>
            ))}
          </g>

          {Array.from({ length: 6 }, (_, index) => {
            const depth = depthTop + (index / 5) * (depthBottom - depthTop);
            const y = frame.paddingTop + (index / 5) * plotHeight;
            return (
              <g key={`d-${index}`}>
                <line stroke="rgba(255,255,255,0.26)" strokeWidth={0.6} x1={frame.paddingLeft} x2={frame.paddingLeft + plotWidth} y1={y} y2={y} />
                <text className="chart-axis-label" x={frame.paddingLeft - 8} y={y + 4} textAnchor="end" fill="#64748b">
                  {formatDepthTickMeters(depth)}
                </text>
              </g>
            );
          })}

          {timeTickIndices.map((mapped) => {
            const x = frame.paddingLeft + (mapped + 0.5) * cellWidth;
            return (
              <text key={`t-${mapped}`} className="chart-axis-label" x={x} y={timeTickY} textAnchor="middle" fill="#64748b">
                {formatCompactTimeLabel(resolvedTimeLabels[mapped] ?? `t${mapped + 1}`)}
              </text>
            );
          })}

          {hoverOverlayX !== null && hoverOverlayY !== null ? (
            <>
              <line x1={hoverOverlayX} x2={hoverOverlayX} y1={frame.paddingTop} y2={frame.paddingTop + plotHeight} stroke="rgba(15,23,42,0.45)" strokeWidth={1} strokeDasharray="4 3" clipPath="url(#hov-plot-clip)" />
              <line x1={frame.paddingLeft} x2={frame.paddingLeft + plotWidth} y1={hoverOverlayY} y2={hoverOverlayY} stroke="rgba(15,23,42,0.45)" strokeWidth={1} strokeDasharray="4 3" clipPath="url(#hov-plot-clip)" />
            </>
          ) : null}

          <rect x={frame.paddingLeft} y={frame.paddingTop} width={plotWidth} height={plotHeight} fill="none" stroke="rgba(100,116,139,0.45)" strokeWidth={0.8} />
          <rect x={colorbarX} y={frame.paddingTop} width={colorbarW} height={colorbarH} fill="url(#hov-colorbar-grad)" rx={2} />
          <text className="chart-axis-label" x={colorbarLabelX} y={frame.paddingTop - 4} textAnchor="middle" fill="#64748b">
            {formatValue(valueMax)}
          </text>
          <text className="chart-axis-label" x={colorbarLabelX} y={frame.paddingTop + colorbarH + 14} textAnchor="middle" fill="#64748b">
            {formatValue(valueMin)}
          </text>
          <text className="chart-axis-title" x={colorbarUnitX} y={frame.paddingTop + colorbarH / 2} textAnchor="middle" transform={`rotate(90 ${colorbarUnitX} ${frame.paddingTop + colorbarH / 2})`}>
            {units}
          </text>
          <text className="chart-axis-title" x={24} y={frame.paddingTop + plotHeight / 2} textAnchor="middle" transform={`rotate(-90 24 ${frame.paddingTop + plotHeight / 2})`}>
            Depth (m)
          </text>

          {hasIntegratedCurve ? (
            <g>
              <rect
                x={frame.paddingLeft}
                y={curveTop}
                width={plotWidth}
                height={curveHeight}
                fill="rgba(255,255,255,0.72)"
                stroke="rgba(100,116,139,0.35)"
                strokeWidth={0.8}
              />
              {Array.from({ length: 3 }, (_, index) => {
                const t = index / 2;
                const y = curveTop + t * curveHeight;
                const value = integratedMax - t * (integratedMax - integratedMin);
                return (
                  <g key={`int-grid-${index}`}>
                    <line x1={frame.paddingLeft} x2={frame.paddingLeft + plotWidth} y1={y} y2={y} stroke="rgba(100,116,139,0.18)" strokeWidth={0.7} />
                    <text className="chart-axis-label" x={frame.paddingLeft - 8} y={y + 4} textAnchor="end" fill="#64748b">
                      {formatValue(value)}
                    </text>
                  </g>
                );
              })}
              {integratedPath ? (
                <path d={integratedPath} fill="none" stroke="#0f172a" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
              ) : null}
              {integratedCoords.length === 1 ? (
                <circle cx={integratedCoords[0][0]} cy={integratedCoords[0][1]} r={3} fill="#0f172a" />
              ) : null}
              <text className="chart-axis-label" x={frame.paddingLeft} y={curveTop - 8} fill="#334155">
                Depth-integrated
              </text>
              {Number.isFinite(integratedMean) ? (
                <text className="chart-axis-label" x={frame.paddingLeft + plotWidth} y={curveTop - 8} textAnchor="end" fill="#334155">
                  {formatValue(integratedMean)} ± {Number.isFinite(integratedStd) ? formatValue(integratedStd) : "0"} {integratedUnits}
                </text>
              ) : null}
            </g>
          ) : null}

          <text className="chart-axis-title" x={frame.paddingLeft + plotWidth / 2} y={timeAxisY} textAnchor="middle">
            Time
          </text>
        </svg>
      </div>
    </div>
  );
}

function SectionChart({
  rows,
  distanceKm,
  axisTitle,
  sliceLabel,
}: {
  rows: WorkspaceData["sectionRows"];
  distanceKm?: WorkspaceData["sectionDistanceKm"];
  axisTitle?: WorkspaceData["sectionAxisTitle"];
  sliceLabel?: WorkspaceData["sectionSliceLabel"];
}) {
  const shellRef = useRef<HTMLDivElement | null>(null);
  const [hoverCell, setHoverCell] = useState<{ row: number; col: number } | null>(null);
  const [heatmapUrl, setHeatmapUrl] = useState<string>("");

  const filteredRows = useMemo(
    () => rows.filter((row) => row.values.some((value) => Number.isFinite(value))),
    [rows]
  );
  const nColumns = useMemo(
    () => (filteredRows.length > 0 ? Math.max(...filteredRows.map((row) => row.values.length), 1) : 0),
    [filteredRows]
  );
  const valueExtent = useMemo(() => finiteExtentFromRows(filteredRows), [filteredRows]);

  const valueMin = valueExtent?.min ?? 0;
  const valueMax = valueExtent?.max ?? 1;
  const plotWidth = DETAIL_FRAME.width - DETAIL_FRAME.paddingLeft - DETAIL_FRAME.paddingRight;
  const plotHeight = DETAIL_FRAME.height - DETAIL_FRAME.paddingTop - DETAIL_FRAME.paddingBottom;

  useEffect(() => {
    if (filteredRows.length === 0 || !valueExtent || nColumns === 0) return;

    const canvas = document.createElement("canvas");
    canvas.width = plotWidth;
    canvas.height = plotHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(plotWidth, plotHeight);
    const range = Math.max(valueMax - valueMin, 1e-9);
    const nRows = filteredRows.length;

    for (let py = 0; py < plotHeight; py++) {
      for (let px = 0; px < plotWidth; px++) {
        const fracCol = (px / plotWidth) * nColumns;
        const fracRow = (py / plotHeight) * nRows;
        const gc = fracCol - 0.5;
        const gr = fracRow - 0.5;

        const c0 = clamp(Math.floor(gc), 0, nColumns - 1);
        const c1 = clamp(c0 + 1, 0, nColumns - 1);
        const r0 = clamp(Math.floor(gr), 0, nRows - 1);
        const r1 = clamp(r0 + 1, 0, nRows - 1);
        const dc = clamp(gc - c0, 0, 1);
        const dr = clamp(gr - r0, 0, 1);

        const getValue = (r: number, c: number) => filteredRows[r]?.values[c] ?? valueMin;
        const value =
          getValue(r0, c0) * (1 - dr) * (1 - dc) +
          getValue(r0, c1) * (1 - dr) * dc +
          getValue(r1, c0) * dr * (1 - dc) +
          getValue(r1, c1) * dr * dc;

        const finiteValue = Number.isFinite(value) ? value : valueMin;
        const [red, green, blue] = colormapRGB(clamp((finiteValue - valueMin) / range, 0, 1));
        const index = (py * plotWidth + px) * 4;
        imageData.data[index] = red;
        imageData.data[index + 1] = green;
        imageData.data[index + 2] = blue;
        imageData.data[index + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
    setHeatmapUrl(canvas.toDataURL());
  }, [filteredRows, nColumns, plotHeight, plotWidth, valueExtent, valueMax, valueMin]);

  if (filteredRows.length === 0 || !valueExtent || nColumns === 0) {
    return <EmptyState message="No section payload is available for this result yet." />;
  }

  const cellWidth = plotWidth / nColumns;
  const cellHeight = plotHeight / filteredRows.length;
  const xLabels =
    distanceKm && distanceKm.length >= nColumns
      ? distanceKm.slice(0, nColumns).map((value) => `${formatValue(value)} km`)
      : Array.from({ length: nColumns }, (_, index) => `x${index + 1}`);
  const hovered = hoverCell ? filteredRows[hoverCell.row]?.values[hoverCell.col] : null;
  const hoveredRow = hoverCell ? filteredRows[hoverCell.row]?.label : null;
  const hoveredDistance = hoverCell ? xLabels[hoverCell.col] : null;
  const colorbarStops = Array.from({ length: 9 }, (_, i) => {
    const t = i / 8;
    const [r, g, b] = colormapRGB(t);
    return { t, color: `rgb(${r},${g},${b})` };
  });
  const colorbarX = DETAIL_FRAME.width - DETAIL_FRAME.paddingRight + 6;
  const colorbarW = 10;
  const colorbarH = plotHeight;

  return (
    <div className="detail-chart-card">
      <div className="detail-chart-header">
        <div>
          <p className="eyebrow">Distance-Depth Structure</p>
          <h4>Transect Section</h4>
          {sliceLabel ? <p className="chart-subtitle">{sliceLabel}</p> : null}
        </div>
        <div className="chart-readout">
          <strong>
            {hovered !== null && hoveredDistance && hoveredRow
              ? `${hoveredDistance} · ${hoveredRow} · ${formatValue(hovered)}`
              : "Hover to inspect"}
          </strong>
          <div className="chart-readout-values">
            <span>Min: {formatValue(valueMin)}</span>
            <span>Max: {formatValue(valueMax)}</span>
          </div>
        </div>
      </div>

      <div
        ref={shellRef}
        className="detail-chart-shell"
        onMouseLeave={() => setHoverCell(null)}
        onMouseMove={(event) => {
          const rect = shellRef.current?.getBoundingClientRect();
          if (!rect) return;
          const relX =
            ((event.clientX - rect.left) / Math.max(rect.width, 1)) * DETAIL_FRAME.width - DETAIL_FRAME.paddingLeft;
          const relY =
            ((event.clientY - rect.top) / Math.max(rect.height, 1)) * DETAIL_FRAME.height - DETAIL_FRAME.paddingTop;
          const col = clamp(Math.floor(relX / Math.max(cellWidth, 1)), 0, nColumns - 1);
          const row = clamp(Math.floor(relY / Math.max(cellHeight, 1)), 0, filteredRows.length - 1);
          setHoverCell({ row, col });
        }}
      >
        <svg viewBox={`0 0 ${DETAIL_FRAME.width + 20} ${DETAIL_FRAME.height}`} aria-hidden="true">
          <defs>
            <linearGradient id="section-colorbar-grad" x1="0" x2="0" y1="1" y2="0">
              {colorbarStops.map(({ t, color }) => (
                <stop key={t} offset={`${t * 100}%`} stopColor={color} />
              ))}
            </linearGradient>
          </defs>

          <rect
            x={DETAIL_FRAME.paddingLeft}
            y={DETAIL_FRAME.paddingTop}
            width={plotWidth}
            height={plotHeight}
            fill="#18293d"
          />

          {heatmapUrl ? (
            <image
              href={heatmapUrl}
              x={DETAIL_FRAME.paddingLeft}
              y={DETAIL_FRAME.paddingTop}
              width={plotWidth}
              height={plotHeight}
              preserveAspectRatio="none"
            />
          ) : null}

          {Array.from({ length: Math.min(filteredRows.length, 6) }, (_, i) => {
            const mapped =
              filteredRows.length === 1 ? 0 : Math.round((i / Math.max(Math.min(filteredRows.length, 6) - 1, 1)) * (filteredRows.length - 1));
            const y = DETAIL_FRAME.paddingTop + (mapped + 0.5) * cellHeight;
            return (
              <g key={`section-y-${mapped}`}>
                <line
                  stroke="rgba(255,255,255,0.1)"
                  strokeWidth={0.5}
                  x1={DETAIL_FRAME.paddingLeft}
                  x2={DETAIL_FRAME.paddingLeft + plotWidth}
                  y1={y}
                  y2={y}
                />
                <text className="chart-axis-label" x={DETAIL_FRAME.paddingLeft - 8} y={y + 4} textAnchor="end" fill="#94a3b8">
                  {filteredRows[mapped]?.label}
                </text>
              </g>
            );
          })}

          {Array.from({ length: Math.min(nColumns, 6) }, (_, i) => {
            const mapped = nColumns === 1 ? 0 : Math.round((i / Math.max(Math.min(nColumns, 6) - 1, 1)) * (nColumns - 1));
            const x = DETAIL_FRAME.paddingLeft + (mapped + 0.5) * cellWidth;
            return (
              <text key={`section-x-${mapped}`} className="chart-axis-label" x={x} y={DETAIL_FRAME.height - 12} textAnchor="middle" fill="#94a3b8">
                {xLabels[mapped]}
              </text>
            );
          })}

          {hoverCell ? (
            <rect
              x={DETAIL_FRAME.paddingLeft + hoverCell.col * cellWidth}
              y={DETAIL_FRAME.paddingTop + hoverCell.row * cellHeight}
              width={cellWidth}
              height={cellHeight}
              fill="none"
              stroke="rgba(255,255,255,0.85)"
              strokeWidth={1.5}
            />
          ) : null}

          <rect
            x={DETAIL_FRAME.paddingLeft}
            y={DETAIL_FRAME.paddingTop}
            width={plotWidth}
            height={plotHeight}
            fill="none"
            stroke="rgba(148,163,184,0.4)"
            strokeWidth={0.8}
          />

          <rect
            x={colorbarX}
            y={DETAIL_FRAME.paddingTop}
            width={colorbarW}
            height={colorbarH}
            fill="url(#section-colorbar-grad)"
            rx={2}
          />
          <text className="chart-axis-label" x={colorbarX + colorbarW / 2} y={DETAIL_FRAME.paddingTop - 4} textAnchor="middle" fill="#94a3b8">
            {formatValue(valueMax)}
          </text>
          <text className="chart-axis-label" x={colorbarX + colorbarW / 2} y={DETAIL_FRAME.paddingTop + colorbarH + 12} textAnchor="middle" fill="#94a3b8">
            {formatValue(valueMin)}
          </text>

          <text className="chart-axis-title" x={DETAIL_FRAME.paddingLeft + plotWidth / 2} y={DETAIL_FRAME.height - 2} textAnchor="middle">
            Distance along transect
          </text>
          <text
            className="chart-axis-title"
            x={14}
            y={DETAIL_FRAME.paddingTop + plotHeight / 2}
            textAnchor="middle"
            transform={`rotate(-90 14 ${DETAIL_FRAME.paddingTop + plotHeight / 2})`}
          >
            {axisTitle || "Depth (m)"}
          </text>
        </svg>
      </div>
    </div>
  );
}

function HistogramChart({ bins }: { bins: HistogramBin[] }) {
  if (bins.length === 0) {
    return <EmptyState message="No histogram payload is available for this result yet." />;
  }

  const shellRef = useRef<HTMLDivElement | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(bins.length - 1);
  const maxValue = Math.max(...bins.map((bin) => bin.value));
  const plotWidth = DETAIL_FRAME.width - DETAIL_FRAME.paddingLeft - DETAIL_FRAME.paddingRight;
  const plotHeight = DETAIL_FRAME.height - DETAIL_FRAME.paddingTop - DETAIL_FRAME.paddingBottom;
  const barWidth = plotWidth / bins.length;
  const hoveredBin = bins[hoverIndex ?? bins.length - 1];

  return (
    <div className="detail-chart-card">
      <div className="detail-chart-header">
        <div>
          <p className="eyebrow">Distribution</p>
          <h4>Histogram</h4>
        </div>
        <div className="chart-readout">
          <strong>{hoveredBin ? `${hoveredBin.label} · ${formatValue(hoveredBin.value)}` : "Histogram"}</strong>
          <div className="chart-readout-values">
            <span>Peak: {formatValue(maxValue)}</span>
            <span>Bins: {bins.length}</span>
          </div>
        </div>
      </div>

      <div
        ref={shellRef}
        className="detail-chart-shell"
        onMouseLeave={() => setHoverIndex(bins.length - 1)}
        onMouseMove={(event) => {
          const rect = shellRef.current?.getBoundingClientRect();
          if (!rect) {
            return;
          }
          const relativeX =
            ((event.clientX - rect.left) / Math.max(rect.width, 1)) * DETAIL_FRAME.width - DETAIL_FRAME.paddingLeft;
          const index = clamp(Math.floor(relativeX / Math.max(barWidth, 1)), 0, bins.length - 1);
          setHoverIndex(index);
        }}
      >
        <svg viewBox={`0 0 ${DETAIL_FRAME.width} ${DETAIL_FRAME.height}`} aria-hidden="true">
          <defs>
            <linearGradient id="bar-gradient" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#4f7df3" />
              <stop offset="100%" stopColor="#7ea6f3" />
            </linearGradient>
            <linearGradient id="bar-gradient-hover" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#3a5fd6" />
              <stop offset="100%" stopColor="#4f7df3" />
            </linearGradient>
          </defs>

          {Array.from({ length: 5 }, (_, index) => index).map((index) => {
            const value = (1 - index / 4) * maxValue;
            const y = DETAIL_FRAME.paddingTop + (index / 4) * plotHeight;
            return (
              <g key={value}>
                <line className="chart-grid-line" x1={DETAIL_FRAME.paddingLeft} x2={DETAIL_FRAME.width - DETAIL_FRAME.paddingRight} y1={y} y2={y} />
                <text className="chart-axis-label" x={DETAIL_FRAME.paddingLeft - 12} y={y + 4} textAnchor="end">
                  {formatValue(value)}
                </text>
              </g>
            );
          })}

          {bins.map((bin, index) => {
            const height = (bin.value / Math.max(maxValue, 1e-6)) * plotHeight;
            const x = DETAIL_FRAME.paddingLeft + index * barWidth + 4;
            const y = DETAIL_FRAME.height - DETAIL_FRAME.paddingBottom - height;
            const isHovered = hoverIndex === index;
            return (
              <g key={bin.label}>
                <rect
                  x={x}
                  y={y}
                  width={Math.max(barWidth - 8, 6)}
                  height={Math.max(height, 2)}
                  rx={4}
                  fill={isHovered ? "url(#bar-gradient-hover)" : "url(#bar-gradient)"}
                  opacity={isHovered ? 1 : 0.78}
                />
                <text
                  className="chart-axis-label"
                  x={x + Math.max(barWidth - 8, 6) / 2}
                  y={DETAIL_FRAME.height - 12}
                  textAnchor="middle"
                >
                  {bin.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

function TsDiagramChart({
  points,
  temperatureLabel = "Temperature",
  salinityLabel = "Salinity",
  colorLabel,
  colorRange,
  classColorMap,
  watermassBins,
}: {
  points: TsDiagramPoint[];
  temperatureLabel?: string;
  salinityLabel?: string;
  colorLabel?: string | null;
  colorRange?: number[] | null;
  classColorMap?: Record<string, string>;
  watermassBins?: TsDiagramWatermassBin[];
}) {
  if (points.length === 0) {
    return <EmptyState message="No T-S diagram payload is available for this result yet." />;
  }

  const shellRef = useRef<HTMLDivElement | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(points.length - 1);

  const salinityValues = points.map((point) => point.salinity);
  const temperatureValues = points.map((point) => point.temperature);
  const colorValues = points
    .map((point) => point.colorValue)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));

  const salinityMin = Math.min(...salinityValues);
  const salinityMax = Math.max(...salinityValues);
  const temperatureMin = Math.min(...temperatureValues);
  const temperatureMax = Math.max(...temperatureValues);
  const salinityPad = salinityMax === salinityMin ? Math.max(Math.abs(salinityMax) * 0.02, 0.1) : (salinityMax - salinityMin) * 0.08;
  const temperaturePad =
    temperatureMax === temperatureMin ? Math.max(Math.abs(temperatureMax) * 0.02, 0.1) : (temperatureMax - temperatureMin) * 0.08;

  const xMin = salinityMin - salinityPad;
  const xMax = salinityMax + salinityPad;
  const yMin = temperatureMin - temperaturePad;
  const yMax = temperatureMax + temperaturePad;
  const plotWidth = DETAIL_FRAME.width - DETAIL_FRAME.paddingLeft - DETAIL_FRAME.paddingRight;
  const plotHeight = DETAIL_FRAME.height - DETAIL_FRAME.paddingTop - DETAIL_FRAME.paddingBottom;
  const xForValue = (value: number) =>
    DETAIL_FRAME.paddingLeft + ((value - xMin) / Math.max(xMax - xMin, 1e-6)) * plotWidth;
  const yForValue = (value: number) =>
    DETAIL_FRAME.paddingTop + ((yMax - value) / Math.max(yMax - yMin, 1e-6)) * plotHeight;

  const derivedColorRange =
    colorValues.length > 0 ? [Math.min(...colorValues), Math.max(...colorValues)] : null;
  const activeColorRange =
    colorRange && colorRange.length === 2 && Number.isFinite(colorRange[0]) && Number.isFinite(colorRange[1])
      ? colorRange
      : derivedColorRange;
  const pointRadius = points.length > 1800 ? 1.6 : points.length > 800 ? 2 : 2.4;
  const hoveredPoint = points[hoverIndex ?? points.length - 1];
  const colorByClass = classColorMap ?? {};
  const classLabelById = new Map(
    (watermassBins ?? []).map((item) => [item.id, item.short_name || item.name || item.id]),
  );

  const pointFill = (point: TsDiagramPoint) => {
    if (point.pointClass && colorByClass[point.pointClass]) {
      return colorByClass[point.pointClass];
    }
    if (!activeColorRange || typeof point.colorValue !== "number") {
      return "rgba(79, 125, 243, 0.72)";
    }
    const range = Math.max(activeColorRange[1] - activeColorRange[0], 1e-9);
    const [red, green, blue] = colormapRGB((point.colorValue - activeColorRange[0]) / range);
    return `rgba(${red}, ${green}, ${blue}, 0.72)`;
  };
  const watermassRangeLabel = (bin: TsDiagramWatermassBin) => {
    const parts = [
      Array.isArray(bin.temp_range) && bin.temp_range.length === 2
        ? `T ${formatValue(bin.temp_range[0])}–${formatValue(bin.temp_range[1])}`
        : null,
      Array.isArray(bin.salt_range) && bin.salt_range.length === 2
        ? `S ${formatValue(bin.salt_range[0])}–${formatValue(bin.salt_range[1])}`
        : null,
    ].filter((item): item is string => Boolean(item));
    return parts.join(" · ");
  };

  return (
    <div className="detail-chart-card">
      <div className="detail-chart-header">
        <div>
          <p className="eyebrow">Water-Mass Structure</p>
          <h4>Temperature-Salinity Diagram</h4>
        </div>
        <div className="chart-readout">
          <strong>
            S {formatValue(hoveredPoint.salinity)} · T {formatValue(hoveredPoint.temperature)}
          </strong>
          <div className="chart-readout-values">
            <span>Points: {points.length}</span>
            {hoveredPoint?.pointClass ? (
              <span>
                Class: {classLabelById.get(hoveredPoint.pointClass) ?? hoveredPoint.pointClass}
              </span>
            ) : null}
            {typeof hoveredPoint.colorValue === "number" && colorLabel ? (
              <span>
                {colorLabel}: {formatValue(hoveredPoint.colorValue)}
              </span>
            ) : null}
          </div>
        </div>
      </div>

      <div
        ref={shellRef}
        className="detail-chart-shell"
        onMouseLeave={() => setHoverIndex(points.length - 1)}
        onMouseMove={(event) => {
          const rect = shellRef.current?.getBoundingClientRect();
          if (!rect) {
            return;
          }
          const svgX =
            ((event.clientX - rect.left) / Math.max(rect.width, 1)) * DETAIL_FRAME.width;
          const svgY =
            ((event.clientY - rect.top) / Math.max(rect.height, 1)) * DETAIL_FRAME.height;

          let nearestIndex = 0;
          let nearestDistance = Number.POSITIVE_INFINITY;
          for (let index = 0; index < points.length; index += 1) {
            const dx = xForValue(points[index].salinity) - svgX;
            const dy = yForValue(points[index].temperature) - svgY;
            const distance = dx * dx + dy * dy;
            if (distance < nearestDistance) {
              nearestDistance = distance;
              nearestIndex = index;
            }
          }
          setHoverIndex(nearestIndex);
        }}
      >
        <svg viewBox={`0 0 ${DETAIL_FRAME.width} ${DETAIL_FRAME.height}`} aria-hidden="true">
          {Array.from({ length: 5 }, (_, index) => index).map((index) => {
            const salinity = xMin + (index / 4) * (xMax - xMin);
            const x = xForValue(salinity);
            return (
              <g key={`x-${salinity}`}>
                <line
                  className="chart-grid-line"
                  x1={x}
                  x2={x}
                  y1={DETAIL_FRAME.paddingTop}
                  y2={DETAIL_FRAME.height - DETAIL_FRAME.paddingBottom}
                />
                <text className="chart-axis-label" x={x} y={DETAIL_FRAME.height - 12} textAnchor="middle">
                  {formatValue(salinity)}
                </text>
              </g>
            );
          })}

          {Array.from({ length: 5 }, (_, index) => index).map((index) => {
            const temperature = yMax - (index / 4) * (yMax - yMin);
            const y = yForValue(temperature);
            return (
              <g key={`y-${temperature}`}>
                <line
                  className="chart-grid-line"
                  x1={DETAIL_FRAME.paddingLeft}
                  x2={DETAIL_FRAME.width - DETAIL_FRAME.paddingRight}
                  y1={y}
                  y2={y}
                />
                <text className="chart-axis-label" x={DETAIL_FRAME.paddingLeft - 12} y={y + 4} textAnchor="end">
                  {formatValue(temperature)}
                </text>
              </g>
            );
          })}

          <rect
            x={DETAIL_FRAME.paddingLeft}
            y={DETAIL_FRAME.paddingTop}
            width={plotWidth}
            height={plotHeight}
            rx={18}
            fill="rgba(255, 255, 255, 0.72)"
            stroke="rgba(124, 154, 194, 0.18)"
          />

          {(watermassBins ?? []).map((bin) => {
            const tempRange = Array.isArray(bin.temp_range) && bin.temp_range.length === 2 ? bin.temp_range : null;
            const saltRange = Array.isArray(bin.salt_range) && bin.salt_range.length === 2 ? bin.salt_range : null;
            if (!tempRange || !saltRange) {
              return null;
            }
            const left = xForValue(Math.max(Math.min(...saltRange), xMin));
            const right = xForValue(Math.min(Math.max(...saltRange), xMax));
            const top = yForValue(Math.min(Math.max(...tempRange), yMax));
            const bottom = yForValue(Math.max(Math.min(...tempRange), yMin));
            const width = Math.max(right - left, 1);
            const height = Math.max(bottom - top, 1);
            return (
              <g key={bin.id}>
                <rect
                  x={left}
                  y={top}
                  width={width}
                  height={height}
                  fill="none"
                  stroke={bin.color}
                  strokeWidth={1.4}
                  strokeDasharray="5 4"
                  rx={10}
                />
                <text
                  className="chart-axis-label"
                  x={left + 6}
                  y={top + 14}
                  textAnchor="start"
                  fill={bin.color}
                >
                  {bin.short_name || bin.name}
                </text>
              </g>
            );
          })}

          {points.map((point, index) => (
            <circle
              key={`${point.salinity}-${point.temperature}-${index}`}
              cx={xForValue(point.salinity)}
              cy={yForValue(point.temperature)}
              r={hoverIndex === index ? pointRadius + 1.2 : pointRadius}
              fill={pointFill(point)}
              opacity={hoverIndex === index ? 0.95 : 0.7}
            />
          ))}

          {hoveredPoint ? (
            <>
              <line
                x1={xForValue(hoveredPoint.salinity)}
                x2={xForValue(hoveredPoint.salinity)}
                y1={DETAIL_FRAME.paddingTop}
                y2={DETAIL_FRAME.height - DETAIL_FRAME.paddingBottom}
                stroke="rgba(31, 48, 79, 0.28)"
                strokeDasharray="5 5"
              />
              <line
                x1={DETAIL_FRAME.paddingLeft}
                x2={DETAIL_FRAME.width - DETAIL_FRAME.paddingRight}
                y1={yForValue(hoveredPoint.temperature)}
                y2={yForValue(hoveredPoint.temperature)}
                stroke="rgba(31, 48, 79, 0.28)"
                strokeDasharray="5 5"
              />
            </>
          ) : null}

          <text
            className="chart-axis-title"
            x={DETAIL_FRAME.paddingLeft + plotWidth / 2}
            y={DETAIL_FRAME.height - 2}
            textAnchor="middle"
          >
            {salinityLabel}
          </text>
          <text
            className="chart-axis-title"
            transform={`translate(18 ${DETAIL_FRAME.paddingTop + plotHeight / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            {temperatureLabel}
          </text>
        </svg>
      </div>
      {(watermassBins ?? []).length > 0 ? (
        <div className="map-preview-meta">
          {(watermassBins ?? []).map((bin) => (
            <span key={bin.id} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: 999,
                  background: bin.color,
                  display: "inline-block",
                }}
              />
              {bin.short_name || bin.name}
              {watermassRangeLabel(bin) ? ` (${watermassRangeLabel(bin)})` : ""}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function EofChart({
  variance,
  pcs,
  modes,
  onShowOnMap,
}: {
  variance: ResultCardSummary["metrics"];
  pcs: EofPoint[];
  modes: EofModePreview[];
  onShowOnMap?: (field: MapFieldData) => void;
}) {
  if (variance.length === 0 && pcs.length === 0 && modes.length === 0) {
    return <EmptyState message="No EOF payload is available for this result yet." />;
  }

  const palette = ["#dd8f15", "#4f7df3", "#34c88a"];
  const pcGroups: LineSeriesGroup[] =
    modes.length > 0
      ? modes
          .filter((mode) => mode.pcSeries.length > 0)
          .map((mode, index) => ({
            id: mode.id,
            label: mode.title.replace("EOF ", "PC "),
            color: palette[index % palette.length],
            fill: index === 0 ? "rgba(221, 143, 21, 0.12)" : undefined,
            data: mode.pcSeries,
          }))
      : [
          {
            id: "pc_1",
            label: "PC 1",
            color: palette[0],
            fill: "rgba(221, 143, 21, 0.12)",
            data: pcs.map((point) => ({ label: point.day, value: point.value })),
          },
        ];

  return (
    <div className="eof-panel">
      <div className="variance-grid" style={{ gridTemplateColumns: `repeat(${Math.max(variance.length, 1)}, minmax(0, 1fr))` }}>
        {variance.map((item) => (
          <div key={item.label} className="metric-card">
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>
      {modes.length > 0 ? (
        <div className="eof-mode-grid">
          {modes.map((mode) => (
            <SpatialFieldPreview
              key={mode.id}
              field={mode.mapField}
              compact
              onShowOnMap={onShowOnMap}
            />
          ))}
        </div>
      ) : null}
      <InteractiveLineChart
        emptyMessage="No EOF PC payload is available for this result yet."
        groups={pcGroups}
        title="Principal Components"
      />
    </div>
  );
}

function EventSummaryPanel({
  events,
  activeEventId,
  onSelectEvent
}: {
  events: EventOverlay[];
  activeEventId?: string | null;
  onSelectEvent?: (eventId: string) => void;
}) {
  if (events.length === 0) {
    return <EmptyState message="No event footprints are attached to this result yet." />;
  }

  const selectedEvent = events.find((event) => event.id === activeEventId) ?? events[0];
  const sampleEvents = events.filter((event) => event.id !== selectedEvent.id).slice(0, 3);

  return (
    <div className="event-catalog-card">
      <div className="detail-chart-header">
        <div>
          <p className="eyebrow">Event Summary</p>
          <h4>{events.length} detected events</h4>
        </div>
        <div className="chart-readout">
          <strong>{selectedEvent.title}</strong>
          <div className="chart-readout-values">
            <span>Map is the primary event view</span>
          </div>
        </div>
      </div>

      <div className="event-summary-lead">
        <strong>{selectedEvent.title}</strong>
        <span>{selectedEvent.timestamp ?? selectedEvent.endTimestamp ?? selectedEvent.eventType}</span>
        <div className="event-catalog-meta">
          <span>
            {selectedEvent.center.lon.toFixed(2)}E, {selectedEvent.center.lat.toFixed(2)}N
          </span>
          {selectedEvent.severity ? <span>{selectedEvent.severity}</span> : null}
        </div>
        <div className="event-catalog-details">
          {selectedEvent.details.slice(0, 6).map((detail) => (
            <span key={detail}>{detail}</span>
          ))}
        </div>
      </div>

      {sampleEvents.length > 0 ? (
        <div className="event-summary-grid">
          {sampleEvents.map((event) => (
            <button
              key={event.id}
              className={`event-catalog-item ${event.id === activeEventId ? "is-active" : ""}`}
              onClick={() => onSelectEvent?.(event.id)}
              type="button"
            >
              <div className="event-catalog-header">
                <strong>{event.title}</strong>
                <span>{event.timestamp ?? event.endTimestamp ?? event.eventType}</span>
              </div>
              <div className="event-catalog-meta">
                <span>
                  {event.center.lon.toFixed(2)}E, {event.center.lat.toFixed(2)}N
                </span>
                {event.severity ? <span>{event.severity}</span> : null}
              </div>
              <div className="event-catalog-details">
                {event.details.slice(0, 3).map((detail) => (
                  <span key={detail}>{detail}</span>
                ))}
              </div>
            </button>
          ))}
        </div>
      ) : null}

      <p className="event-summary-footnote">
        Click a map event to inspect more details. Representative events shown here are sampled from the full set.
      </p>
    </div>
  );
}

function ResultNarrative({ card }: { card: ResultCardSummary }) {
  return (
    <div className="result-hero-card">
      <div className="result-hero-header">
        <div>
          <p className="result-kicker">{card.type}</p>
          <h4>{card.title}</h4>
        </div>
        <div className="result-hero-badge">{card.renderer}</div>
      </div>
      <p className="result-hero-headline">{card.headline}</p>
      <p className="result-hero-description">{card.description}</p>
      <div className="metric-pill-grid">
        {card.metrics.map((metric) => (
          <div key={metric.label} className="metric-pill">
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
          </div>
        ))}
      </div>
      {card.detailSections && card.detailSections.length > 0 ? (
        <div className="result-detail-section-list">
          {card.detailSections.map((section) => (
            <div key={`${card.id}-${section.title}`} className="result-detail-section">
              <span className="mini-label">{section.title}</span>
              {section.items.map((item) => (
                <p key={`${card.id}-${section.title}-${item}`}>{item}</p>
              ))}
            </div>
          ))}
        </div>
      ) : null}
      {card.interpretation ? (
        <div className="result-detail-section">
          <span className="mini-label">Interpretation</span>
          <p>{card.interpretation}</p>
        </div>
      ) : null}
    </div>
  );
}

function ResultSplit({
  card,
  visual,
  showSummary = true,
}: {
  card: ResultCardSummary;
  visual?: ReactNode;
  showSummary?: boolean;
}) {
  return (
    <div className={`dock-panel dock-panel-split ${showSummary ? "" : "is-visual-only"}`}>
      {showSummary ? (
        <div className="result-summary-pane">
          <ResultNarrative card={card} />
        </div>
      ) : null}
      <div className={`result-visual-pane ${visual ? "" : "is-empty"}`}>{visual ?? null}</div>
    </div>
  );
}

function ResultSummaryOnly({ card }: { card: ResultCardSummary }) {
  return (
    <div className="dock-panel">
      <ResultNarrative card={card} />
    </div>
  );
}

function ResultSummaryWithChart({
  card,
  groups,
}: {
  card: ResultCardSummary;
  groups: LineSeriesGroup[];
}) {
  return (
    <ResultSplit
      card={card}
      visual={
        <InteractiveLineChart
          emptyMessage="No grouped event statistics are available for this result yet."
          groups={groups}
          title={card.title}
        />
      }
    />
  );
}

export function renderDockPanel(
  activeResult: ResultCardSummary,
  workspaceData: WorkspaceData,
  options?: {
    activeEventId?: string | null;
    onSelectEvent?: (eventId: string) => void;
    onPromoteMapField?: (field: MapFieldData) => void;
    showSummary?: boolean;
  }
) {
  const showSummary = options?.showSummary ?? true;
  switch (activeResult.renderer) {
    case "ts_diagram":
      if (workspaceData.tsDiagramPoints.length === 0) {
        return <ResultSplit card={activeResult} showSummary={showSummary} />;
      }
      return (
        <ResultSplit
          card={activeResult}
          showSummary={showSummary}
          visual={
              <TsDiagramChart
                points={workspaceData.tsDiagramPoints}
                temperatureLabel={workspaceData.tsDiagramTemperatureLabel}
                salinityLabel={workspaceData.tsDiagramSalinityLabel}
                colorLabel={workspaceData.tsDiagramColorLabel}
                colorRange={workspaceData.tsDiagramColorRange}
                classColorMap={workspaceData.tsDiagramClassColorMap}
                watermassBins={workspaceData.tsDiagramWatermassBins}
              />
            }
          />
        );
    case "profile":
      if (workspaceData.profileSeries.length === 0) {
        return <ResultSplit card={activeResult} showSummary={showSummary} />;
      }
      return (
        <ResultSplit
          card={activeResult}
          showSummary={showSummary}
          visual={<ProfileChart series={workspaceData.profileSeries} markers={workspaceData.profileMarkers} />}
        />
      );
    case "hovmoller":
      if (workspaceData.hovmollerRows.length === 0) {
        return <ResultSplit card={activeResult} showSummary={showSummary} />;
      }
      return (
        <ResultSplit
          card={activeResult}
          showSummary={showSummary}
          visual={
            <HovmollerChart
              rows={workspaceData.hovmollerRows}
              displayInfo={workspaceData.hovmollerDisplayInfo}
              overlay={workspaceData.overlaySeries}
              timeLabels={workspaceData.hovmollerTimeLabels}
              depthIntegratedSeries={workspaceData.hovmollerDepthIntegratedSeries}
            />
          }
        />
      );
    case "section":
      if (workspaceData.sectionRows.length === 0) {
        return <ResultSplit card={activeResult} showSummary={showSummary} />;
      }
      return (
        <ResultSplit
          card={activeResult}
          showSummary={showSummary}
          visual={
            <SectionChart
              rows={workspaceData.sectionRows}
              distanceKm={workspaceData.sectionDistanceKm}
              axisTitle={workspaceData.sectionAxisTitle}
              sliceLabel={workspaceData.sectionSliceLabel}
            />
          }
        />
      );
    case "histogram":
      if (workspaceData.histogramBins.length === 0) {
        return <ResultSplit card={activeResult} showSummary={showSummary} />;
      }
      return (
        <ResultSplit
          card={activeResult}
          showSummary={showSummary}
          visual={<HistogramChart bins={workspaceData.histogramBins} />}
        />
      );
    case "eof":
      if (
        workspaceData.eofVariance.length === 0 &&
        workspaceData.eofPcSeries.length === 0 &&
        workspaceData.eofModes.length === 0
      ) {
        return <ResultSplit card={activeResult} showSummary={showSummary} />;
      }
      return (
        <ResultSplit
          card={activeResult}
          showSummary={showSummary}
          visual={
            <EofChart
              variance={workspaceData.eofVariance}
              pcs={workspaceData.eofPcSeries}
              modes={workspaceData.eofModes}
              onShowOnMap={options?.onPromoteMapField}
            />
          }
        />
      );
    case "composite":
      if (workspaceData.compositeFields.length === 0) {
        return <ResultSplit card={activeResult} showSummary={showSummary} />;
      }
      return (
        <ResultSplit
          card={activeResult}
          showSummary={showSummary}
          visual={
            <CompositeFieldPanel
              fields={workspaceData.compositeFields}
              onShowOnMap={options?.onPromoteMapField}
            />
          }
        />
      );
    case "summary":
      {
        const lineGroups = buildLineSeries(workspaceData);
        if (lineGroups.length > 0) {
          return <ResultSummaryWithChart card={activeResult} groups={lineGroups} />;
        }
        if (workspaceData.mapField?.subregionGrid) {
          return (
            <ResultSplit
              card={activeResult}
              showSummary={showSummary}
              visual={<SubregionDiagnosisPanel field={workspaceData.mapField} />}
            />
          );
        }
        if (workspaceData.mapField) {
          return (
            <ResultSplit
              card={activeResult}
              showSummary={showSummary}
              visual={<SpatialFieldPreview field={workspaceData.mapField} onShowOnMap={options?.onPromoteMapField} />}
            />
          );
        }
        return <ResultSummaryOnly card={activeResult} />;
      }
    case "reference":
      return (
        <ResultSplit
          card={activeResult}
          showSummary={showSummary}
          visual={
            workspaceData.mapField ? (
              <SpatialFieldPreview field={workspaceData.mapField} onShowOnMap={options?.onPromoteMapField} />
            ) : undefined
          }
        />
      );
    case "event":
      return (
        <ResultSplit
          card={activeResult}
          showSummary={showSummary}
          visual={
            <EventSummaryPanel
              activeEventId={options?.activeEventId}
              events={workspaceData.eventOverlays}
              onSelectEvent={options?.onSelectEvent}
            />
          }
        />
      );
    case "timeseries":
    default: {
      const lineGroups = buildLineSeries(workspaceData);
      if (lineGroups.length === 0) {
        if (hasNoFiniteTimeseriesValues(activeResult, workspaceData)) {
          return (
            <ResultSplit
              card={activeResult}
              showSummary={showSummary}
              visual={<EmptyState message={NO_FINITE_TIMESERIES_MESSAGE} />}
            />
          );
        }
        return <ResultSplit card={activeResult} showSummary={showSummary} />;
      }
      return (
        <ResultSplit
          card={activeResult}
          showSummary={showSummary}
          visual={
            <InteractiveLineChart
              emptyMessage="No time-series payload is available for this result yet."
              groups={lineGroups}
              subtitle={timeseriesDisplaySubtitle(workspaceData.timeseriesDisplayInfo)}
              title={activeResult.title}
            />
          }
        />
      );
    }
  }
}

export function renderInlineChart(
  card: ResultCardSummary,
  data: WorkspaceData,
  options?: { onPromoteMapField?: (field: MapFieldData) => void },
): ReactNode {
  if (card.type === "mechanism_score_result" && data.mapField?.subregionGrid) {
    return <SubregionDiagnosisPanel field={data.mapField} compact />;
  }
  if (card.renderer === "summary" && data.mapField) {
    return <SpatialFieldPreview field={data.mapField} compact onShowOnMap={options?.onPromoteMapField} />;
  }
  if (card.surface === "map" && data.mapField) {
    return <SpatialFieldPreview field={data.mapField} compact onShowOnMap={options?.onPromoteMapField} />;
  }
  if (card.renderer === "timeseries" && data.resultSeries.length > 0) {
    return (
      <InteractiveLineChart
        emptyMessage="No time-series payload is available for this result yet."
        groups={buildLineSeries(data)}
        subtitle={timeseriesDisplaySubtitle(data.timeseriesDisplayInfo)}
        title={card.title}
      />
    );
  }
  if (hasNoFiniteTimeseriesValues(card, data)) {
    return <EmptyState message={NO_FINITE_TIMESERIES_MESSAGE} />;
  }
  if (card.renderer === "ts_diagram" && data.tsDiagramPoints.length > 0) {
    return (
      <TsDiagramChart
        points={data.tsDiagramPoints}
        temperatureLabel={data.tsDiagramTemperatureLabel}
        salinityLabel={data.tsDiagramSalinityLabel}
        colorLabel={data.tsDiagramColorLabel}
        colorRange={data.tsDiagramColorRange}
        classColorMap={data.tsDiagramClassColorMap}
        watermassBins={data.tsDiagramWatermassBins}
      />
    );
  }
  if (card.renderer === "hovmoller" && data.hovmollerRows.length > 0) {
    return (
      <HovmollerChart
        rows={data.hovmollerRows}
        displayInfo={data.hovmollerDisplayInfo}
        overlay={data.overlaySeries}
        timeLabels={data.hovmollerTimeLabels}
        depthIntegratedSeries={data.hovmollerDepthIntegratedSeries}
      />
    );
  }
  if (card.renderer === "histogram" && data.histogramBins.length > 0) {
    return <HistogramChart bins={data.histogramBins} />;
  }
  if (
    card.renderer === "eof" &&
    (data.eofVariance.length > 0 || data.eofPcSeries.length > 0 || data.eofModes.length > 0)
  ) {
    return (
      <EofChart
        variance={data.eofVariance}
        pcs={data.eofPcSeries}
        modes={data.eofModes}
        onShowOnMap={options?.onPromoteMapField}
      />
    );
  }
  if (card.renderer === "composite" && data.compositeFields.length > 0) {
    return <CompositeFieldPanel fields={data.compositeFields} compact />;
  }
  if (card.renderer === "event" && data.eventOverlays.length > 0) {
    return <div className="inline-stat">{data.eventOverlays.length} events detected</div>;
  }
  return null;
}
