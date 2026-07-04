"use client";

import { useEffect, useState } from "react";
import { gridCellStrokeColor, sortSubregionCells } from "./subregion-grid";
import type { MapColorScale, MapFieldData, TransportFilledRegionData } from "./types";

const WEB_MERCATOR_MAX_LAT = 85.05112878;
const DEFAULT_CONTINUOUS_COLORMAP = "ocean_diverging";
const TRANSPORT_FILLED_LEVELS = 22;
const TRANSPORT_FILLED_ALPHA = 255;

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function latToWebMercatorY(lat: number) {
  const clipped = clamp(lat, -WEB_MERCATOR_MAX_LAT, WEB_MERCATOR_MAX_LAT);
  const radians = (clipped * Math.PI) / 180;
  return Math.log(Math.tan(Math.PI / 4 + radians / 2));
}

function webMercatorYToLat(y: number) {
  return ((2 * Math.atan(Math.exp(y)) - Math.PI / 2) * 180) / Math.PI;
}

function fieldExtent(field: MapFieldData) {
  if (field.bounds && field.bounds.length === 2) {
    return {
      latMin: Math.min(field.bounds[0][0], field.bounds[1][0]),
      latMax: Math.max(field.bounds[0][0], field.bounds[1][0]),
      lonMin: Math.min(field.bounds[0][1], field.bounds[1][1]),
      lonMax: Math.max(field.bounds[0][1], field.bounds[1][1]),
    };
  }

  return {
    latMin: Math.min(...field.lat),
    latMax: Math.max(...field.lat),
    lonMin: Math.min(...field.lon),
    lonMax: Math.max(...field.lon),
  };
}

const OCEAN_DIVERGING_COLORMAP_STOPS: [number, [number, number, number]][] = [
  [0.0, [112, 0, 168]],
  [0.125, [88, 42, 214]],
  [0.25, [35, 98, 221]],
  [0.375, [37, 185, 225]],
  [0.5, [244, 255, 246]],
  [0.625, [255, 244, 72]],
  [0.75, [255, 177, 45]],
  [0.875, [229, 72, 30]],
  [1.0, [164, 24, 12]],
];

const LEGACY_RDYLBUR_COLORMAP_STOPS: [number, [number, number, number]][] = [
  [0.0, [49, 54, 149]],
  [0.125, [69, 117, 180]],
  [0.25, [116, 173, 209]],
  [0.375, [171, 217, 233]],
  [0.5, [255, 255, 191]],
  [0.625, [254, 224, 144]],
  [0.75, [253, 174, 97]],
  [0.875, [244, 109, 67]],
  [1.0, [165, 0, 38]],
];

const TRANSPORT_DIVERGING_COLORMAP_STOPS: [number, [number, number, number]][] = [
  [0.0, [82, 31, 143]],
  [0.18, [128, 80, 174]],
  [0.34, [190, 174, 214]],
  [0.5, [247, 247, 244]],
  [0.66, [253, 213, 118]],
  [0.82, [239, 138, 58]],
  [1.0, [179, 55, 33]],
];

const TRANSPORT_BLUE_RED_COLORMAP_STOPS: [number, [number, number, number]][] = [
  [0.0, [33, 102, 172]],
  [0.18, [67, 147, 195]],
  [0.34, [146, 197, 222]],
  [0.5, [247, 247, 247]],
  [0.66, [244, 165, 130]],
  [0.82, [214, 96, 77]],
  [1.0, [178, 24, 43]],
];

const TRANSPORT_GREEN_PURPLE_COLORMAP_STOPS: [number, [number, number, number]][] = [
  [0.0, [0, 104, 55]],
  [0.18, [49, 163, 84]],
  [0.34, [166, 217, 106]],
  [0.5, [247, 247, 247]],
  [0.66, [194, 165, 207]],
  [0.82, [123, 50, 148]],
  [1.0, [76, 0, 109]],
];

function colormapStops(colormap?: string) {
  const key = String(colormap ?? DEFAULT_CONTINUOUS_COLORMAP).trim().toLowerCase();
  if (key === "blue_white_red" || key === "transport_blue_red") {
    return TRANSPORT_BLUE_RED_COLORMAP_STOPS;
  }
  if (key === "transport_green_purple") {
    return TRANSPORT_GREEN_PURPLE_COLORMAP_STOPS;
  }
  if (key === "transport_diverging") {
    return TRANSPORT_DIVERGING_COLORMAP_STOPS;
  }
  if (key === "rdylbu_r") {
    return LEGACY_RDYLBUR_COLORMAP_STOPS;
  }
  return OCEAN_DIVERGING_COLORMAP_STOPS;
}

export function colormapRGB(t: number, colormap = "ocean_diverging"): [number, number, number] {
  const stops = colormapStops(colormap);
  const clamped = clamp(t, 0, 1);
  for (let i = 0; i < stops.length - 1; i += 1) {
    const [t0, rgb0] = stops[i];
    const [t1, rgb1] = stops[i + 1];
    if (clamped <= t1) {
      const factor = (clamped - t0) / (t1 - t0);
      return [
        Math.round(rgb0[0] + (rgb1[0] - rgb0[0]) * factor),
        Math.round(rgb0[1] + (rgb1[1] - rgb0[1]) * factor),
        Math.round(rgb0[2] + (rgb1[2] - rgb0[2]) * factor),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

function quantile(sortedValues: number[], q: number) {
  if (sortedValues.length === 0) {
    return Number.NaN;
  }
  if (sortedValues.length === 1) {
    return sortedValues[0];
  }
  const position = clamp(q, 0, 1) * (sortedValues.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) {
    return sortedValues[lower];
  }
  const weight = position - lower;
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
}

export function formatMapColorbarValue(value: number) {
  if (!Number.isFinite(value)) {
    return "NA";
  }
  const magnitude = Math.abs(value);
  if (magnitude === 0) {
    return "0";
  }
  if (magnitude < 1e-3 || magnitude >= 1e4) {
    return value.toExponential(2);
  }
  if (magnitude < 1) {
    return value.toFixed(4).replace(/\.?0+$/, "");
  }
  return value.toFixed(3).replace(/\.?0+$/, "");
}

export function mapColorbarGradient(stops = 9, colormap = DEFAULT_CONTINUOUS_COLORMAP) {
  const count = Math.max(2, stops);
  return Array.from({ length: count }, (_, index) => {
    const t = index / (count - 1);
    const [red, green, blue] = colormapRGB(t, colormap);
    return `rgb(${red}, ${green}, ${blue}) ${Math.round(t * 100)}%`;
  }).join(", ");
}

export function mapFieldTileKind(field: MapFieldData | null | undefined) {
  return String(field?.tileMapKind ?? "").trim().toLowerCase();
}

export function resolveMapColorScale(field: MapFieldData | null | undefined): MapColorScale | null {
  const tileMapKind = mapFieldTileKind(field);
  if (
    !field ||
    (
      tileMapKind !== "event_hotspot" &&
      (
        field.subregionGrid ||
        (Array.isArray(field.discreteLegend) && field.discreteLegend.length > 0)
      )
    )
  ) {
    return null;
  }

  const allValues = field.values.flat().filter((value) => Number.isFinite(value));
  if (allValues.length === 0) {
    return null;
  }

  const sortedValues = [...allValues].sort((a, b) => a - b);
  const rawMin = sortedValues[0];
  const rawMax = sortedValues[sortedValues.length - 1];
  const provided = field.colorScale;
  if (
    provided &&
    Number.isFinite(provided.min) &&
    Number.isFinite(provided.max)
  ) {
    const scale = normalizeSignedColorScale(
      {
        ...provided,
        rawMin: Number.isFinite(provided.rawMin) ? provided.rawMin : rawMin,
        rawMax: Number.isFinite(provided.rawMax) ? provided.rawMax : rawMax,
        colormap: provided.colormap ?? DEFAULT_CONTINUOUS_COLORMAP,
        units: provided.units ?? field.units ?? "",
        label: provided.label ?? field.label ?? field.variable,
      },
      sortedValues,
    );
    return {
      ...scale,
      colormap: scale.colormap ?? DEFAULT_CONTINUOUS_COLORMAP,
    };
  }

  const robustMin = quantile(sortedValues, 0.02);
  const robustMax = quantile(sortedValues, 0.98);
  const useRobust = Number.isFinite(robustMin) && Number.isFinite(robustMax) && robustMax > robustMin;

  return normalizeSignedColorScale({
    min: useRobust ? robustMin : rawMin,
    max: useRobust ? robustMax : rawMax,
    rawMin,
    rawMax,
    colormap: DEFAULT_CONTINUOUS_COLORMAP,
    units: field.units ?? "",
    label: field.label ?? field.variable,
    scaleStrategy: useRobust ? "p02_p98" : "raw_extent",
  }, sortedValues);
}

function normalizeSignedColorScale(scale: MapColorScale, sortedValues: number[]): MapColorScale {
  const rawMin = Number.isFinite(scale.rawMin) ? Number(scale.rawMin) : sortedValues[0];
  const rawMax = Number.isFinite(scale.rawMax) ? Number(scale.rawMax) : sortedValues[sortedValues.length - 1];
  const crossesZero = rawMin < 0 && rawMax > 0;
  if (!crossesZero && !scale.symmetric) {
    return scale;
  }

  const existingLimit = Math.max(Math.abs(scale.min), Math.abs(scale.max));
  const absoluteValues = sortedValues.map((value) => Math.abs(value)).sort((a, b) => a - b);
  const robustLimit = quantile(absoluteValues, 0.98);
  const rawLimit = Math.max(Math.abs(rawMin), Math.abs(rawMax));
  const limit =
    Number.isFinite(existingLimit) && existingLimit > 0
      ? existingLimit
      : Number.isFinite(robustLimit) && robustLimit > 0
        ? robustLimit
        : rawLimit;

  return {
    ...scale,
    min: -limit,
    max: limit,
    rawMin,
    rawMax,
    symmetric: true,
    scaleStrategy: scale.scaleStrategy ?? "symmetric_p98_abs",
    colormap: scale.colormap ?? DEFAULT_CONTINUOUS_COLORMAP,
  };
}

function hexToRgb(color: string): [number, number, number] | null {
  const match = /^#?([0-9a-f]{6})$/i.exec(String(color ?? "").trim());
  if (!match) {
    return null;
  }
  const hex = match[1];
  return [
    Number.parseInt(hex.slice(0, 2), 16),
    Number.parseInt(hex.slice(2, 4), 16),
    Number.parseInt(hex.slice(4, 6), 16),
  ];
}

function cssRgb(rgb: [number, number, number]) {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function readableContourRgb(rgb: [number, number, number]): [number, number, number] {
  const luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  if (luminance <= 218) {
    return rgb;
  }
  return [
    Math.round(rgb[0] * 0.55),
    Math.round(rgb[1] * 0.55),
    Math.round(rgb[2] * 0.55),
  ];
}

function finiteCellValue(cell: NonNullable<MapFieldData["subregionGrid"]>["cells"][number]) {
  if (typeof cell.value === "number" && Number.isFinite(cell.value)) {
    return cell.value;
  }
  if (typeof cell.dominantScore === "number" && Number.isFinite(cell.dominantScore)) {
    return cell.dominantScore;
  }
  return null;
}

function subregionValueExtent(field: MapFieldData) {
  const values = (field.subregionGrid?.cells ?? [])
    .map((cell) => finiteCellValue(cell))
    .filter((value): value is number => value !== null);
  if (values.length === 0) {
    return null;
  }
  return {
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

function subregionFillOpacity(
  cell: NonNullable<MapFieldData["subregionGrid"]>["cells"][number],
  extent: { min: number; max: number } | null,
) {
  if (String(cell.status ?? "").trim().toLowerCase() !== "ok") {
    return 0.12;
  }
  const value = finiteCellValue(cell);
  if (value === null) {
    return 0.42;
  }
  let strength = 0.65;
  if (extent && extent.max > extent.min) {
    strength = clamp((value - extent.min) / (extent.max - extent.min), 0, 1);
  } else if (value >= 0 && value <= 1) {
    strength = value;
  }
  return 0.22 + strength * 0.56;
}

function renderSubregionGridImage(field: MapFieldData, width: number, height: number) {
  const subregionGrid = field.subregionGrid;
  if (!subregionGrid || subregionGrid.cells.length === 0) {
    return "";
  }

  const bounds = fieldExtent(field);
  const northY = latToWebMercatorY(bounds.latMax);
  const southY = latToWebMercatorY(bounds.latMin);
  const lonSpan = Math.max(bounds.lonMax - bounds.lonMin, 1e-9);
  const mercatorSpan = Math.max(northY - southY, 1e-9);
  const valueExtent = subregionValueExtent(field);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return "";
  }

  let renderedCells = 0;
  sortSubregionCells(subregionGrid.cells).forEach((cell) => {
    if (!cell.bounds) {
      return;
    }
    const rgb = hexToRgb(gridCellStrokeColor(cell));
    if (!rgb) {
      return;
    }
    const left = Math.floor(((cell.bounds.lonMin - bounds.lonMin) / lonSpan) * width);
    const right = Math.ceil(((cell.bounds.lonMax - bounds.lonMin) / lonSpan) * width);
    const top = Math.floor(((northY - latToWebMercatorY(cell.bounds.latMax)) / mercatorSpan) * height);
    const bottom = Math.ceil(((northY - latToWebMercatorY(cell.bounds.latMin)) / mercatorSpan) * height);
    const x = clamp(left, 0, width);
    const y = clamp(top, 0, height);
    const cellWidth = clamp(right, 0, width) - x;
    const cellHeight = clamp(bottom, 0, height) - y;
    if (cellWidth <= 0 || cellHeight <= 0) {
      return;
    }
    ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${subregionFillOpacity(cell, valueExtent)})`;
    ctx.fillRect(x, y, cellWidth, cellHeight);
    renderedCells += 1;
  });

  return renderedCells > 0 ? canvas.toDataURL() : "";
}

export function isTransportStreamfunctionRendering(field: MapFieldData | null | undefined) {
  return String(field?.transportRendering?.mode ?? "").trim().toLowerCase() === "gan_fig10_china_seas";
}

function maskValue(mask: boolean[][] | undefined, row: number, col: number) {
  return Boolean(mask?.[row]?.[col]);
}

export function transportCellRenderMode(field: MapFieldData, row: number, col: number) {
  if (!isTransportStreamfunctionRendering(field)) {
    return "none";
  }
  const filledRegions = Array.isArray(field.transportRendering?.filledRegions)
    ? field.transportRendering.filledRegions
    : [];
  if (filledRegions.some((region) => maskValue(region.mask, row, col))) {
    return "filled";
  }
  if (maskValue(field.transportRendering?.filledMask, row, col)) {
    return "filled";
  }
  if (maskValue(field.transportRendering?.contourMask, row, col)) {
    return "contour";
  }
  return "none";
}

function normalizeTransportRegionKey(value: string | undefined) {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (!normalized) {
    return "";
  }
  if (normalized.includes("wpo") || normalized.includes("global")) {
    return "wpo";
  }
  if (normalized.includes("china") || normalized.includes("cs")) {
    return "china_seas";
  }
  return normalized;
}

function transportScaleMatchesRegion(scale: MapColorScale, regionKey: string) {
  if (!regionKey) {
    return false;
  }
  const text = `${scale.scaleStrategy ?? ""} ${scale.label ?? ""}`.toLowerCase();
  if (regionKey === "wpo") {
    return text.includes("wpo") || text.includes("global");
  }
  if (regionKey === "china_seas") {
    return text.includes("china_seas") || text.includes("china seas") || text.includes("cs regional");
  }
  return text.includes(regionKey);
}

function transportRegionalScale(field: MapFieldData, renderMode: "filled" | "contours", regionKey?: string) {
  const scales = Array.isArray(field.regionalColorScales) ? field.regionalColorScales : [];
  const normalizedRegion = normalizeTransportRegionKey(regionKey);
  if (normalizedRegion) {
    const regionalMatch = scales.find((scale) => (
      scale.renderMode === renderMode && transportScaleMatchesRegion(scale, normalizedRegion)
    )) ?? scales.find((scale) => transportScaleMatchesRegion(scale, normalizedRegion));
    if (regionalMatch) {
      return regionalMatch;
    }
  }
  return scales.find((scale) => scale.renderMode === renderMode)
    ?? scales.find((scale) => String(scale.scaleStrategy ?? "").includes(renderMode === "filled" ? "china_seas" : "wpo"))
    ?? null;
}

export function transportContourColorForLevel(field: MapFieldData, level: number) {
  const scale = transportRegionalScale(field, "contours", "wpo");
  if (
    scale
    && Number.isFinite(scale.min)
    && Number.isFinite(scale.max)
    && Number(scale.max) > Number(scale.min)
  ) {
    const rgb = colormapRGB(
      clamp((level - Number(scale.min)) / (Number(scale.max) - Number(scale.min)), 0, 1),
      scale.colormap ?? DEFAULT_CONTINUOUS_COLORMAP,
    );
    return cssRgb(readableContourRgb(rgb));
  }
  return Math.abs(level) < 1e-12
    ? field.transportRendering?.zeroContourColor || "#0f172a"
    : field.transportRendering?.contourColor || "#1f2937";
}

function canvasPoint(
  field: MapFieldData,
  bounds: ReturnType<typeof fieldExtent>,
  width: number,
  height: number,
  row: number,
  col: number,
) {
  const lonSpan = Math.max(bounds.lonMax - bounds.lonMin, 1e-9);
  const northY = latToWebMercatorY(bounds.latMax);
  const southY = latToWebMercatorY(bounds.latMin);
  const mercatorSpan = Math.max(northY - southY, 1e-9);
  return {
    x: ((field.lon[col] - bounds.lonMin) / lonSpan) * width,
    y: ((northY - latToWebMercatorY(field.lat[row])) / mercatorSpan) * height,
    value: Number(field.values[row]?.[col]),
  };
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

type AxisPosition = { lower: number; upper: number; t: number };
type ContourPoint = { x: number; y: number };
type ContourSegment = { start: ContourPoint; end: ContourPoint };
type ContourPath = ContourPoint[];
type ContourLabelAnchor = {
  level: number;
  label: string;
  x: number;
  y: number;
  angle: number;
  color: string;
};

function pointKey(point: ContourPoint) {
  return `${point.x.toFixed(2)},${point.y.toFixed(2)}`;
}

function segmentLength(segment: ContourSegment) {
  return Math.hypot(segment.end.x - segment.start.x, segment.end.y - segment.start.y);
}

function pathLength(path: ContourPath) {
  let total = 0;
  for (let index = 1; index < path.length; index += 1) {
    total += Math.hypot(path[index].x - path[index - 1].x, path[index].y - path[index - 1].y);
  }
  return total;
}

function stitchContourPaths(segments: ContourSegment[]) {
  const unused = new Set(segments.map((_, index) => index));
  const endpointIndex = new Map<string, number[]>();
  segments.forEach((segment, index) => {
    [pointKey(segment.start), pointKey(segment.end)].forEach((key) => {
      const existing = endpointIndex.get(key) ?? [];
      existing.push(index);
      endpointIndex.set(key, existing);
    });
  });

  const paths: ContourPath[] = [];
  while (unused.size > 0) {
    const firstIndex = unused.values().next().value as number;
    const first = segments[firstIndex];
    unused.delete(firstIndex);
    const path: ContourPath = [first.start, first.end];

    let extended = true;
    while (extended) {
      extended = false;
      const headKey = pointKey(path[0]);
      const tailKey = pointKey(path[path.length - 1]);
      for (const [key, insertAtTail] of [[tailKey, true], [headKey, false]] as const) {
        const candidates = endpointIndex.get(key) ?? [];
        const nextIndex = candidates.find((candidate) => unused.has(candidate));
        if (nextIndex === undefined) {
          continue;
        }
        const next = segments[nextIndex];
        unused.delete(nextIndex);
        const matchesStart = pointKey(next.start) === key;
        const otherPoint = matchesStart ? next.end : next.start;
        if (insertAtTail) {
          path.push(otherPoint);
        } else {
          path.unshift(otherPoint);
        }
        extended = true;
        break;
      }
    }
    paths.push(path);
  }
  return paths;
}

function collectContourSegments(
  field: MapFieldData,
  width: number,
  height: number,
  bounds: ReturnType<typeof fieldExtent>,
  level: number,
  contourMask: boolean[][],
) {
  const segments: ContourSegment[] = [];
  for (let row = 0; row < field.lat.length - 1; row += 1) {
    for (let col = 0; col < field.lon.length - 1; col += 1) {
      if (
        !maskValue(contourMask, row, col)
        || !maskValue(contourMask, row, col + 1)
        || !maskValue(contourMask, row + 1, col + 1)
        || !maskValue(contourMask, row + 1, col)
      ) {
        continue;
      }
      const corners = [
        canvasPoint(field, bounds, width, height, row, col),
        canvasPoint(field, bounds, width, height, row, col + 1),
        canvasPoint(field, bounds, width, height, row + 1, col + 1),
        canvasPoint(field, bounds, width, height, row + 1, col),
      ];
      if (!corners.every((corner) => Number.isFinite(corner.value))) {
        continue;
      }
      const crossings = [
        contourEdgePoint(corners[0], corners[1], level),
        contourEdgePoint(corners[1], corners[2], level),
        contourEdgePoint(corners[2], corners[3], level),
        contourEdgePoint(corners[3], corners[0], level),
      ].filter((point): point is ContourPoint => Boolean(point));
      if (crossings.length === 2) {
        segments.push({ start: crossings[0], end: crossings[1] });
      } else if (crossings.length === 4) {
        segments.push({ start: crossings[0], end: crossings[1] });
        segments.push({ start: crossings[2], end: crossings[3] });
      }
    }
  }
  return segments.filter((segment) => segmentLength(segment) > 0.5);
}

export function transportContourLabelText(level: number) {
  if (Number.isInteger(level)) {
    return String(level);
  }
  return formatMapColorbarValue(level);
}

function axisPosition(values: number[], target: number): AxisPosition | null {
  const length = values.length;
  if (length === 0) {
    return null;
  }
  if (length === 1) {
    return { lower: 0, upper: 0, t: 0 };
  }
  const first = values[0];
  const last = values[length - 1];
  const ascending = last >= first;
  const minValue = ascending ? first : last;
  const maxValue = ascending ? last : first;
  if (target < minValue || target > maxValue) {
    return null;
  }

  let low = 0;
  let high = length - 1;
  while (high - low > 1) {
    const mid = Math.floor((low + high) / 2);
    const value = values[mid];
    if (ascending ? value <= target : value >= target) {
      low = mid;
    } else {
      high = mid;
    }
  }
  const start = values[low];
  const end = values[high];
  const denominator = end - start;
  const t = Math.abs(denominator) < 1e-12 ? 0 : (target - start) / denominator;
  return { lower: low, upper: high, t: clamp(t, 0, 1) };
}

function indexPosition(index: number, length: number): AxisPosition | null {
  if (length <= 0 || !Number.isFinite(index)) {
    return null;
  }
  const clamped = clamp(index, 0, Math.max(0, length - 1));
  const lower = Math.floor(clamped);
  const upper = Math.min(length - 1, Math.ceil(clamped));
  return { lower, upper, t: clamped - lower };
}

function nearestGridIndex(position: AxisPosition) {
  return position.t < 0.5 ? position.lower : position.upper;
}

function gridValue(field: MapFieldData, row: number, col: number) {
  const value = Number(field.values[row]?.[col]);
  return Number.isFinite(value) ? value : null;
}

function interpolateGridValue(
  field: MapFieldData,
  rowPosition: AxisPosition | null,
  colPosition: AxisPosition | null,
) {
  if (!rowPosition || !colPosition) {
    return null;
  }

  const nearestRow = nearestGridIndex(rowPosition);
  const nearestCol = nearestGridIndex(colPosition);
  const nearestValue = gridValue(field, nearestRow, nearestCol);
  if (nearestValue === null) {
    return null;
  }

  const corners = [
    {
      row: rowPosition.lower,
      col: colPosition.lower,
      weight: (1 - rowPosition.t) * (1 - colPosition.t),
    },
    {
      row: rowPosition.lower,
      col: colPosition.upper,
      weight: (1 - rowPosition.t) * colPosition.t,
    },
    {
      row: rowPosition.upper,
      col: colPosition.lower,
      weight: rowPosition.t * (1 - colPosition.t),
    },
    {
      row: rowPosition.upper,
      col: colPosition.upper,
      weight: rowPosition.t * colPosition.t,
    },
  ];

  let weightedSum = 0;
  let weightSum = 0;
  corners.forEach((corner) => {
    const value = gridValue(field, corner.row, corner.col);
    if (corner.weight <= 0 || value === null) {
      return;
    }
    weightedSum += value * corner.weight;
    weightSum += corner.weight;
  });

  if (weightSum >= 0.45) {
    return weightedSum / weightSum;
  }
  return nearestValue;
}

function interpolateTransportFilledValue(
  field: MapFieldData,
  filledMask: boolean[][] | undefined,
  rowPosition: AxisPosition | null,
  colPosition: AxisPosition | null,
) {
  if (!filledMask || !rowPosition || !colPosition) {
    return null;
  }

  const nearestRow = nearestGridIndex(rowPosition);
  const nearestCol = nearestGridIndex(colPosition);
  if (!maskValue(filledMask, nearestRow, nearestCol)) {
    return null;
  }

  const corners = [
    {
      row: rowPosition.lower,
      col: colPosition.lower,
      weight: (1 - rowPosition.t) * (1 - colPosition.t),
    },
    {
      row: rowPosition.lower,
      col: colPosition.upper,
      weight: (1 - rowPosition.t) * colPosition.t,
    },
    {
      row: rowPosition.upper,
      col: colPosition.lower,
      weight: rowPosition.t * (1 - colPosition.t),
    },
    {
      row: rowPosition.upper,
      col: colPosition.upper,
      weight: rowPosition.t * colPosition.t,
    },
  ];

  let weightedSum = 0;
  let weightSum = 0;
  corners.forEach((corner) => {
    const value = gridValue(field, corner.row, corner.col);
    if (corner.weight <= 0 || value === null || !maskValue(filledMask, corner.row, corner.col)) {
      return;
    }
    weightedSum += value * corner.weight;
    weightSum += corner.weight;
  });

  if (weightSum >= 0.45) {
    return weightedSum / weightSum;
  }
  return gridValue(field, nearestRow, nearestCol);
}

export function quantizedTransportColorPosition(t: number, levels = TRANSPORT_FILLED_LEVELS) {
  const levelCount = Math.max(2, Math.round(levels));
  return Math.round(clamp(t, 0, 1) * (levelCount - 1)) / (levelCount - 1);
}

export function transportInterpolatedFilledValue(field: MapFieldData, rowIndex: number, colIndex: number) {
  return interpolateTransportFilledValue(
    field,
    field.transportRendering?.filledMask,
    indexPosition(rowIndex, field.lat.length),
    indexPosition(colIndex, field.lon.length),
  );
}

function transportFilledRegionKey(region: TransportFilledRegionData) {
  return region.scaleStrategy || region.region || region.id || region.label || "";
}

function transportFilledRegionDefinitions(field: MapFieldData): TransportFilledRegionData[] {
  const explicitRegions = Array.isArray(field.transportRendering?.filledRegions)
    ? field.transportRendering.filledRegions.filter((region) => Array.isArray(region.mask))
    : [];
  if (explicitRegions.length > 0) {
    return explicitRegions;
  }
  const legacyMask = field.transportRendering?.filledMask;
  if (legacyMask) {
    return [{
      id: field.transportRendering?.filledRegion ?? "filled",
      region: field.transportRendering?.filledRegion ?? "china_seas",
      scaleStrategy: field.transportRendering?.filledRegion ?? "china_seas",
      mask: legacyMask,
    }];
  }
  return [];
}

type TransportFilledRegionStyle = {
  region: TransportFilledRegionData;
  min: number;
  max: number;
  colormap: string;
};

function transportFilledRegionStyles(field: MapFieldData, fallbackScale: MapColorScale | null) {
  const allValues = field.values.flat().filter((value) => Number.isFinite(value));
  const fallbackMin = fallbackScale?.min ?? Math.min(...allValues);
  const fallbackMax = fallbackScale?.max ?? Math.max(...allValues);
  return transportFilledRegionDefinitions(field).map((region) => {
    const scale = transportRegionalScale(field, "filled", transportFilledRegionKey(region)) ?? fallbackScale;
    return {
      region,
      min: scale?.min ?? fallbackMin,
      max: scale?.max ?? fallbackMax,
      colormap: scale?.colormap ?? field.transportRendering?.filledColormap ?? DEFAULT_CONTINUOUS_COLORMAP,
    };
  });
}

function sampleTransportFilledRegion(
  field: MapFieldData,
  styles: TransportFilledRegionStyle[],
  rowPosition: AxisPosition | null,
  colPosition: AxisPosition | null,
) {
  for (const style of styles) {
    const value = interpolateTransportFilledValue(field, style.region.mask, rowPosition, colPosition);
    if (Number.isFinite(value)) {
      return { style, value: Number(value) };
    }
  }
  return null;
}

export function transportFilledRegionRenderSample(field: MapFieldData, rowIndex: number, colIndex: number) {
  const fallbackScale = field.colorScale ?? resolveMapColorScale(field);
  const styles = transportFilledRegionStyles(field, fallbackScale);
  const sample = sampleTransportFilledRegion(
    field,
    styles,
    indexPosition(rowIndex, field.lat.length),
    indexPosition(colIndex, field.lon.length),
  );
  if (!sample) {
    return null;
  }
  const range = Math.max(sample.style.max - sample.style.min, 1e-9);
  const rgb = colormapRGB(
    quantizedTransportColorPosition((sample.value - sample.style.min) / range),
    sample.style.colormap,
  );
  return {
    region: sample.style.region.region ?? sample.style.region.id ?? "",
    value: sample.value,
    colormap: sample.style.colormap,
    color: cssRgb(rgb),
  };
}

function readableAngle(angle: number) {
  let normalized = angle;
  if (normalized > Math.PI / 2) {
    normalized -= Math.PI;
  } else if (normalized < -Math.PI / 2) {
    normalized += Math.PI;
  }
  return normalized;
}

function pathAnchorAt(path: ContourPath, fraction: number) {
  const total = pathLength(path);
  if (path.length < 2 || total <= 0) {
    return null;
  }
  const target = clamp(fraction, 0, 1) * total;
  let traveled = 0;
  for (let index = 1; index < path.length; index += 1) {
    const start = path[index - 1];
    const end = path[index];
    const segmentDistance = Math.hypot(end.x - start.x, end.y - start.y);
    if (segmentDistance <= 0) {
      continue;
    }
    if (traveled + segmentDistance >= target) {
      const t = (target - traveled) / segmentDistance;
      return {
        x: start.x + (end.x - start.x) * t,
        y: start.y + (end.y - start.y) * t,
        angle: readableAngle(Math.atan2(end.y - start.y, end.x - start.x)),
      };
    }
    traveled += segmentDistance;
  }
  const start = path[path.length - 2];
  const end = path[path.length - 1];
  return {
    x: end.x,
    y: end.y,
    angle: readableAngle(Math.atan2(end.y - start.y, end.x - start.x)),
  };
}

export function transportContourLabelAnchors(field: MapFieldData, width: number, height: number) {
  const levels = field.transportRendering?.contourLevels?.filter((level) => Number.isFinite(level)) ?? [];
  const contourMask = field.transportRendering?.contourMask;
  if (!contourMask || levels.length === 0 || field.lat.length < 2 || field.lon.length < 2) {
    return [] as ContourLabelAnchor[];
  }
  const bounds = fieldExtent(field);
  const anchors: ContourLabelAnchor[] = [];
  const minimumPathLength = 78;
  const minimumSpacing = 92;
  const maxLabelsPerLevel = width >= 520 ? 5 : 3;
  const margin = 24;

  levels.filter((level) => Number.isInteger(level)).forEach((level) => {
    const color = transportContourColorForLevel(field, level);
    const segments = collectContourSegments(field, width, height, bounds, level, contourMask);
    const paths = stitchContourPaths(segments)
      .map((path) => ({ path, length: pathLength(path) }))
      .filter((item) => item.length >= minimumPathLength)
      .sort((left, right) => right.length - left.length);

    let labelCount = 0;
    for (const { path, length } of paths) {
      const fractions = length > 260 ? [0.32, 0.68] : [0.5];
      for (const fraction of fractions) {
        if (labelCount >= maxLabelsPerLevel) {
          break;
        }
        const anchor = pathAnchorAt(path, fraction);
        if (!anchor) {
          continue;
        }
        if (
          anchor.x < margin
          || anchor.x > width - margin
          || anchor.y < margin
          || anchor.y > height - margin
        ) {
          continue;
        }
        const tooClose = anchors.some((existing) => (
          Math.hypot(existing.x - anchor.x, existing.y - anchor.y) < minimumSpacing
        ));
        if (tooClose) {
          continue;
        }
        anchors.push({
          level,
          label: transportContourLabelText(level),
          x: anchor.x,
          y: anchor.y,
          angle: anchor.angle,
          color,
        });
        labelCount += 1;
      }
      if (labelCount >= maxLabelsPerLevel) {
        break;
      }
    }
  });
  return anchors;
}

function drawInlineContourLabels(ctx: CanvasRenderingContext2D, anchors: ContourLabelAnchor[]) {
  if (anchors.length === 0) {
    return;
  }
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "600 12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  anchors.forEach((anchor) => {
    const labelWidth = ctx.measureText(anchor.label).width;
    const boxWidth = labelWidth + 9;
    const boxHeight = 15;
    ctx.save();
    ctx.translate(anchor.x, anchor.y);
    ctx.rotate(anchor.angle);
    ctx.fillStyle = "rgba(248, 250, 252, 0.88)";
    ctx.strokeStyle = "rgba(15, 23, 42, 0.18)";
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
      ctx.roundRect(-boxWidth / 2, -boxHeight / 2, boxWidth, boxHeight, 4);
    } else {
      ctx.rect(-boxWidth / 2, -boxHeight / 2, boxWidth, boxHeight);
    }
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = anchor.color;
    ctx.fillText(anchor.label, 0, 0.3);
    ctx.restore();
  });
  ctx.restore();
}

function drawTransportContours(
  ctx: CanvasRenderingContext2D,
  field: MapFieldData,
  width: number,
  height: number,
  bounds: ReturnType<typeof fieldExtent>,
) {
  const levels = field.transportRendering?.contourLevels?.filter((level) => Number.isFinite(level)) ?? [];
  const contourMask = field.transportRendering?.contourMask;
  if (!contourMask || levels.length === 0 || field.lat.length < 2 || field.lon.length < 2) {
    return;
  }

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  levels.forEach((level) => {
    const segments = collectContourSegments(field, width, height, bounds, level, contourMask);
    if (segments.length === 0) {
      return;
    }
    ctx.beginPath();
    ctx.strokeStyle = transportContourColorForLevel(field, level);
    ctx.lineWidth = Math.abs(level) < 1e-12 ? 1.7 : 1.05;
    segments.forEach((segment) => {
      ctx.moveTo(segment.start.x, segment.start.y);
      ctx.lineTo(segment.end.x, segment.end.y);
    });
    ctx.stroke();
  });
  drawInlineContourLabels(ctx, transportContourLabelAnchors(field, width, height));
  ctx.restore();
}

function renderTransportStreamfunctionImage(field: MapFieldData, width: number, height: number) {
  const rendering = field.transportRendering;
  const hasExplicitFilledRegions = Array.isArray(rendering?.filledRegions) && rendering.filledRegions.length > 0;
  const fallbackScale = field.colorScale ?? resolveMapColorScale(field);
  const filledStyles = transportFilledRegionStyles(field, fallbackScale);
  if (!rendering || (filledStyles.length === 0 && !rendering.contourMask)) {
    return "";
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return "";
  }

  const imageData = ctx.createImageData(width, height);
  const bounds = fieldExtent(field);
  const northY = latToWebMercatorY(bounds.latMax);
  const southY = latToWebMercatorY(bounds.latMin);
  const lonSpan = Math.max(bounds.lonMax - bounds.lonMin, 1e-9);
  const mercatorSpan = Math.max(northY - southY, 1e-9);
  const rowPositions = Array.from({ length: height }, (_, py) => {
    const mercatorY = northY - ((py + 0.5) / height) * mercatorSpan;
    return axisPosition(field.lat, webMercatorYToLat(mercatorY));
  });
  const colPositions = Array.from({ length: width }, (_, px) => {
    const targetLon = bounds.lonMin + ((px + 0.5) / width) * lonSpan;
    return axisPosition(field.lon, targetLon);
  });

  for (let py = 0; py < height; py += 1) {
    const rowPosition = rowPositions[py];
    for (let px = 0; px < width; px += 1) {
      const colPosition = colPositions[px];
      const sample = sampleTransportFilledRegion(field, filledStyles, rowPosition, colPosition);
      const index = (py * width + px) * 4;
      if (!sample) {
        imageData.data[index] = 255;
        imageData.data[index + 1] = 255;
        imageData.data[index + 2] = 255;
        imageData.data[index + 3] = 0;
        continue;
      }
      const range = Math.max(sample.style.max - sample.style.min, 1e-9);
      const [red, green, blue] = colormapRGB(
        quantizedTransportColorPosition((sample.value - sample.style.min) / range),
        sample.style.colormap,
      );
      imageData.data[index] = red;
      imageData.data[index + 1] = green;
      imageData.data[index + 2] = blue;
      imageData.data[index + 3] = TRANSPORT_FILLED_ALPHA;
    }
  }

  ctx.putImageData(imageData, 0, 0);
  if (!hasExplicitFilledRegions) {
    drawTransportContours(ctx, field, width, height, bounds);
  }
  return canvas.toDataURL();
}

export function useMapFieldImageUrl(field: MapFieldData | null | undefined, width: number, height: number) {
  const [heatmapUrl, setHeatmapUrl] = useState<string>("");

  useEffect(() => {
    if (!field) {
      setHeatmapUrl("");
      return;
    }

    if (field.values.length === 0 || field.lat.length === 0 || field.lon.length === 0) {
      setHeatmapUrl("");
      return;
    }

    const rows = field.values.length;
    const cols = Math.max(...field.values.map((row) => row.length), 0);
    if (rows === 0 || cols === 0) {
      setHeatmapUrl("");
      return;
    }

    const allValues = field.values.flat().filter((value) => Number.isFinite(value));
    if (allValues.length === 0) {
      setHeatmapUrl("");
      return;
    }

    const tileMapKind = mapFieldTileKind(field);
    if (field.subregionGrid && tileMapKind !== "event_hotspot") {
      setHeatmapUrl(renderSubregionGridImage(field, width, height));
      return;
    }
    if (isTransportStreamfunctionRendering(field) && field.contourImage) {
      setHeatmapUrl(`data:image/png;base64,${field.contourImage}`);
      return;
    }
    if (isTransportStreamfunctionRendering(field)) {
      setHeatmapUrl(renderTransportStreamfunctionImage(field, width, height));
      return;
    }

    const discreteLegend = tileMapKind === "event_hotspot"
      ? []
      : Array.isArray(field.discreteLegend)
        ? field.discreteLegend
        : [];
    const discreteColorMap = new Map<number, [number, number, number]>();
    discreteLegend.forEach((item) => {
      const match = /^#?([0-9a-f]{6})$/i.exec(String(item.color ?? "").trim());
      if (!match || !Number.isFinite(item.value)) {
        return;
      }
      const hex = match[1];
      discreteColorMap.set(Number(item.value), [
        Number.parseInt(hex.slice(0, 2), 16),
        Number.parseInt(hex.slice(2, 4), 16),
        Number.parseInt(hex.slice(4, 6), 16),
      ]);
    });

    if (field.contourImage && discreteColorMap.size === 0) {
      setHeatmapUrl(`data:image/png;base64,${field.contourImage}`);
      return;
    }

    const colorScale = resolveMapColorScale(field);
    const valueMin = colorScale?.min ?? Math.min(...allValues);
    const valueMax = colorScale?.max ?? Math.max(...allValues);
    const range = Math.max(valueMax - valueMin, 1e-9);

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setHeatmapUrl("");
      return;
    }

    const imageData = ctx.createImageData(width, height);
    const extent = fieldExtent(field);
    const northY = latToWebMercatorY(extent.latMax);
    const southY = latToWebMercatorY(extent.latMin);
    const lonSpan = Math.max(extent.lonMax - extent.lonMin, 1e-9);
    const mercatorSpan = Math.max(northY - southY, 1e-9);
    const rowPositions = Array.from({ length: height }, (_, py) => {
      const mercatorY = northY - ((py + 0.5) / height) * mercatorSpan;
      return axisPosition(field.lat, webMercatorYToLat(mercatorY));
    });
    const colPositions = Array.from({ length: width }, (_, px) => {
      const targetLon = extent.lonMin + ((px + 0.5) / width) * lonSpan;
      return axisPosition(field.lon, targetLon);
    });
    const shouldInterpolateValues = discreteColorMap.size === 0;

    for (let py = 0; py < height; py += 1) {
      const rowPosition = rowPositions[py];
      const nearestRow = rowPosition ? nearestGridIndex(rowPosition) : 0;
      for (let px = 0; px < width; px += 1) {
        const colPosition = colPositions[px];
        const nearestCol = colPosition ? nearestGridIndex(colPosition) : 0;
        const value = shouldInterpolateValues
          ? interpolateGridValue(field, rowPosition, colPosition)
          : rowPosition && colPosition
            ? field.values[nearestRow]?.[nearestCol]
            : null;
        const index = (py * width + px) * 4;
        if (typeof value !== "number" || !Number.isFinite(value)) {
          imageData.data[index] = 255;
          imageData.data[index + 1] = 255;
          imageData.data[index + 2] = 255;
          imageData.data[index + 3] = 0;
          continue;
        }
        const numericValue = value;
        const discreteRgb =
          discreteColorMap.size > 0
            ? [...discreteColorMap.entries()].sort((a, b) => Math.abs(a[0] - numericValue) - Math.abs(b[0] - numericValue))[0]?.[1]
            : null;
        const [red, green, blue] = discreteRgb ?? colormapRGB(
          clamp((numericValue - valueMin) / range, 0, 1),
          colorScale?.colormap,
        );
        imageData.data[index] = red;
        imageData.data[index + 1] = green;
        imageData.data[index + 2] = blue;
        imageData.data[index + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
    setHeatmapUrl(canvas.toDataURL());
  }, [field, height, width]);

  return heatmapUrl;
}
