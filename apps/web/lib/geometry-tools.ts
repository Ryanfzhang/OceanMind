import type { GeoPoint, ManualViewState, RegionBounds, SelectionMode } from "./types";

const TRANSECT_REGION_PADDING_DEGREES = 1;

type QueryRegion = {
  lon_range: [number, number];
  lat_range: [number, number];
};

export type WorkspaceSelectionContext = {
  selection_mode: SelectionMode;
  active_selection: SelectionMode;
  selected_region: (
    | {
        type: "box";
        region_bounds: RegionBounds;
        region: QueryRegion;
        label: string;
      }
    | {
        type: "polygon";
        points: [number, number][];
        region_bounds: RegionBounds;
        region: QueryRegion;
      }
  ) | null;
  selected_point: {
    type: "point";
    lat: number;
    lon: number;
  } | null;
  selected_transect: {
    type: "transect";
    points: [number, number][];
    region_bounds: RegionBounds;
    region: QueryRegion;
    padded_region: QueryRegion;
  } | null;
};

function formatBound(value: number) {
  const rounded = Number.parseFloat(value.toFixed(2));
  return Number.isInteger(rounded) ? `${rounded}` : `${rounded}`;
}

function formatRegionLabel(bounds: RegionBounds) {
  return `${formatBound(bounds.lonMin)}-${formatBound(bounds.lonMax)}E · ${formatBound(bounds.latMin)}-${formatBound(bounds.latMax)}N`;
}

function normalizeBounds(bounds: RegionBounds): RegionBounds {
  return {
    lonMin: Math.min(bounds.lonMin, bounds.lonMax),
    lonMax: Math.max(bounds.lonMin, bounds.lonMax),
    latMin: Math.min(bounds.latMin, bounds.latMax),
    latMax: Math.max(bounds.latMin, bounds.latMax),
  };
}

function boundsToQueryRegion(bounds: RegionBounds): QueryRegion {
  const normalized = normalizeBounds(bounds);
  return {
    lon_range: [normalized.lonMin, normalized.lonMax],
    lat_range: [normalized.latMin, normalized.latMax],
  };
}

function padBounds(bounds: RegionBounds, paddingDegrees: number): RegionBounds {
  return normalizeBounds({
    lonMin: bounds.lonMin - paddingDegrees,
    lonMax: bounds.lonMax + paddingDegrees,
    latMin: bounds.latMin - paddingDegrees,
    latMax: bounds.latMax + paddingDegrees,
  });
}

function samePoint(a: GeoPoint | undefined, b: GeoPoint | undefined) {
  if (!a || !b) {
    return false;
  }
  return Math.abs(a.lat - b.lat) < 1e-9 && Math.abs(a.lon - b.lon) < 1e-9;
}

export function appendGeometryPoint(points: GeoPoint[], point: GeoPoint): GeoPoint[] {
  const lastPoint = points[points.length - 1];
  if (samePoint(lastPoint, point)) {
    return points;
  }
  return [...points, point];
}

export function geometryPointsToLonLat(points: GeoPoint[]): [number, number][] {
  return points.map((point) => [point.lon, point.lat]);
}

export function geometryBounds(points: GeoPoint[]): RegionBounds | null {
  if (points.length === 0) {
    return null;
  }
  const lons = points.map((point) => point.lon);
  const lats = points.map((point) => point.lat);
  return normalizeBounds({
    lonMin: Math.min(...lons),
    lonMax: Math.max(...lons),
    latMin: Math.min(...lats),
    latMax: Math.max(...lats),
  });
}

export function geometryCenter(points: GeoPoint[]): [number, number] | null {
  const bounds = geometryBounds(points);
  if (!bounds) {
    return null;
  }
  return [
    (bounds.latMin + bounds.latMax) / 2,
    (bounds.lonMin + bounds.lonMax) / 2,
  ];
}

export function hasValidTransect(points: GeoPoint[]): boolean {
  return points.length >= 2;
}

export function hasValidPolygon(points: GeoPoint[]): boolean {
  return points.length >= 3;
}

export function clearGeometry(state: ManualViewState, mode: "transect" | "polygon"): ManualViewState {
  if (mode === "transect") {
    return {
      ...state,
      transectPoints: [],
      selectionMode: state.selectionMode === "transect" ? "none" : state.selectionMode,
    };
  }
  return {
    ...state,
    polygonPoints: [],
    selectionMode: state.selectionMode === "polygon" ? "none" : state.selectionMode,
  };
}

export function reverseTransect(state: ManualViewState): ManualViewState {
  return {
    ...state,
    transectPoints: [...state.transectPoints].reverse(),
  };
}

export function finalizeGeometrySelection(state: ManualViewState): ManualViewState {
  const points =
    state.selectionMode === "transect"
      ? state.transectPoints
      : state.selectionMode === "polygon"
        ? state.polygonPoints
        : [];

  const isValid =
    state.selectionMode === "transect"
      ? hasValidTransect(points)
      : state.selectionMode === "polygon"
        ? hasValidPolygon(points)
        : false;

  if (!isValid) {
    return state;
  }

  const bounds = geometryBounds(points);
  const center = geometryCenter(points);
  if (!bounds || !center) {
    return state;
  }

  const normalizedBounds = normalizeBounds(bounds);
  return {
    ...state,
    regionBounds: normalizedBounds,
    regionLabel: formatRegionLabel(normalizedBounds),
    selectedPoint: center,
    selectionMode: "none",
  };
}

export function geometrySummary(points: GeoPoint[], mode: "transect" | "polygon"): string {
  if (mode === "transect") {
    if (!hasValidTransect(points)) {
      return `Transect draft · ${points.length} point${points.length === 1 ? "" : "s"}`;
    }
    const start = points[0];
    const end = points[points.length - 1];
    return `${points.length} pts · ${start.lon.toFixed(2)}E/${start.lat.toFixed(2)}N → ${end.lon.toFixed(2)}E/${end.lat.toFixed(2)}N`;
  }

  if (!hasValidPolygon(points)) {
    return `Polygon draft · ${points.length} point${points.length === 1 ? "" : "s"}`;
  }
  return `${points.length} vertices`;
}

function inferActiveSelection(state: ManualViewState, hasTransect: boolean, hasPolygon: boolean): SelectionMode {
  if (state.selectionMode === "point") {
    return "point";
  }
  if (state.selectionMode === "transect") {
    return hasTransect ? "transect" : "none";
  }
  if (hasPolygon) {
    return "polygon";
  }
  if (state.selectionMode === "box") {
    return "box";
  }
  if (hasTransect) {
    return "transect";
  }
  return "box";
}

export function buildWorkspaceSelectionContext(state: ManualViewState): WorkspaceSelectionContext {
  const normalizedBoxBounds = normalizeBounds(state.regionBounds);
  const transectBounds = geometryBounds(state.transectPoints);
  const polygonBounds = geometryBounds(state.polygonPoints);
  const hasTransect = hasValidTransect(state.transectPoints);
  const hasPolygon = hasValidPolygon(state.polygonPoints);
  const activeSelection = inferActiveSelection(state, hasTransect, hasPolygon);
  const transectPaddedBounds = transectBounds ? padBounds(transectBounds, TRANSECT_REGION_PADDING_DEGREES) : null;
  const selectedRegion =
    activeSelection === "polygon" && hasPolygon && polygonBounds
      ? {
          type: "polygon" as const,
          points: geometryPointsToLonLat(state.polygonPoints),
          region_bounds: polygonBounds,
          region: boundsToQueryRegion(polygonBounds),
        }
      : activeSelection === "box"
        ? {
            type: "box" as const,
            region_bounds: normalizedBoxBounds,
            region: boundsToQueryRegion(normalizedBoxBounds),
            label: state.regionLabel,
          }
        : null;

  return {
    selection_mode: state.selectionMode,
    active_selection: activeSelection,
    selected_region: selectedRegion,
    selected_point:
      activeSelection === "point"
        ? {
            type: "point",
            lat: state.selectedPoint[0],
            lon: state.selectedPoint[1],
          }
        : null,
    selected_transect:
      activeSelection === "transect" && hasTransect && transectBounds && transectPaddedBounds
        ? {
            type: "transect",
            points: geometryPointsToLonLat(state.transectPoints),
            region_bounds: transectBounds,
            region: boundsToQueryRegion(transectBounds),
            padded_region: boundsToQueryRegion(transectPaddedBounds),
          }
        : null,
  };
}

function inferVariableFromQuery(queryText: string, fallbackVariable: string): string {
  const normalized = queryText.trim().toLowerCase();
  const mapping: Array<{ terms: string[]; value: string }> = [
    { terms: ["sst", "sea surface temperature", "temperature", "temp", "温度", "海温"], value: "temp" },
    { terms: ["salinity", "salt", "盐度"], value: "salt" },
    { terms: ["chlorophyll", "chl", "chla", "叶绿素"], value: "chlorophyll" },
    { terms: ["oxygen", "dissolved oxygen", "o2", "溶解氧", "氧气"], value: "oxygen" },
    { terms: ["zonal velocity", "u velocity", "u current", "东向流速", "纬向流速"], value: "u" },
    { terms: ["meridional velocity", "v velocity", "v current", "北向流速", "经向流速"], value: "v" },
  ];

  for (const item of mapping) {
    if (item.terms.some((term) => normalized.includes(term))) {
      return item.value;
    }
  }

  return fallbackVariable;
}

function queryWantsSection(queryText: string): boolean {
  const normalized = queryText.trim().toLowerCase();
  return ["section", "transect", "剖面", "断面"].some((term) => normalized.includes(term));
}

function queryWantsSectionEvolution(queryText: string): boolean {
  const normalized = queryText.trim().toLowerCase();
  return [
    "变化",
    "演变",
    "随时间",
    "这段时间",
    "这期间",
    "time-distance",
    "time distance",
    "hovmoller",
    "hovmöller",
  ].some((term) => normalized.includes(term));
}

function queryUsesCurrentTimeWindow(queryText: string): boolean {
  const normalized = queryText.trim().toLowerCase();
  return [
    "这段时间",
    "这期间",
    "当前时间",
    "当前时段",
    "最近这段时间",
    "during this period",
    "over this period",
    "current time range",
  ].some((term) => normalized.includes(term));
}

export function buildQueryExtractedParams(state: ManualViewState, queryText = ""): Record<string, unknown> {
  const workspaceSelection = buildWorkspaceSelectionContext(state);
  const extractedParams: Record<string, unknown> = {
    selection_mode: state.selectionMode,
    workspace_selection: workspaceSelection,
  };
  const activeGeometryPoints: GeoPoint[] = [];
  const hasTransect = hasValidTransect(state.transectPoints);
  const hasPolygon = hasValidPolygon(state.polygonPoints);
  const wantsSection = queryWantsSection(queryText);

  if (workspaceSelection.selected_region?.type === "box") {
    extractedParams.region = workspaceSelection.selected_region.region;
    extractedParams.region_selection_type = "box";
  }

  if (workspaceSelection.selected_region?.type === "polygon") {
    extractedParams.mask_polygon = workspaceSelection.selected_region.points;
    extractedParams.drawn_polygon_points = workspaceSelection.selected_region.points;
    extractedParams.region = workspaceSelection.selected_region.region;
    extractedParams.region_selection_type = "polygon";
    activeGeometryPoints.push(...state.polygonPoints);
  }

  if (workspaceSelection.selected_point) {
    extractedParams.selected_point = {
      lat: workspaceSelection.selected_point.lat,
      lon: workspaceSelection.selected_point.lon,
    };
  }

  if (workspaceSelection.selected_transect || (wantsSection && hasTransect)) {
    extractedParams.transect_points = geometryPointsToLonLat(state.transectPoints);
    extractedParams.drawn_transect_points = extractedParams.transect_points;
    extractedParams.transect_selection_type = "transect";
    activeGeometryPoints.push(...state.transectPoints);
  }

  const bounds = geometryBounds(activeGeometryPoints);
  if (bounds && !extractedParams.region) {
    const queryBounds = hasTransect && !hasPolygon
      ? padBounds(bounds, TRANSECT_REGION_PADDING_DEGREES)
      : bounds;
    extractedParams.region = {
      lon_range: [queryBounds.lonMin, queryBounds.lonMax],
      lat_range: [queryBounds.latMin, queryBounds.latMax],
    };
  }

  if (queryText.trim()) {
    extractedParams.variable = inferVariableFromQuery(queryText, state.variable);

    if (hasValidTransect(state.transectPoints) && wantsSection) {
      extractedParams.analysis_type = queryWantsSectionEvolution(queryText) ? "section_hovmoller" : "section";
      if (extractedParams.analysis_type === "section_hovmoller") {
        extractedParams.diagram_type = "time_distance";
      }
      extractedParams.time_range = state.timeRange;
    }

    if (queryUsesCurrentTimeWindow(queryText) || queryWantsSectionEvolution(queryText)) {
      extractedParams.time_range = state.timeRange;
    }
  }

  return extractedParams;
}

export function nextSelectionMode(current: SelectionMode, target: SelectionMode): SelectionMode {
  return current === target ? "none" : target;
}
