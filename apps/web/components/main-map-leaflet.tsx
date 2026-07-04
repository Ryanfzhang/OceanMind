"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  appendGeometryPoint,
  clearGeometry,
  finalizeGeometrySelection,
  geometrySummary,
  hasValidPolygon,
  hasValidTransect,
} from "@/lib/geometry-tools";
import { isTransportStreamfunctionRendering, mapFieldTileKind, useMapFieldImageUrl } from "@/lib/map-field-preview";
import { sampleNearestMapFieldValue, type HoverSample } from "@/lib/map-hover";
import { MapColorbar } from "@/components/map-colorbar";
import {
  LIGHT_BASEMAP,
  OSM_FALLBACK_BASEMAP,
  leafletTileLayerOptions,
  normalizeBasemapConfig,
  type BasemapConfig,
  type MapConfigResponse,
} from "@/lib/basemap";
import { withRegionBounds } from "@/lib/manual-state";
import {
  gridCellDisplayName,
  gridCellStrokeColor,
  sortSubregionCells,
} from "@/lib/subregion-grid";
import {
  eventMarkerRadiusPx,
  groupEventOverlaysForMap,
  hasRenderableRectangleBounds,
  shouldSimplifyEventOverlays,
} from "@/lib/event-display";
import type { EventOverlay, ManualViewState, MapFieldData, ResultCardSummary, WorkspaceData } from "@/lib/types";

type MainMapLeafletProps = {
  activeResult: ResultCardSummary | null;
  activeEventId?: string | null;
  manualState: ManualViewState;
  onManualStateChange: (next: ManualViewState) => void;
  onSelectEvent?: (eventId: string) => void;
  workspaceData: WorkspaceData;
};

type DragState = {
  startLat: number;
  startLon: number;
};

function deriveCenter(bounds: ManualViewState["regionBounds"]): [number, number] {
  return [(bounds.latMin + bounds.latMax) / 2, (bounds.lonMin + bounds.lonMax) / 2];
}

function leafletBoundsFromRegion(bounds: ManualViewState["regionBounds"]): [[number, number], [number, number]] {
  return [
    [bounds.latMin, bounds.lonMin],
    [bounds.latMax, bounds.lonMax],
  ];
}

function formatPointLabel(point: [number, number]) {
  const [lat, lon] = point;
  return `${lon.toFixed(2)}E, ${lat.toFixed(2)}N`;
}

function formatHoverValue(value: number) {
  if (!Number.isFinite(value)) {
    return "NaN";
  }

  const magnitude = Math.abs(value);
  if (magnitude === 0) {
    return "0";
  }
  if (magnitude < 1e-3 || magnitude >= 1e4) {
    return value.toExponential(3);
  }
  if (magnitude < 1) {
    return value.toFixed(6).replace(/\.?0+$/, "");
  }
  return value.toFixed(4).replace(/\.?0+$/, "");
}

function interpolateColor(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) {
    return "rgba(0,0,0,0)";
  }

  const ratio = max === min ? 0.5 : (value - min) / (max - min);
  const clamped = Math.max(0, Math.min(1, ratio));
  const hue = 212 - clamped * 176;
  const saturation = 72;
  const lightness = 74 - clamped * 28;
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

function eventStrokeColor(eventType: string, severity?: string) {
  if (eventType === "hypoxia") {
    return severity === "severe" ? "#b42318" : "#d97706";
  }
  if (eventType === "algal_bloom") {
    return "#15803d";
  }
  if (eventType === "upwelling") {
    return "#0369a1";
  }
  if (eventType === "heatwave") {
    return "#dc2626";
  }
  if (eventType === "eutrophication") {
    return "#7c3aed";
  }
  return "#2563eb";
}

function eventFillColor(eventType: string, severity?: string) {
  if (eventType === "hypoxia") {
    return severity === "severe" ? "rgba(180, 35, 24, 0.16)" : "rgba(217, 119, 6, 0.16)";
  }
  if (eventType === "algal_bloom") {
    return "rgba(21, 128, 61, 0.16)";
  }
  if (eventType === "upwelling") {
    return "rgba(3, 105, 161, 0.16)";
  }
  if (eventType === "heatwave") {
    return "rgba(220, 38, 38, 0.16)";
  }
  if (eventType === "eutrophication") {
    return "rgba(124, 58, 237, 0.16)";
  }
  return "rgba(37, 99, 235, 0.16)";
}

function eventTooltipHtml(event: EventOverlay) {
  const details = event.details.map((line) => `<div>${line}</div>`).join("");
  return `
    <div class="event-tooltip">
      <strong>${event.title}</strong>
      ${details}
    </div>
  `;
}

function subregionTooltipHtml(cell: NonNullable<MapFieldData["subregionGrid"]>["cells"][number]) {
  const dominantScore =
    typeof cell.value === "number" && Number.isFinite(cell.value)
      ? cell.value.toFixed(3).replace(/\.?0+$/, "")
      : typeof cell.dominantScore === "number" && Number.isFinite(cell.dominantScore)
        ? cell.dominantScore.toFixed(3).replace(/\.?0+$/, "")
      : "NA";
  const status = String(cell.status ?? "").trim().toLowerCase();
  const statusLabel =
    status === "skipped_no_valid_ocean"
      ? "No valid ocean cells"
      : status === "skipped_no_valid_samples"
        ? "No valid event/background samples"
      : status === "ok"
          ? "Valid"
          : cell.status;
  const support = status === "ok" ? cell.valueLabel ?? cell.claimStrength ?? "OK" : statusLabel;
  const runnerUp =
    cell.runnerUpMechanism && typeof cell.runnerUpScore === "number"
      ? `<div>Runner-up: ${cell.runnerUpMechanism.replaceAll("_", " ")} (${cell.runnerUpScore
          .toFixed(3)
          .replace(/\.?0+$/, "")})</div>`
      : "";
  return `
    <div class="event-tooltip">
      <strong>${cell.shortLabel ?? cell.label}</strong>
      <div>Dominant: ${gridCellDisplayName(cell)}</div>
      <div>Share / Score: ${dominantScore}</div>
      <div>Support: ${support}</div>
      ${runnerUp}
    </div>
  `;
}

function mapFieldOverlayOpacity(mapField: MapFieldData) {
  const tileMapKind = mapFieldTileKind(mapField);
  if (tileMapKind === "event_hotspot") {
    return 0.78;
  }
  if (mapField.subregionGrid) {
    return 1;
  }
  if (isTransportStreamfunctionRendering(mapField)) {
    return 0.86;
  }
  return 0.75;
}

function eventSymbol(event: EventOverlay) {
  if (event.symbol) {
    return event.symbol;
  }
  if (event.eventType === "jet") {
    return "square";
  }
  if (event.eventType === "front" || event.eventType === "eddy" || event.eventType === "meander") {
    return "diamond";
  }
  return "triangle";
}

function eventOverlayBounds(eventOverlays: EventOverlay[]): [[number, number], [number, number]] | null {
  const lats: number[] = [];
  const lons: number[] = [];

  eventOverlays.forEach((event) => {
    if (event.bounds) {
      lats.push(event.bounds.latMin, event.bounds.latMax);
      lons.push(event.bounds.lonMin, event.bounds.lonMax);
    }
    if (event.path?.length) {
      event.path.forEach((point) => {
        lats.push(point.lat);
        lons.push(point.lon);
      });
    }
    lats.push(event.center.lat);
    lons.push(event.center.lon);
  });

  if (lats.length === 0 || lons.length === 0) {
    return null;
  }

  return [
    [Math.min(...lats), Math.min(...lons)],
    [Math.max(...lats), Math.max(...lons)],
  ];
}

async function loadRuntimeBasemap(): Promise<{ basemap: BasemapConfig; fallback: BasemapConfig }> {
  try {
    const response = await fetch("/api/map-config", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Map config request failed: ${response.status}`);
    }
    const payload = (await response.json()) as Partial<MapConfigResponse>;
    return {
      basemap: normalizeBasemapConfig(payload.basemap, LIGHT_BASEMAP),
      fallback: normalizeBasemapConfig(payload.fallback, OSM_FALLBACK_BASEMAP),
    };
  } catch {
    return {
      basemap: LIGHT_BASEMAP,
      fallback: OSM_FALLBACK_BASEMAP,
    };
  }
}

function createTileLayerStack(L: any, basemap: BasemapConfig) {
  return [basemap, ...(basemap.overlays ?? [])].map((layer) =>
    L.tileLayer(layer.url, leafletTileLayerOptions(layer)),
  );
}

export default function MainMapLeaflet({
  activeResult,
  activeEventId = null,
  manualState,
  onManualStateChange,
  onSelectEvent,
  workspaceData
}: MainMapLeafletProps) {
  const regionBounds = manualState.regionBounds;
  const mapField = workspaceData.mapField;
  const mapFieldImageUrl = useMapFieldImageUrl(mapField, 720, 520);
  const eventOverlays = workspaceData.eventOverlays;
  const mapEventOverlays = useMemo(() => groupEventOverlaysForMap(eventOverlays), [eventOverlays]);
  const simplifyEventOverlays = useMemo(() => shouldSimplifyEventOverlays(eventOverlays), [eventOverlays]);
  const isTransectMode = manualState.selectionMode === "transect";
  const isPolygonMode = manualState.selectionMode === "polygon";
  const isDrawMode = isTransectMode || isPolygonMode;
  const showBoxSelection = manualState.selectionMode === "box";
  const showPointSelection = manualState.selectionMode === "point";
  const showTransectSelection = manualState.transectPoints.length > 0;
  const showPolygonSelection = manualState.polygonPoints.length > 0;
  const showSelectionOverlay = showBoxSelection || showPointSelection || showTransectSelection || showPolygonSelection;
  const activeGeometryPoints = isTransectMode
    ? manualState.transectPoints
    : isPolygonMode
      ? manualState.polygonPoints
      : [];
  const activeGeometrySummary = isTransectMode
    ? geometrySummary(manualState.transectPoints, "transect")
    : isPolygonMode
      ? geometrySummary(manualState.polygonPoints, "polygon")
      : "";
  const canFinishDrawing = isTransectMode
    ? hasValidTransect(manualState.transectPoints)
    : isPolygonMode
      ? hasValidPolygon(manualState.polygonPoints)
      : false;
  const [selectedPoint, setSelectedPoint] = useState<[number, number]>(manualState.selectedPoint);
  const [hoverSample, setHoverSample] = useState<HoverSample | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<any>(null);
  const selectionLayerRef = useRef<any>(null);
  const fieldLayerRef = useRef<any>(null);
  const subregionLayerRef = useRef<any>(null);
  const eventLayerRef = useRef<any>(null);
  const previewRectRef = useRef<any>(null);
  const dragStateRef = useRef<DragState | null>(null);
  const suppressNextClickRef = useRef(false);
  const leafletRef = useRef<any>(null);
  const latestRegionBoundsRef = useRef(regionBounds);
  const latestSelectionModeRef = useRef(manualState.selectionMode);
  latestRegionBoundsRef.current = regionBounds;
  latestSelectionModeRef.current = manualState.selectionMode;

  useEffect(() => {
    setSelectedPoint(manualState.selectedPoint);
  }, [manualState.selectedPoint]);

  const handleUndoDrawing = () => {
    if (isTransectMode) {
      onManualStateChange({
        ...manualState,
        transectPoints: manualState.transectPoints.slice(0, -1),
      });
      return;
    }

    if (isPolygonMode) {
      onManualStateChange({
        ...manualState,
        polygonPoints: manualState.polygonPoints.slice(0, -1),
      });
    }
  };

  const handleClearDrawing = () => {
    if (isTransectMode) {
      onManualStateChange(clearGeometry(manualState, "transect"));
      return;
    }

    if (isPolygonMode) {
      onManualStateChange(clearGeometry(manualState, "polygon"));
    }
  };

  const handleFinishDrawing = () => {
    if (!canFinishDrawing) {
      return;
    }
    onManualStateChange(finalizeGeometrySelection(manualState));
  };

  const selectedEvent = useMemo(
    () =>
      mapEventOverlays.find((event) => event.id === activeEventId) ??
      eventOverlays.find((event) => event.id === activeEventId) ??
      null,
    [activeEventId, eventOverlays, mapEventOverlays]
  );

  useEffect(() => {
    let mounted = true;

    async function initMap() {
      if (!containerRef.current || mapRef.current) {
        return;
      }

      const [L, runtimeBasemap] = await Promise.all([import("leaflet"), loadRuntimeBasemap()]);
      if (!mounted || !containerRef.current || mapRef.current) {
        return;
      }

      leafletRef.current = L;

      const initialBounds = latestRegionBoundsRef.current;
      const map = L.map(containerRef.current, {
        center: deriveCenter(initialBounds),
        zoom: 4,
        zoomControl: true,
        preferCanvas: true
      });

      let fallbackInstalled = false;
      let basemapLayers = createTileLayerStack(L, runtimeBasemap.basemap);
      const installFallbackBasemap = () => {
        if (fallbackInstalled || !mounted) {
          return;
        }
        fallbackInstalled = true;
        basemapLayers.forEach((layer: any) => {
          if (map.hasLayer(layer)) {
            map.removeLayer(layer);
          }
        });
        basemapLayers = createTileLayerStack(L, runtimeBasemap.fallback);
        basemapLayers.forEach((layer: any) => layer.addTo(map));
      };
      basemapLayers.forEach((layer: any) => {
        layer.on("tileerror", installFallbackBasemap);
        layer.addTo(map);
      });

      mapRef.current = map;
      if (latestSelectionModeRef.current === "box") {
        map.dragging?.disable?.();
        map.boxZoom?.disable?.();
      }
      map.fitBounds(leafletBoundsFromRegion(initialBounds) as any, { padding: [28, 28], maxZoom: 7 });

      window.setTimeout(() => {
        if (mounted && mapRef.current === map && containerRef.current?.isConnected) {
          map.invalidateSize();
        }
      }, 60);
    }

    initMap();

    return () => {
      mounted = false;
      if (mapRef.current) {
        const map = mapRef.current;
        map.dragging?.disable?.();
        map.touchZoom?.disable?.();
        map.doubleClickZoom?.disable?.();
        map.scrollWheelZoom?.disable?.();
        map.boxZoom?.disable?.();
        map.keyboard?.disable?.();
        map.off();
        map.remove();
        mapRef.current = null;
      }
      selectionLayerRef.current = null;
      fieldLayerRef.current = null;
      subregionLayerRef.current = null;
      eventLayerRef.current = null;
      previewRectRef.current = null;
      leafletRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }
    window.setTimeout(() => {
      if (mapRef.current === map && containerRef.current?.isConnected) {
        map.invalidateSize();
      }
    }, 0);
  }, [activeResult?.id, mapField, regionBounds]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || activeResult || mapField || eventOverlays.length > 0) {
      return;
    }
    map.fitBounds(leafletBoundsFromRegion(regionBounds) as any, { padding: [28, 28], maxZoom: 7 });
  }, [activeResult, eventOverlays.length, mapField, regionBounds]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !activeResult) {
      return;
    }

    if (mapField?.bounds) {
      map.fitBounds(mapField.bounds as any, { padding: [28, 28] });
      return;
    }

    if (eventOverlays.length > 0) {
      const bounds = eventOverlayBounds(eventOverlays);
      if (bounds) {
        map.fitBounds(bounds as any, { padding: [28, 28], maxZoom: 7 });
      }
    }
  }, [activeResult?.id, eventOverlays, mapField]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.doubleClickZoom) {
      return;
    }

    if (isDrawMode) {
      map.doubleClickZoom.disable();
    } else {
      map.doubleClickZoom.enable();
    }

    return () => {
      if (map.doubleClickZoom && !map.doubleClickZoom.enabled()) {
        map.doubleClickZoom.enable();
      }
    };
  }, [isDrawMode]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }

    if (manualState.selectionMode === "box") {
      map.dragging?.disable?.();
      map.boxZoom?.disable?.();
      return () => {
        map.dragging?.enable?.();
        map.boxZoom?.enable?.();
      };
    }

    map.dragging?.enable?.();
    map.boxZoom?.enable?.();
  }, [manualState.selectionMode]);

  useEffect(() => {
    const map = mapRef.current;
    const L = leafletRef.current;
    if (!map || !L) {
      return;
    }

    const handleClick = (event: any) => {
      if (suppressNextClickRef.current) {
        suppressNextClickRef.current = false;
        return;
      }

      if (isTransectMode) {
        onManualStateChange({
          ...manualState,
          transectPoints: appendGeometryPoint(manualState.transectPoints, {
            lat: event.latlng.lat,
            lon: event.latlng.lng,
          }),
        });
        return;
      }

      if (isPolygonMode) {
        onManualStateChange({
          ...manualState,
          polygonPoints: appendGeometryPoint(manualState.polygonPoints, {
            lat: event.latlng.lat,
            lon: event.latlng.lng,
          }),
        });
        return;
      }

      if (manualState.selectionMode !== "point") {
        return;
      }
      const nextPoint: [number, number] = [event.latlng.lat, event.latlng.lng];
      setSelectedPoint(nextPoint);
      onManualStateChange({
        ...manualState,
        selectedPoint: nextPoint
      });
    };

    const handleMouseMove = (event: any) => {
      setHoverSample(sampleNearestMapFieldValue(mapField, event.latlng.lat, event.latlng.lng, regionBounds));

      if (manualState.selectionMode !== "box" || !dragStateRef.current) {
        return;
      }
      const dragStart = dragStateRef.current;
      const previewBounds = [
        [dragStart.startLat, dragStart.startLon],
        [event.latlng.lat, event.latlng.lng]
      ];
      if (!previewRectRef.current) {
        previewRectRef.current = L.rectangle(previewBounds, {
          color: "#2d9cdb",
          weight: 2,
          dashArray: "4 4",
          fillOpacity: 0.05,
          interactive: false,
        }).addTo(map);
      } else {
        previewRectRef.current.setBounds(previewBounds);
      }
    };

    const handleMouseOut = () => {
      setHoverSample(null);
    };

    const handleDoubleClick = () => {
      if (!isDrawMode || !canFinishDrawing) {
        return;
      }
      onManualStateChange(finalizeGeometrySelection(manualState));
    };

    const handleMouseDown = (event: any) => {
      if (manualState.selectionMode !== "box") {
        return;
      }
      dragStateRef.current = {
        startLat: event.latlng.lat,
        startLon: event.latlng.lng
      };
    };

    const handleMouseUp = (event: any) => {
      if (manualState.selectionMode !== "box" || !dragStateRef.current) {
        return;
      }

      const dragStart = dragStateRef.current;
      const nextBounds = {
        lonMin: Math.min(dragStart.startLon, event.latlng.lng),
        lonMax: Math.max(dragStart.startLon, event.latlng.lng),
        latMin: Math.min(dragStart.startLat, event.latlng.lat),
        latMax: Math.max(dragStart.startLat, event.latlng.lat)
      };
      const centerPoint: [number, number] = [
        (nextBounds.latMin + nextBounds.latMax) / 2,
        (nextBounds.lonMin + nextBounds.lonMax) / 2
      ];

      dragStateRef.current = null;
      suppressNextClickRef.current = true;

      if (previewRectRef.current) {
        previewRectRef.current.remove();
        previewRectRef.current = null;
      }

      if (
        Math.abs(nextBounds.lonMax - nextBounds.lonMin) < 0.05 ||
        Math.abs(nextBounds.latMax - nextBounds.latMin) < 0.05
      ) {
        return;
      }

      setSelectedPoint(centerPoint);
      onManualStateChange(
        withRegionBounds(
          {
            ...manualState,
            selectedPoint: centerPoint
          },
          nextBounds
        )
      );
    };

    map.on("click", handleClick);
    map.on("mousemove", handleMouseMove);
    map.on("mouseout", handleMouseOut);
    map.on("dblclick", handleDoubleClick);
    map.on("mousedown", handleMouseDown);
    map.on("mouseup", handleMouseUp);

    return () => {
      map.off("click", handleClick);
      map.off("mousemove", handleMouseMove);
      map.off("mouseout", handleMouseOut);
      map.off("dblclick", handleDoubleClick);
      map.off("mousedown", handleMouseDown);
      map.off("mouseup", handleMouseUp);
    };
  }, [
    canFinishDrawing,
    isDrawMode,
    isPolygonMode,
    isTransectMode,
    manualState,
    mapField,
    onManualStateChange,
    regionBounds,
  ]);

  useEffect(() => {
    const map = mapRef.current;
    const L = leafletRef.current;
    if (!map || !L) {
      return;
    }

    if (selectionLayerRef.current) {
      selectionLayerRef.current.remove();
      selectionLayerRef.current = null;
    }

    if (!showSelectionOverlay) {
      return;
    }

    const layerGroup = L.layerGroup().addTo(map);
    selectionLayerRef.current = layerGroup;

    const bounds = leafletBoundsFromRegion(regionBounds);

    if (showBoxSelection) {
      const rectangle = L.rectangle(bounds, {
        color: "#ffffff",
        weight: 2,
        fillColor: "#8ad8ff",
        fillOpacity: 0.04,
        interactive: false,
      });
      rectangle.bindTooltip(manualState.regionLabel, {
        permanent: true,
        direction: "top",
        interactive: false,
      });
      rectangle.addTo(layerGroup);
    }

    if (showPointSelection) {
      const selectedMarker = L.circleMarker(selectedPoint, {
        radius: 7,
        color: "#ff715b",
        fillColor: "#ff715b",
        fillOpacity: 0.9,
        weight: 2
      });
      selectedMarker.bindPopup(`<strong>Selected point</strong><br />${formatPointLabel(selectedPoint)}`);
      selectedMarker.addTo(layerGroup);
    }

    if (manualState.transectPoints.length > 0) {
      const transectLatLngs = manualState.transectPoints.map((point) => [point.lat, point.lon] as [number, number]);
      L.polyline(transectLatLngs, {
        color: "#0f766e",
        weight: 3,
        opacity: 0.95,
        interactive: false,
      }).addTo(layerGroup);
      manualState.transectPoints.forEach((point, index) => {
        const isStart = index === 0;
        const isEnd = index === manualState.transectPoints.length - 1;
        const vertex = L.circleMarker([point.lat, point.lon], {
          radius: isStart || isEnd ? 6 : 4,
          color: "#115e59",
          fillColor: isStart ? "#14b8a6" : isEnd ? "#0ea5e9" : "#99f6e4",
          fillOpacity: 0.95,
          weight: 2,
          interactive: false,
        });
        vertex.bindTooltip(isStart ? "Start" : isEnd ? "End" : `P${index + 1}`, {
          direction: "top",
        });
        vertex.addTo(layerGroup);
      });
    }

    if (manualState.polygonPoints.length > 0) {
      const polygonLatLngs = manualState.polygonPoints.map((point) => [point.lat, point.lon] as [number, number]);
      if (manualState.polygonPoints.length >= 3) {
        L.polygon(polygonLatLngs, {
          color: "#7c3aed",
          fillColor: "#8b5cf6",
          fillOpacity: 0.12,
          weight: 2,
          interactive: false,
        }).addTo(layerGroup);
      } else {
        L.polyline(polygonLatLngs, {
          color: "#7c3aed",
          weight: 2,
          dashArray: "6 4",
          interactive: false,
        }).addTo(layerGroup);
      }
      manualState.polygonPoints.forEach((point, index) => {
        const vertex = L.circleMarker([point.lat, point.lon], {
          radius: 4,
          color: "#6d28d9",
          fillColor: "#c4b5fd",
          fillOpacity: 0.95,
          weight: 2,
          interactive: false,
        });
        vertex.bindTooltip(`V${index + 1}`, {
          direction: "top",
        });
        vertex.addTo(layerGroup);
      });
    }

    if (showBoxSelection) {
      map.fitBounds(bounds as any, { padding: [28, 28] });
    }
  }, [
    manualState.polygonPoints,
    manualState.regionLabel,
    manualState.transectPoints,
    regionBounds,
    selectedPoint,
    showBoxSelection,
    showPointSelection,
    showPolygonSelection,
    showSelectionOverlay,
    showTransectSelection,
  ]);

  useEffect(() => {
    const map = mapRef.current;
    const L = leafletRef.current;
    if (!map || !L) {
      return;
    }

    if (fieldLayerRef.current) {
      fieldLayerRef.current.remove();
    }

    if (!mapField || !mapFieldImageUrl || !mapField.bounds) {
      fieldLayerRef.current = null;
      return;
    }

    const overlay = L.imageOverlay(mapFieldImageUrl, mapField.bounds, {
      opacity: mapFieldOverlayOpacity(mapField),
    });
    overlay.addTo(map);
    fieldLayerRef.current = overlay;
  }, [mapField, mapFieldImageUrl]);

  useEffect(() => {
    const map = mapRef.current;
    const L = leafletRef.current;
    if (!map || !L) {
      return;
    }

    if (subregionLayerRef.current) {
      subregionLayerRef.current.remove();
    }

    if (!mapField?.subregionGrid || mapField.subregionGrid.cells.length === 0) {
      subregionLayerRef.current = null;
      return;
    }

    const layerGroup = L.layerGroup().addTo(map);
    subregionLayerRef.current = layerGroup;
    const showLabels = mapField.subregionGrid.cells.length <= 25;
    const baseWeight = showLabels ? 2.5 : 1.15;
    const mutedWeight = showLabels ? 1.75 : 0.9;
    const hoverWeight = showLabels ? 3.5 : 2;
    const hoverMutedWeight = showLabels ? 2.5 : 1.5;

    sortSubregionCells(mapField.subregionGrid.cells).forEach((cell) => {
      if (!cell.bounds) {
        return;
      }
      const strokeColor = gridCellStrokeColor(cell);
      const dashed = cell.status !== "ok";
      const rectangle = L.rectangle(
        [
          [cell.bounds.latMin, cell.bounds.lonMin],
          [cell.bounds.latMax, cell.bounds.lonMax],
        ],
        {
          color: strokeColor,
          weight: cell.status === "ok" ? baseWeight : mutedWeight,
          dashArray: dashed ? "6 4" : undefined,
          fillColor: strokeColor,
          fillOpacity: cell.status === "ok" ? 0 : 0.02,
        },
      );
      rectangle.bindTooltip(subregionTooltipHtml(cell), {
        direction: "top",
        sticky: true,
        opacity: 0.96,
      });
      rectangle.on("mouseover", () =>
        rectangle.setStyle({
          weight: cell.status === "ok" ? hoverWeight : hoverMutedWeight,
          fillOpacity: cell.status === "ok" ? 0.08 : 0.05,
        }),
      );
      rectangle.on("mouseout", () =>
        rectangle.setStyle({
          weight: cell.status === "ok" ? baseWeight : mutedWeight,
          fillOpacity: cell.status === "ok" ? 0 : 0.02,
        }),
      );
      rectangle.addTo(layerGroup);

      if (!showLabels) {
        return;
      }
      const centerLat = (cell.bounds.latMin + cell.bounds.latMax) / 2;
      const centerLon = (cell.bounds.lonMin + cell.bounds.lonMax) / 2;
      const labelIcon = L.divIcon({
        className: "subregion-grid-label",
        html: `<span style="
          display:inline-flex;
          align-items:center;
          justify-content:center;
          min-width:32px;
          padding:3px 8px;
          border-radius:999px;
          border:1px solid ${strokeColor};
          background:rgba(255,255,255,0.88);
          color:${strokeColor};
          font-size:11px;
          font-weight:700;
          letter-spacing:0.04em;
        ">${cell.shortLabel ?? cell.label}</span>`,
        iconSize: [44, 24],
        iconAnchor: [22, 12],
      });
      L.marker([centerLat, centerLon], { icon: labelIcon, interactive: false }).addTo(layerGroup);
    });
  }, [mapField]);

  useEffect(() => {
    const map = mapRef.current;
    const L = leafletRef.current;
    if (!map || !L) {
      return;
    }

    if (eventLayerRef.current) {
      eventLayerRef.current.remove();
    }

    const eventGroup = L.layerGroup().addTo(map);
    eventLayerRef.current = eventGroup;

    if (!mapEventOverlays || mapEventOverlays.length === 0) {
      return;
    }

    mapEventOverlays.forEach((event) => {
      const strokeColor = eventStrokeColor(event.eventType, event.severity);
      const fillColor = eventFillColor(event.eventType, event.severity);
      const tooltip = eventTooltipHtml(event);
      const isSelected = event.id === activeEventId;

      if (simplifyEventOverlays) {
        const marker = L.circleMarker([event.center.lat, event.center.lon], {
          radius: eventMarkerRadiusPx(event),
          color: strokeColor,
          fillColor: strokeColor,
          fillOpacity: isSelected ? 0.42 : 0.22,
          opacity: isSelected ? 0.95 : 0.72,
          weight: isSelected ? 2.5 : 1,
        });
        marker.bindTooltip(tooltip, {
          direction: "top",
          sticky: true,
          opacity: 0.96,
        });
        marker.on("mouseover", () =>
          marker.setStyle({ weight: isSelected ? 3 : 1.5, fillOpacity: isSelected ? 0.5 : 0.3 }),
        );
        marker.on("mouseout", () =>
          marker.setStyle({ weight: isSelected ? 2.5 : 1, fillOpacity: isSelected ? 0.42 : 0.22 }),
        );
        marker.on("click", () => onSelectEvent?.(event.id));
        marker.addTo(eventGroup);
        if (typeof event.occurrenceCount === "number" && event.occurrenceCount > 1) {
          const labelIcon = L.divIcon({
            className: "event-count-label",
            html: `<span style="
              display:inline-flex;
              align-items:center;
              justify-content:center;
              min-width:24px;
              height:18px;
              padding:0 6px;
              border-radius:999px;
              background:rgba(255,255,255,0.92);
              border:1px solid ${strokeColor};
              color:${strokeColor};
              font-size:11px;
              font-weight:800;
              line-height:18px;
              box-shadow:0 1px 4px rgba(15,23,42,0.2);
            ">${event.occurrenceCount}</span>`,
            iconSize: [36, 18],
            iconAnchor: [18, 9],
          });
          L.marker([event.center.lat, event.center.lon], { icon: labelIcon, interactive: false }).addTo(eventGroup);
        }
        return;
      }

      if (hasRenderableRectangleBounds(event) && event.bounds) {
        const rectangle = L.rectangle(
          [
            [event.bounds.latMin, event.bounds.lonMin],
            [event.bounds.latMax, event.bounds.lonMax]
          ],
          {
            color: strokeColor,
            weight: isSelected ? 4 : 2,
            fillColor: strokeColor,
            fillOpacity: isSelected ? 0.24 : 0.16
          }
        );
        rectangle.bindTooltip(tooltip, {
          direction: "top",
          sticky: true,
          opacity: 0.96
        });
        rectangle.on("mouseover", () => rectangle.setStyle({ weight: isSelected ? 5 : 3, fillOpacity: isSelected ? 0.28 : 0.22 }));
        rectangle.on("mouseout", () => rectangle.setStyle({ weight: isSelected ? 4 : 2, fillOpacity: isSelected ? 0.24 : 0.16 }));
        rectangle.on("click", () => onSelectEvent?.(event.id));
        rectangle.addTo(eventGroup);
        return;
      }

      if (event.shape === "circle" && typeof event.radiusKm === "number") {
        const circle = L.circle([event.center.lat, event.center.lon], {
          radius: event.radiusKm * 1000,
          color: strokeColor,
          fillColor: strokeColor,
          fillOpacity: isSelected ? 0.18 : 0.12,
          weight: isSelected ? 4 : 2
        });
        circle.bindTooltip(tooltip, {
          direction: "top",
          sticky: true,
          opacity: 0.96
        });
        circle.on("mouseover", () => circle.setStyle({ weight: isSelected ? 5 : 3, fillOpacity: isSelected ? 0.22 : 0.18 }));
        circle.on("mouseout", () => circle.setStyle({ weight: isSelected ? 4 : 2, fillOpacity: isSelected ? 0.18 : 0.12 }));
        circle.on("click", () => onSelectEvent?.(event.id));
        circle.addTo(eventGroup);
        return;
      }

      if (event.shape === "polyline" && event.path && event.path.length >= 2) {
        const polyline = L.polyline(
          event.path.map((point) => [point.lat, point.lon] as [number, number]),
          {
            color: strokeColor,
            weight: isSelected ? 5 : 3,
            opacity: 0.92,
            lineCap: "round",
            lineJoin: "round"
          }
        );
        polyline.bindTooltip(tooltip, {
          direction: "top",
          sticky: true,
          opacity: 0.96
        });
        polyline.on("mouseover", () => polyline.setStyle({ weight: isSelected ? 6 : 4 }));
        polyline.on("mouseout", () => polyline.setStyle({ weight: isSelected ? 5 : 3 }));
        polyline.on("click", () => onSelectEvent?.(event.id));
        polyline.addTo(eventGroup);
        return;
      }

      const icon = L.divIcon({
        className: `event-symbol-marker event-symbol-${eventSymbol(event)}`,
        html: `<span style="--event-color:${strokeColor}; --event-fill:${fillColor};"></span>`,
        iconSize: [18, 18],
        iconAnchor: [9, 9]
      });
      const marker = L.marker([event.center.lat, event.center.lon], { icon });
      marker.bindTooltip(tooltip, {
        direction: "top",
        sticky: true,
        opacity: 0.96
      });
      marker.on("click", () => onSelectEvent?.(event.id));
      marker.addTo(eventGroup);
    });
  }, [activeEventId, mapEventOverlays, onSelectEvent, simplifyEventOverlays]);

  return (
    <div className="map-stage">
      <div className="map-canvas live-map-shell">
        <div
          ref={containerRef}
          className={`leaflet-map ${manualState.selectionMode === "box" ? "is-box-mode" : ""} ${isDrawMode ? "is-draw-mode" : ""}`}
        />
        {isDrawMode ? (
          <div className="map-floating-card top-left-card map-draw-toolbar">
            <span className="mini-label">{isTransectMode ? "Transect Tool" : "Polygon Tool"}</span>
            <strong>{activeGeometrySummary}</strong>
            <span>Single click adds vertices. Double click or Finish completes the shape.</span>
            <div className="step-card-action-row">
              <button className="result-expand-btn" onClick={handleUndoDrawing} type="button">
                Undo
              </button>
              <button className="result-expand-btn" onClick={handleFinishDrawing} type="button">
                Finish
              </button>
              <button className="result-expand-btn" onClick={handleClearDrawing} type="button">
                Clear
              </button>
            </div>
          </div>
        ) : null}
        {mapField ? (
          <div className="map-hover-panel">
            <span className="mini-label">Hover Value</span>
            {hoverSample ? (
              <>
                <strong>{formatHoverValue(hoverSample.value)}</strong>
                <span>
                  {hoverSample.lon.toFixed(2)}E, {hoverSample.lat.toFixed(2)}N
                </span>
                {mapField.units ? <span>Units: {mapField.units}</span> : null}
              </>
            ) : (
              <>
                <strong>Move over the field</strong>
                {mapField.units ? <span>Units: {mapField.units}</span> : null}
              </>
            )}
          </div>
        ) : null}
        <MapColorbar field={mapField} compact floating />
        {selectedEvent ? (
          <div className="map-floating-card bottom-right-card">
            <span className="mini-label">Selected Event</span>
            <strong>{selectedEvent.title}</strong>
            <span>
              {selectedEvent.center.lon.toFixed(2)}E, {selectedEvent.center.lat.toFixed(2)}N
            </span>
            {selectedEvent.details.slice(0, 4).map((detail) => (
              <span key={detail}>{detail}</span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
