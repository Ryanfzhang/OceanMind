export type BasemapRegion = "global" | "cn";

export type BasemapLayerConfig = {
  id: string;
  label: string;
  region: BasemapRegion;
  url: string;
  attribution: string;
  maxZoom: number;
  subdomains?: string | string[];
};

export type BasemapConfig = BasemapLayerConfig & {
  overlays?: BasemapLayerConfig[];
};

export type MapConfigResponse = {
  region: BasemapRegion;
  detectedCountry: string | null;
  detectionSource: string | null;
  basemap: BasemapConfig;
  fallback: BasemapConfig;
  reason?: string;
};

export const LIGHT_BASEMAP: BasemapConfig = {
  id: "carto-light",
  label: "CARTO Light",
  region: "global",
  url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
  attribution:
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
  maxZoom: 20,
};

export const OSM_FALLBACK_BASEMAP: BasemapConfig = {
  id: "osm-standard",
  label: "OpenStreetMap",
  region: "global",
  url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  maxZoom: 19,
};

type TileLayerOptions = {
  attribution: string;
  maxZoom: number;
  subdomains?: string | string[];
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizeRegion(value: unknown, fallback: BasemapRegion): BasemapRegion {
  return value === "cn" || value === "global" ? value : fallback;
}

function normalizeSubdomains(value: unknown): string | string[] | undefined {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  if (Array.isArray(value) && value.every((entry) => typeof entry === "string" && entry.trim())) {
    return value.map((entry) => entry.trim());
  }
  return undefined;
}

function normalizeBasemapLayerConfig(
  value: unknown,
  fallback: BasemapLayerConfig,
): BasemapLayerConfig {
  if (!isRecord(value)) {
    return fallback;
  }

  const maxZoom =
    typeof value.maxZoom === "number" && Number.isFinite(value.maxZoom) && value.maxZoom > 0
      ? value.maxZoom
      : fallback.maxZoom;

  return {
    id: typeof value.id === "string" && value.id.trim() ? value.id.trim() : fallback.id,
    label: typeof value.label === "string" && value.label.trim() ? value.label.trim() : fallback.label,
    region: normalizeRegion(value.region, fallback.region),
    url: typeof value.url === "string" && value.url.trim() ? value.url.trim() : fallback.url,
    attribution:
      typeof value.attribution === "string" ? value.attribution : fallback.attribution,
    maxZoom,
    subdomains: normalizeSubdomains(value.subdomains) ?? fallback.subdomains,
  };
}

export function normalizeBasemapConfig(value: unknown, fallback: BasemapConfig): BasemapConfig {
  const base = normalizeBasemapLayerConfig(value, fallback);
  const overlays =
    isRecord(value) && Array.isArray(value.overlays)
      ? value.overlays
          .filter(isRecord)
          .map((overlay, index) =>
            normalizeBasemapLayerConfig(overlay, {
              id: `${base.id}-overlay-${index}`,
              label: `${base.label} overlay ${index + 1}`,
              region: base.region,
              url: "",
              attribution: base.attribution,
              maxZoom: base.maxZoom,
              subdomains: base.subdomains,
            }),
          )
          .filter((overlay) => overlay.url)
      : fallback.overlays;

  return overlays?.length ? { ...base, overlays } : base;
}

export function leafletTileLayerOptions(layer: BasemapLayerConfig): TileLayerOptions {
  const options: TileLayerOptions = {
    attribution: layer.attribution,
    maxZoom: layer.maxZoom,
  };
  if (layer.subdomains) {
    options.subdomains = layer.subdomains;
  }
  return options;
}
