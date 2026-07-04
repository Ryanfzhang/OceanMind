import { NextRequest, NextResponse } from "next/server";
import {
  LIGHT_BASEMAP,
  OSM_FALLBACK_BASEMAP,
  type BasemapConfig,
  type BasemapRegion,
  type MapConfigResponse,
} from "@/lib/basemap";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const CHINA_REGION_COUNTRIES = new Set(["CN", "HK", "MO"]);
const COUNTRY_HEADERS = [
  "cf-ipcountry",
  "x-vercel-ip-country",
  "x-country-code",
  "x-geo-country",
  "cloudfront-viewer-country",
  "fastly-client-country",
  "x-appengine-country",
  "x-client-country",
] as const;

function envValue(name: string) {
  const value = process.env[name]?.trim();
  return value ? value : null;
}

function envFlag(name: string, defaultValue = false) {
  const value = envValue(name)?.toLowerCase();
  if (!value) {
    return defaultValue;
  }
  return value === "1" || value === "true" || value === "yes" || value === "on";
}

function normalizeCountry(value: string | null) {
  const country = value?.split(",")[0]?.trim().toUpperCase();
  if (!country || country === "XX" || country === "ZZ") {
    return null;
  }
  return /^[A-Z]{2}$/.test(country) ? country : null;
}

function detectCountry(headers: Headers): { country: string | null; source: string | null } {
  const customHeader = envValue("MAP_COUNTRY_HEADER")?.toLowerCase();
  if (customHeader) {
    const country = normalizeCountry(headers.get(customHeader));
    if (country) {
      return { country, source: customHeader };
    }
  }

  for (const headerName of COUNTRY_HEADERS) {
    const country = normalizeCountry(headers.get(headerName));
    if (country) {
      return { country, source: headerName };
    }
  }
  return { country: null, source: null };
}

function normalizeRegion(value: string | null): BasemapRegion | null {
  const region = value?.trim().toLowerCase();
  return region === "cn" || region === "global" ? region : null;
}

function selectRegion(request: NextRequest, country: string | null): BasemapRegion {
  const queryRegion = normalizeRegion(request.nextUrl.searchParams.get("region"));
  if (queryRegion) {
    return queryRegion;
  }

  const envRegion = normalizeRegion(envValue("MAP_REGION_OVERRIDE"));
  if (envRegion) {
    return envRegion;
  }

  if (!envFlag("MAP_ENABLE_REGION_SWITCH")) {
    return "global";
  }

  return country && CHINA_REGION_COUNTRIES.has(country) ? "cn" : "global";
}

function parseMaxZoom(value: string | null, fallback: number) {
  if (!value) {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseSubdomains(value: string | null) {
  if (!value) {
    return undefined;
  }
  if (value.includes(",")) {
    return value
      .split(",")
      .map((entry) => entry.trim())
      .filter(Boolean);
  }
  return value.trim();
}

function buildCustomChinaBasemap(): BasemapConfig | null {
  const url = envValue("MAP_TILE_CN_URL");
  if (!url) {
    return null;
  }

  const maxZoom = parseMaxZoom(envValue("MAP_TILE_CN_MAX_ZOOM"), 18);
  const subdomains = parseSubdomains(envValue("MAP_TILE_CN_SUBDOMAINS"));
  const overlayUrl = envValue("MAP_TILE_CN_OVERLAY_URL");

  return {
    id: envValue("MAP_TILE_CN_ID") ?? "china-custom",
    label: envValue("MAP_TILE_CN_LABEL") ?? "China regional basemap",
    region: "cn",
    url,
    attribution: envValue("MAP_TILE_CN_ATTRIBUTION") ?? "",
    maxZoom,
    subdomains,
    overlays: overlayUrl
      ? [
          {
            id: envValue("MAP_TILE_CN_OVERLAY_ID") ?? "china-custom-overlay",
            label: envValue("MAP_TILE_CN_OVERLAY_LABEL") ?? "China regional basemap overlay",
            region: "cn",
            url: overlayUrl,
            attribution: envValue("MAP_TILE_CN_OVERLAY_ATTRIBUTION") ?? "",
            maxZoom,
            subdomains: parseSubdomains(envValue("MAP_TILE_CN_OVERLAY_SUBDOMAINS")) ?? subdomains,
          },
        ]
      : undefined,
  };
}

function tiandituTileUrl(layer: "vec" | "cva", token: string) {
  return [
    `https://t{s}.tianditu.gov.cn/${layer}_w/wmts?SERVICE=WMTS`,
    "REQUEST=GetTile",
    "VERSION=1.0.0",
    `LAYER=${layer}`,
    "STYLE=default",
    "TILEMATRIXSET=w",
    "FORMAT=tiles",
    "TILEMATRIX={z}",
    "TILEROW={y}",
    "TILECOL={x}",
    `tk=${encodeURIComponent(token)}`,
  ].join("&");
}

function buildTiandituBasemap(): BasemapConfig | null {
  const token = envValue("TIANDITU_TOKEN") ?? envValue("NEXT_PUBLIC_TIANDITU_TOKEN");
  if (!token) {
    return null;
  }

  const maxZoom = parseMaxZoom(envValue("MAP_TILE_CN_MAX_ZOOM"), 18);
  const subdomains = parseSubdomains(envValue("MAP_TILE_CN_SUBDOMAINS")) ?? "01234567";
  const attribution = envValue("MAP_TILE_CN_ATTRIBUTION") ?? '&copy; <a href="https://www.tianditu.gov.cn/">Tianditu</a>';

  return {
    id: "tianditu-vector",
    label: "Tianditu Vector",
    region: "cn",
    url: tiandituTileUrl("vec", token),
    attribution,
    maxZoom,
    subdomains,
    overlays: [
      {
        id: "tianditu-annotation",
        label: "Tianditu Annotation",
        region: "cn",
        url: tiandituTileUrl("cva", token),
        attribution: "",
        maxZoom,
        subdomains,
      },
    ],
  };
}

function buildChinaBasemap(): BasemapConfig | null {
  return buildCustomChinaBasemap() ?? buildTiandituBasemap();
}

export async function GET(request: NextRequest) {
  const { country, source } = detectCountry(request.headers);
  const region = selectRegion(request, country);
  const chinaBasemap = region === "cn" ? buildChinaBasemap() : null;
  const basemap = region === "cn" ? chinaBasemap ?? LIGHT_BASEMAP : LIGHT_BASEMAP;
  const reason =
    region === "cn" && !chinaBasemap
      ? "china_basemap_not_configured"
      : country && CHINA_REGION_COUNTRIES.has(country) && !envFlag("MAP_ENABLE_REGION_SWITCH")
        ? "region_switch_disabled"
        : undefined;

  const response: MapConfigResponse = {
    region,
    detectedCountry: country,
    detectionSource: source,
    basemap,
    fallback: OSM_FALLBACK_BASEMAP,
    reason,
  };

  return NextResponse.json(response, {
    headers: {
      "Cache-Control": "no-store",
    },
  });
}
