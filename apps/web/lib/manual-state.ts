import type { ManualViewState, RegionBounds } from "@/lib/types";

function formatBound(value: number) {
  const rounded = Number.parseFloat(value.toFixed(2));
  return Number.isInteger(rounded) ? `${rounded}` : `${rounded}`;
}

export function formatRegionLabel(bounds: RegionBounds) {
  return `${formatBound(bounds.lonMin)}-${formatBound(bounds.lonMax)}E · ${formatBound(bounds.latMin)}-${formatBound(bounds.latMax)}N`;
}

export function withRegionBounds(state: ManualViewState, regionBounds: RegionBounds): ManualViewState {
  const normalizedBounds = {
    lonMin: Math.min(regionBounds.lonMin, regionBounds.lonMax),
    lonMax: Math.max(regionBounds.lonMin, regionBounds.lonMax),
    latMin: Math.min(regionBounds.latMin, regionBounds.latMax),
    latMax: Math.max(regionBounds.latMin, regionBounds.latMax)
  };
  return {
    ...state,
    regionBounds: normalizedBounds,
    regionLabel: formatRegionLabel(normalizedBounds)
  };
}
