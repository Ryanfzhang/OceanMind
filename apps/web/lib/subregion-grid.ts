import type { GeoBounds, MapFieldData, SubregionGridCell } from "./types";

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

export function normalizeMechanism(value?: string | null) {
  return String(value ?? "").trim().toLowerCase();
}

export function mechanismDisplayName(value?: string | null) {
  switch (normalizeMechanism(value)) {
    case "front_proximity":
      return "Front";
    case "eddy_influence":
      return "Eddy";
    case "flow_context":
      return "Flow";
    case "background_like":
      return "Background-like";
    default:
      return value ? value.replaceAll("_", " ") : "Unknown";
  }
}

export function mechanismAbbreviation(value?: string | null) {
  switch (normalizeMechanism(value)) {
    case "front_proximity":
      return "FRT";
    case "eddy_influence":
      return "EDD";
    case "flow_context":
      return "FLW";
    case "background_like":
      return "BG";
    default:
      if (!value) {
        return "NA";
      }
      const tokens = value
        .replaceAll("-", "_")
        .split("_")
        .map((token) => token.trim())
        .filter(Boolean);
      if (tokens.length === 0) {
        return "NA";
      }
      return tokens
        .slice(0, 3)
        .map((token) => token[0]?.toUpperCase() ?? "")
        .join("");
  }
}

export function supportDisplay(value?: string | null) {
  switch (String(value ?? "").trim().toLowerCase()) {
    case "supported":
      return "Supported";
    case "limited":
      return "Limited";
    case "untestable":
      return "Untestable";
    default:
      return value ? value : "Unknown";
  }
}

export function mechanismStrokeColor(mechanism?: string | null, status?: string | null) {
  if (String(status ?? "").trim().toLowerCase() !== "ok") {
    return "#7b8794";
  }
  switch (normalizeMechanism(mechanism)) {
    case "front_proximity":
      return "#c96f12";
    case "eddy_influence":
      return "#1f6feb";
    case "flow_context":
      return "#158f77";
    case "background_like":
      return "#6b7280";
    default:
      return "#475569";
  }
}

export function mechanismTint(mechanism?: string | null, status?: string | null) {
  if (String(status ?? "").trim().toLowerCase() !== "ok") {
    return "rgba(123, 135, 148, 0.16)";
  }
  switch (normalizeMechanism(mechanism)) {
    case "front_proximity":
      return "rgba(201, 111, 18, 0.16)";
    case "eddy_influence":
      return "rgba(31, 111, 235, 0.16)";
    case "flow_context":
      return "rgba(21, 143, 119, 0.16)";
    case "background_like":
      return "rgba(107, 114, 128, 0.16)";
    default:
      return "rgba(71, 85, 105, 0.16)";
  }
}

export function gridCellDisplayName(cell: SubregionGridCell) {
  if (cell.categoryLabel) {
    return cell.categoryLabel;
  }
  if (cell.category) {
    return cell.category.replaceAll("_", " ");
  }
  return mechanismDisplayName(cell.dominantMechanism);
}

export function gridCellBadge(cell: SubregionGridCell) {
  if (cell.categoryShortLabel) {
    return cell.categoryShortLabel;
  }
  if (cell.category) {
    return mechanismAbbreviation(cell.category);
  }
  return mechanismAbbreviation(cell.dominantMechanism);
}

export function gridCellStrokeColor(cell: SubregionGridCell) {
  if (String(cell.status ?? "").trim().toLowerCase() !== "ok") {
    return "#7b8794";
  }
  if (cell.color) {
    return cell.color;
  }
  if (cell.category && cell.category !== "background") {
    return mechanismStrokeColor(cell.category, cell.status);
  }
  return mechanismStrokeColor(cell.dominantMechanism, cell.status);
}

export function gridCellTint(cell: SubregionGridCell) {
  if (String(cell.status ?? "").trim().toLowerCase() !== "ok") {
    return "rgba(123, 135, 148, 0.16)";
  }
  if (cell.color) {
    const match = /^#?([0-9a-f]{6})$/i.exec(cell.color);
    if (match) {
      const hex = match[1];
      const red = Number.parseInt(hex.slice(0, 2), 16);
      const green = Number.parseInt(hex.slice(2, 4), 16);
      const blue = Number.parseInt(hex.slice(4, 6), 16);
      return `rgba(${red}, ${green}, ${blue}, 0.16)`;
    }
  }
  if (cell.category && cell.category !== "background") {
    return mechanismTint(cell.category, cell.status);
  }
  return mechanismTint(cell.dominantMechanism, cell.status);
}

export function gridCellSupportText(cell: SubregionGridCell) {
  const status = String(cell.status ?? "").trim().toLowerCase();
  if (status !== "ok") {
    return status;
  }
  if (cell.valueLabel) {
    return cell.valueLabel;
  }
  return supportDisplay(cell.claimStrength);
}

export function fieldBounds(field: MapFieldData): GeoBounds | null {
  if (field.bounds && field.bounds.length === 2) {
    return {
      latMin: Math.min(field.bounds[0][0], field.bounds[1][0]),
      latMax: Math.max(field.bounds[0][0], field.bounds[1][0]),
      lonMin: Math.min(field.bounds[0][1], field.bounds[1][1]),
      lonMax: Math.max(field.bounds[0][1], field.bounds[1][1]),
    };
  }
  if (field.lon.length === 0 || field.lat.length === 0) {
    return null;
  }
  return {
    lonMin: Math.min(...field.lon),
    lonMax: Math.max(...field.lon),
    latMin: Math.min(...field.lat),
    latMax: Math.max(...field.lat),
  };
}

export function subregionOverlayBox(field: MapFieldData, cell: SubregionGridCell) {
  const bounds = fieldBounds(field);
  if (!bounds || !cell.bounds) {
    return null;
  }
  const lonSpan = Math.max(bounds.lonMax - bounds.lonMin, 1e-9);
  const latSpan = Math.max(bounds.latMax - bounds.latMin, 1e-9);
  return {
    left: clamp(((cell.bounds.lonMin - bounds.lonMin) / lonSpan) * 100, 0, 100),
    width: clamp(((cell.bounds.lonMax - cell.bounds.lonMin) / lonSpan) * 100, 0, 100),
    top: clamp(((bounds.latMax - cell.bounds.latMax) / latSpan) * 100, 0, 100),
    height: clamp(((cell.bounds.latMax - cell.bounds.latMin) / latSpan) * 100, 0, 100),
  };
}

export function sortSubregionCells(cells: SubregionGridCell[]) {
  return [...cells].sort((a, b) => {
    const aBounds = a.bounds;
    const bBounds = b.bounds;
    if (!aBounds || !bBounds) {
      return String(a.subregionId).localeCompare(String(b.subregionId));
    }
    if (aBounds.latMax !== bBounds.latMax) {
      return bBounds.latMax - aBounds.latMax;
    }
    return aBounds.lonMin - bBounds.lonMin;
  });
}
