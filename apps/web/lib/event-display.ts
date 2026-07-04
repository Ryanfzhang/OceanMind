import type { EventOverlay } from "./types";

const MIN_RECTANGLE_SPAN_DEG = 1e-6;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function isFiniteNumber(value: number | undefined) {
  return typeof value === "number" && Number.isFinite(value);
}

export function hasRenderableRectangleBounds(event: EventOverlay): boolean {
  const bounds = event.bounds;
  if (event.shape !== "rectangle" || !bounds) {
    return false;
  }

  if (
    !isFiniteNumber(bounds.lonMin) ||
    !isFiniteNumber(bounds.lonMax) ||
    !isFiniteNumber(bounds.latMin) ||
    !isFiniteNumber(bounds.latMax)
  ) {
    return false;
  }

  return (
    Math.abs(bounds.lonMax - bounds.lonMin) > MIN_RECTANGLE_SPAN_DEG &&
    Math.abs(bounds.latMax - bounds.latMin) > MIN_RECTANGLE_SPAN_DEG
  );
}

export function shouldSimplifyEventOverlays(events: EventOverlay[]): boolean {
  if (events.length < 40) {
    return false;
  }

  const rectangularCount = events.filter((event) => event.shape === "rectangle" && event.bounds).length;
  return rectangularCount / events.length >= 0.5;
}

function roundedCoordinateKey(value: number | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "na";
}

function overlayGroupKey(event: EventOverlay) {
  const bounds = event.bounds;
  const boundsKey = bounds
    ? [
        roundedCoordinateKey(bounds.lonMin),
        roundedCoordinateKey(bounds.lonMax),
        roundedCoordinateKey(bounds.latMin),
        roundedCoordinateKey(bounds.latMax),
      ].join(",")
    : "none";
  return [
    event.eventType,
    event.severity ?? "",
    event.shape ?? "",
    roundedCoordinateKey(event.center.lon),
    roundedCoordinateKey(event.center.lat),
    boundsKey,
  ].join("|");
}

function eventTimeLabel(event: EventOverlay) {
  return event.timestamp ?? event.endTimestamp ?? "";
}

export function groupEventOverlaysForMap(events: EventOverlay[]): EventOverlay[] {
  if (!shouldSimplifyEventOverlays(events)) {
    return events;
  }

  const groups = new Map<
    string,
    {
      base: EventOverlay;
      count: number;
      firstTime: string;
      lastTime: string;
    }
  >();

  events.forEach((event) => {
    const key = overlayGroupKey(event);
    const timeLabel = eventTimeLabel(event);
    const existing = groups.get(key);
    if (!existing) {
      groups.set(key, {
        base: event,
        count: 1,
        firstTime: timeLabel,
        lastTime: timeLabel,
      });
      return;
    }

    existing.count += 1;
    if (timeLabel && (!existing.firstTime || timeLabel < existing.firstTime)) {
      existing.firstTime = timeLabel;
    }
    if (timeLabel && (!existing.lastTime || timeLabel > existing.lastTime)) {
      existing.lastTime = timeLabel;
    }
  });

  return Array.from(groups.values()).map((group) => {
    if (group.count === 1) {
      return group.base;
    }

    const timeRange =
      group.firstTime && group.lastTime && group.firstTime !== group.lastTime
        ? [`First: ${group.firstTime}`, `Last: ${group.lastTime}`]
        : group.firstTime
          ? [`Time: ${group.firstTime}`]
          : [];

    return {
      ...group.base,
      title: `${group.base.title} (${group.count} events)`,
      details: [`Occurrences: ${group.count}`, ...timeRange, ...group.base.details],
      occurrenceCount: group.count,
    };
  });
}

export function eventMarkerRadiusPx(event: EventOverlay): number {
  if (
    typeof event.occurrenceCount === "number" &&
    Number.isFinite(event.occurrenceCount) &&
    event.occurrenceCount > 1
  ) {
    return clamp(5 + Math.sqrt(event.occurrenceCount) * 0.75, 7, 20);
  }

  if (typeof event.radiusKm === "number" && Number.isFinite(event.radiusKm) && event.radiusKm > 0) {
    return clamp(3 + Math.sqrt(event.radiusKm) * 0.35, 5, 13);
  }

  if (event.bounds) {
    const lonSpan = Math.abs(event.bounds.lonMax - event.bounds.lonMin);
    const latSpan = Math.abs(event.bounds.latMax - event.bounds.latMin);
    const approxSpanDeg = Math.max(lonSpan, latSpan);
    return clamp(4 + approxSpanDeg * 4, 5, 12);
  }

  return 6;
}
