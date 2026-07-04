import type {
  AssistantMessagePayload,
  ChatMessage,
  DatasetInfo,
  MapFieldData,
  PlanStep,
  ResultCardSummary,
  ResultDetailSection,
  ResultMetric,
  ScientificFinding,
  SourceCard,
  StepCard,
  WorkspaceData,
} from "./types";
import {
  colormapRGB,
  formatMapColorbarValue,
  resolveMapColorScale,
} from "./map-field-preview";

export type ReportFigureSnapshot = {
  title?: string;
  mime_type: string;
  data_base64: string;
  width?: number;
  height?: number;
};

export type ReportResultCardPayload = {
  title: string;
  headline: string;
  description: string;
  metrics: ResultMetric[];
  interpretation?: string;
  detail_sections?: ResultDetailSection[];
};

export type ReportStepCardPayload = {
  human_label: string;
  technical_label: string;
  status: StepCard["status"];
  interpretation?: string;
  error?: string;
  results: ReportResultCardPayload[];
};

export type ReportTurnPayload = {
  user_query: string;
  assistant_status: NonNullable<AssistantMessagePayload["state"]>;
  assistant_summary: string;
  plan_steps: PlanStep[];
  step_cards: ReportStepCardPayload[];
  findings: ScientificFinding[];
  source_cards: SourceCard[];
  primary_figure?: ReportFigureSnapshot | null;
};

export type ConversationReportPayload = {
  conversation_id?: string;
  exported_at: string;
  dataset_info?: DatasetInfo | null;
  turns: ReportTurnPayload[];
};

type ReportPayloadOptions = {
  conversationId?: string;
  datasetInfo?: DatasetInfo | null;
  exportedAt?: string;
  messages: ChatMessage[];
  figureSnapshotsByMessageId?: Record<string, ReportFigureSnapshot | null>;
};

const SNAPSHOT_WIDTH = 1200;
const SNAPSHOT_HEIGHT = 680;

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isRenderableMap(workspaceData?: WorkspaceData) {
  return Boolean(workspaceData?.mapField);
}

function isRenderableLineSeries(workspaceData?: WorkspaceData) {
  return Boolean(workspaceData && (workspaceData.resultSeries.length > 0 || workspaceData.profileSeries.length > 0 || workspaceData.eofPcSeries.length > 0));
}

function isRenderableScatter(workspaceData?: WorkspaceData) {
  return Boolean(workspaceData?.tsDiagramPoints.length);
}

function isRenderableHistogram(workspaceData?: WorkspaceData) {
  return Boolean(workspaceData?.histogramBins.length);
}

function isRenderableHeatmap(workspaceData?: WorkspaceData) {
  return Boolean(workspaceData?.hovmollerRows.length);
}

function isRenderableFigure(card: ResultCardSummary) {
  const workspaceData = card.workspaceData as WorkspaceData | undefined;
  return (
    isRenderableMap(workspaceData) ||
    isRenderableLineSeries(workspaceData) ||
    isRenderableScatter(workspaceData) ||
    isRenderableHistogram(workspaceData) ||
    isRenderableHeatmap(workspaceData)
  );
}

function serializeResultCard(card: ResultCardSummary): ReportResultCardPayload {
  return {
    title: card.title,
    headline: card.headline,
    description: card.description,
    metrics: Array.isArray(card.metrics) ? card.metrics : [],
    interpretation: card.interpretation,
    detail_sections: card.detailSections,
  };
}

function serializeStepCard(step: StepCard): ReportStepCardPayload {
  return {
    human_label: step.human_label,
    technical_label: step.technical_label,
    status: step.status,
    interpretation: step.interpretation,
    error: step.error,
    results: step.results.map((result) => serializeResultCard(result)),
  };
}

export function selectPrimaryFigureCard(payload?: AssistantMessagePayload): ResultCardSummary | null {
  const cards = payload?.resultCards ?? [];
  if (cards.length === 0) {
    return null;
  }

  if (payload?.activeResultId) {
    const activeCard = cards.find((card) => card.id === payload.activeResultId);
    if (activeCard && isRenderableFigure(activeCard)) {
      return activeCard;
    }
  }

  return cards.find((card) => isRenderableFigure(card)) ?? null;
}

export function buildConversationReportPayload({
  conversationId,
  datasetInfo,
  exportedAt,
  messages,
  figureSnapshotsByMessageId = {},
}: ReportPayloadOptions): ConversationReportPayload {
  const turns: ReportTurnPayload[] = [];
  let pendingUserQuery = "";

  for (const message of messages) {
    if (message.role === "user") {
      pendingUserQuery = message.text;
      continue;
    }

    const payload = message.payload;
    if (!payload) {
      continue;
    }

    turns.push({
      user_query: pendingUserQuery,
      assistant_status: payload.state,
      assistant_summary: payload.summary,
      plan_steps: payload.planSteps ?? [],
      step_cards: (payload.stepCards ?? []).map((step) => serializeStepCard(step)),
      findings: payload.findings ?? [],
      source_cards: payload.sourceCards ?? [],
      primary_figure: figureSnapshotsByMessageId[message.id] ?? null,
    });
    pendingUserQuery = "";
  }

  return {
    conversation_id: conversationId,
    exported_at: exportedAt ?? new Date().toISOString(),
    dataset_info: datasetInfo ?? undefined,
    turns,
  };
}

export function buildReportFilename(exportedAt: string) {
  const safe = exportedAt.replace(/[:.]/g, "-");
  return `oceanmind-conversation-${safe}.pdf`;
}

function createSnapshotCanvas(width = SNAPSHOT_WIDTH, height = SNAPSHOT_HEIGHT) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Canvas rendering is unavailable.");
  }

  context.fillStyle = "#fbf8ef";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#10253d";
  context.font = "600 34px sans-serif";
  context.fillText("OceanMind Figure Snapshot", 48, 60);

  return { canvas, context };
}

function finalizeSnapshot(canvas: HTMLCanvasElement, title: string): ReportFigureSnapshot {
  const dataUrl = canvas.toDataURL("image/png");
  const [, dataBase64 = ""] = dataUrl.split(",", 2);
  return {
    title,
    mime_type: "image/png",
    data_base64: dataBase64,
    width: canvas.width,
    height: canvas.height,
  };
}

function valueRange(values: number[]) {
  const finiteValues = values.filter((value) => Number.isFinite(value));
  if (finiteValues.length === 0) {
    return { min: 0, max: 1 };
  }
  const min = Math.min(...finiteValues);
  const max = Math.max(...finiteValues);
  if (Math.abs(max - min) < 1e-9) {
    return { min: min - 1, max: max + 1 };
  }
  return { min, max };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

const HOVMOLLER_COLORMAP_STOPS: [number, [number, number, number]][] = [
  [0.0, [32, 85, 191]],
  [0.25, [112, 166, 228]],
  [0.5, [255, 255, 255]],
  [0.75, [248, 154, 146]],
  [1.0, [213, 36, 42]],
];

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

function quantileSorted(values: number[], q: number) {
  if (values.length === 0) return Number.NaN;
  if (values.length === 1) return values[0];
  const position = clamp(q, 0, 1) * (values.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return values[lower];
  return values[lower] + (values[upper] - values[lower]) * (position - lower);
}

function robustSymmetricLimit(values: number[]) {
  const finiteValues = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (finiteValues.length === 0) return 1;
  const limit = Math.max(Math.abs(quantileSorted(finiteValues, 0.02)), Math.abs(quantileSorted(finiteValues, 0.98)), 1e-12);
  return Number.isFinite(limit) ? limit : 1;
}

function hovmollerColorRGB(value: number, limit: number): [number, number, number] {
  if (!Number.isFinite(value)) {
    return [226, 232, 240];
  }
  return interpolateStops(HOVMOLLER_COLORMAP_STOPS, (clamp(value, -limit, limit) + limit) / Math.max(2 * limit, 1e-12));
}

function hovmollerDepthValue(row: WorkspaceData["hovmollerRows"][number]) {
  const raw = typeof row.depthValue === "number" ? row.depthValue : Number.parseFloat(row.depthLabel);
  return Number.isFinite(raw) ? Math.abs(raw) : Number.NaN;
}

function hovmollerDepthBands(rows: WorkspaceData["hovmollerRows"], frameHeight: number) {
  const depths = rows.map(hovmollerDepthValue);
  const finiteDepths = depths.filter(Number.isFinite);
  if (rows.length === 0) {
    return [];
  }
  if (finiteDepths.length <= 1) {
    return rows.map(() => ({ y: 0, height: frameHeight }));
  }

  const depthTop = Math.min(...finiteDepths);
  const depthBottom = Math.max(...finiteDepths);
  const span = Math.max(depthBottom - depthTop, 1e-12);
  return depths.map((depth, index) => {
    if (!Number.isFinite(depth)) {
      const fallbackHeight = frameHeight / Math.max(rows.length, 1);
      return { y: index * fallbackHeight, height: fallbackHeight };
    }

    const previous = index > 0 && Number.isFinite(depths[index - 1]) ? (depths[index - 1] + depth) / 2 : depthTop;
    const next = index < depths.length - 1 && Number.isFinite(depths[index + 1]) ? (depth + depths[index + 1]) / 2 : depthBottom;
    const y0 = ((Math.max(depthTop, previous) - depthTop) / span) * frameHeight;
    const y1 = ((Math.min(depthBottom, next) - depthTop) / span) * frameHeight;
    return { y: y0, height: Math.max(y1 - y0, 1) };
  });
}

function drawFrame(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  title: string,
  subtitle: string,
) {
  context.fillStyle = "#16324f";
  context.font = "700 28px sans-serif";
  context.fillText(title, 48, 106);
  context.fillStyle = "#53657a";
  context.font = "400 18px sans-serif";
  if (subtitle) {
    context.fillText(subtitle, 48, 136);
  }
  context.strokeStyle = "#d1d9e3";
  context.lineWidth = 2;
  context.strokeRect(48, 170, width - 96, height - 230);
}

function drawLineSnapshot(
  title: string,
  subtitle: string,
  points: Array<{ xLabel: string; value: number }>,
  options?: { invertY?: boolean },
) {
  const invertY = options?.invertY ?? false;
  const { canvas, context } = createSnapshotCanvas();
  drawFrame(context, canvas.width, canvas.height, title, subtitle);

  const frameLeft = 90;
  const frameTop = 200;
  const frameWidth = canvas.width - 180;
  const frameHeight = canvas.height - 300;
  const values = points.map((point) => point.value);
  const range = valueRange(values);

  context.strokeStyle = "#c7d3df";
  context.lineWidth = 1;
  for (let index = 0; index < 5; index += 1) {
    const y = frameTop + (frameHeight / 4) * index;
    context.beginPath();
    context.moveTo(frameLeft, y);
    context.lineTo(frameLeft + frameWidth, y);
    context.stroke();
  }

  context.strokeStyle = "#1d5b79";
  context.lineWidth = 4;
  context.beginPath();
  points.forEach((point, index) => {
    const x = frameLeft + (points.length <= 1 ? 0 : (frameWidth * index) / (points.length - 1));
    const normalized = (point.value - range.min) / (range.max - range.min);
    const y = invertY
      ? frameTop + normalized * frameHeight
      : frameTop + frameHeight - normalized * frameHeight;
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });
  context.stroke();

  context.fillStyle = "#1d5b79";
  points.forEach((point, index) => {
    const x = frameLeft + (points.length <= 1 ? 0 : (frameWidth * index) / (points.length - 1));
    const normalized = (point.value - range.min) / (range.max - range.min);
    const y = invertY
      ? frameTop + normalized * frameHeight
      : frameTop + frameHeight - normalized * frameHeight;
    context.beginPath();
    context.arc(x, y, 5, 0, Math.PI * 2);
    context.fill();
    if (index === 0 || index === points.length - 1 || index % Math.max(1, Math.floor(points.length / 4)) === 0) {
      context.fillStyle = "#53657a";
      context.font = "400 14px sans-serif";
      context.fillText(point.xLabel, x - 20, frameTop + frameHeight + 28);
      context.fillStyle = "#1d5b79";
    }
  });

  return finalizeSnapshot(canvas, title);
}

function drawHistogramSnapshot(title: string, subtitle: string, values: Array<{ label: string; value: number }>) {
  const { canvas, context } = createSnapshotCanvas();
  drawFrame(context, canvas.width, canvas.height, title, subtitle);

  const frameLeft = 90;
  const frameTop = 200;
  const frameWidth = canvas.width - 180;
  const frameHeight = canvas.height - 300;
  const maxValue = Math.max(...values.map((item) => item.value), 1);
  const barWidth = Math.max(frameWidth / Math.max(values.length, 1) - 12, 18);

  values.forEach((item, index) => {
    const left = frameLeft + index * (barWidth + 12);
    const barHeight = (item.value / maxValue) * frameHeight;
    context.fillStyle = "#2a6f97";
    context.fillRect(left, frameTop + frameHeight - barHeight, barWidth, barHeight);
    context.fillStyle = "#53657a";
    context.font = "400 13px sans-serif";
    context.fillText(item.label, left, frameTop + frameHeight + 24);
  });

  return finalizeSnapshot(canvas, title);
}

function drawScatterSnapshot(
  title: string,
  subtitle: string,
  points: Array<{ x: number; y: number }>,
  xLabel: string,
  yLabel: string,
) {
  const { canvas, context } = createSnapshotCanvas();
  drawFrame(context, canvas.width, canvas.height, title, subtitle);

  const frameLeft = 90;
  const frameTop = 200;
  const frameWidth = canvas.width - 180;
  const frameHeight = canvas.height - 300;
  const xRange = valueRange(points.map((point) => point.x));
  const yRange = valueRange(points.map((point) => point.y));

  context.fillStyle = "#16324f";
  context.font = "500 16px sans-serif";
  context.fillText(xLabel, frameLeft + frameWidth - 90, frameTop + frameHeight + 38);
  context.save();
  context.translate(frameLeft - 54, frameTop + 110);
  context.rotate(-Math.PI / 2);
  context.fillText(yLabel, 0, 0);
  context.restore();

  points.forEach((point) => {
    const x = frameLeft + ((point.x - xRange.min) / (xRange.max - xRange.min)) * frameWidth;
    const y = frameTop + frameHeight - ((point.y - yRange.min) / (yRange.max - yRange.min)) * frameHeight;
    context.fillStyle = "#bc4749";
    context.beginPath();
    context.arc(x, y, 6, 0, Math.PI * 2);
    context.fill();
  });

  return finalizeSnapshot(canvas, title);
}

function drawHovmollerSnapshot(
  title: string,
  subtitle: string,
  rows: WorkspaceData["hovmollerRows"],
  timeLabels: string[],
) {
  const { canvas, context } = createSnapshotCanvas();
  drawFrame(context, canvas.width, canvas.height, title, subtitle);
  const finiteRows = rows
    .filter((row) => row.values.some(isFiniteNumber) && Number.isFinite(hovmollerDepthValue(row)) && hovmollerDepthValue(row) < 9000)
    .sort((a, b) => hovmollerDepthValue(a) - hovmollerDepthValue(b));

  const frameLeft = 90;
  const frameTop = 200;
  const frameWidth = canvas.width - 180;
  const frameHeight = canvas.height - 300;
  const allValues = finiteRows.flatMap((row) => row.values.filter(isFiniteNumber));
  const limit = robustSymmetricLimit(allValues);
  const columnCount = Math.max(...finiteRows.map((row) => row.values.length), timeLabels.length, 1);
  const cellWidth = frameWidth / columnCount;
  const depthBands = hovmollerDepthBands(finiteRows, frameHeight);

  finiteRows.forEach((row, rowIndex) => {
    const band = depthBands[rowIndex] ?? { y: 0, height: frameHeight / Math.max(finiteRows.length, 1) };
    row.values.forEach((value, columnIndex) => {
      const [red, green, blue] = hovmollerColorRGB(isFiniteNumber(value) ? value : Number.NaN, limit);
      context.fillStyle = `rgb(${red}, ${green}, ${blue})`;
      context.fillRect(
        frameLeft + columnIndex * cellWidth,
        frameTop + band.y,
        cellWidth,
        band.height,
      );
    });
  });

  context.fillStyle = "#53657a";
  context.font = "400 13px sans-serif";
  finiteRows.forEach((row, rowIndex) => {
    const band = depthBands[rowIndex] ?? { y: 0, height: frameHeight / Math.max(finiteRows.length, 1) };
    context.fillText(row.depthLabel, 54, frameTop + band.y + Math.min(18, band.height));
  });
  timeLabels.forEach((label, index) => {
    if (index === 0 || index === timeLabels.length - 1 || index % Math.max(1, Math.floor(timeLabels.length / 4)) === 0) {
      context.fillText(label, frameLeft + index * cellWidth, frameTop + frameHeight + 24);
    }
  });

  return finalizeSnapshot(canvas, title);
}

function drawMapSnapshot(title: string, field: MapFieldData) {
  const { canvas, context } = createSnapshotCanvas();
  drawFrame(context, canvas.width, canvas.height, title, field.label || field.variable);

  const frameLeft = 90;
  const frameTop = 200;
  const frameWidth = canvas.width - 180;
  const frameHeight = canvas.height - 300;

  const rows = field.values.length;
  const cols = Math.max(...field.values.map((row) => row.length), 0);
  const allValues = field.values.flat().filter((value) => Number.isFinite(value));
  const scale = resolveMapColorScale(field);
  const range = scale ?? valueRange(allValues);
  const rangeSpan = Math.max(range.max - range.min, 1e-9);
  const cellWidth = frameWidth / Math.max(cols, 1);
  const cellHeight = frameHeight / Math.max(rows, 1);

  for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < cols; columnIndex += 1) {
      const value = field.values[rowIndex]?.[columnIndex];
      if (!isFiniteNumber(value)) {
        continue;
      }
      const normalized = (value - range.min) / rangeSpan;
      const [red, green, blue] = colormapRGB(normalized, scale?.colormap);
      context.fillStyle = `rgb(${red}, ${green}, ${blue})`;
      context.fillRect(
        frameLeft + columnIndex * cellWidth,
        frameTop + (rows - rowIndex - 1) * cellHeight,
        cellWidth,
        cellHeight,
      );
    }
  }

  context.fillStyle = "#53657a";
  context.font = "400 14px sans-serif";
  context.fillText(`Lon: ${field.lon[0]} to ${field.lon[field.lon.length - 1]}`, frameLeft, frameTop + frameHeight + 26);
  context.fillText(
    `Lat: ${field.lat[0]} to ${field.lat[field.lat.length - 1]}`,
    frameLeft + 300,
    frameTop + frameHeight + 26,
  );
  drawMapColorbar(context, field, frameLeft, frameTop + frameHeight + 56, frameWidth);

  return finalizeSnapshot(canvas, title);
}

function drawMapColorbar(
  context: CanvasRenderingContext2D,
  field: MapFieldData,
  left: number,
  top: number,
  width: number,
) {
  const scale = resolveMapColorScale(field);
  if (!scale) {
    return;
  }

  const rampHeight = 14;
  const steps = Math.max(2, Math.floor(width));
  for (let index = 0; index < steps; index += 1) {
    const t = index / (steps - 1);
    const [red, green, blue] = colormapRGB(t, scale.colormap);
    context.fillStyle = `rgb(${red}, ${green}, ${blue})`;
    context.fillRect(left + (index / steps) * width, top, Math.ceil(width / steps) + 1, rampHeight);
  }

  context.strokeStyle = "rgba(83, 101, 122, 0.45)";
  context.strokeRect(left, top, width, rampHeight);
  context.fillStyle = "#53657a";
  context.font = "400 14px sans-serif";
  context.fillText(formatMapColorbarValue(scale.min), left, top + rampHeight + 20);
  const maxLabel = formatMapColorbarValue(scale.max);
  const maxWidth = context.measureText(maxLabel).width;
  context.fillText(maxLabel, left + width - maxWidth, top + rampHeight + 20);
  const label = [scale.label, scale.units].filter(Boolean).join(" · ");
  if (label) {
    context.fillStyle = "#16324f";
    context.font = "600 14px sans-serif";
    context.fillText(label, left, top - 8);
  }
}

export async function captureReportFigureSnapshot(
  card: ResultCardSummary,
  workspaceData: WorkspaceData,
): Promise<ReportFigureSnapshot | null> {
  if (typeof document === "undefined") {
    return null;
  }

  if (workspaceData.mapField?.contourImage) {
    return {
      title: card.title,
      mime_type: "image/png",
      data_base64: workspaceData.mapField.contourImage,
      width: SNAPSHOT_WIDTH,
      height: SNAPSHOT_HEIGHT,
    };
  }

  if (workspaceData.mapField) {
    return drawMapSnapshot(card.title, workspaceData.mapField);
  }

  if (workspaceData.resultSeries.length > 0) {
    return drawLineSnapshot(
      card.title,
      card.headline,
      workspaceData.resultSeries.map((point) => ({ xLabel: point.label, value: point.value })),
    );
  }

  if (workspaceData.profileSeries.length > 0) {
    return drawLineSnapshot(
      card.title,
      card.headline,
      workspaceData.profileSeries.map((point) => ({ xLabel: `${point.depth} m`, value: point.value })),
      { invertY: true },
    );
  }

  if (workspaceData.eofPcSeries.length > 0) {
    return drawLineSnapshot(
      card.title,
      card.headline,
      workspaceData.eofPcSeries.map((point) => ({ xLabel: point.day, value: point.value })),
    );
  }

  if (workspaceData.histogramBins.length > 0) {
    return drawHistogramSnapshot(card.title, card.headline, workspaceData.histogramBins);
  }

  if (workspaceData.tsDiagramPoints.length > 0) {
    return drawScatterSnapshot(
      card.title,
      card.headline,
      workspaceData.tsDiagramPoints.map((point) => ({
        x: point.temperature,
        y: point.salinity,
      })),
      workspaceData.tsDiagramTemperatureLabel ?? "Temperature",
      workspaceData.tsDiagramSalinityLabel ?? "Salinity",
    );
  }

  if (workspaceData.hovmollerRows.length > 0) {
    return drawHovmollerSnapshot(
      card.title,
      card.headline,
      workspaceData.hovmollerRows,
      workspaceData.hovmollerTimeLabels ?? [],
    );
  }

  return null;
}

export async function capturePrimaryFigureSnapshotForMessage(
  message: ChatMessage,
): Promise<ReportFigureSnapshot | null> {
  if (message.role !== "assistant" || message.payload?.state !== "completed") {
    return null;
  }

  const card = selectPrimaryFigureCard(message.payload);
  const workspaceData = card?.workspaceData as WorkspaceData | undefined;
  if (!card || !workspaceData) {
    return null;
  }
  return captureReportFigureSnapshot(card, workspaceData);
}
