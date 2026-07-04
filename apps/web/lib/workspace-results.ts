import type { ResultCardSummary, ResultDetailSection, StepCard, WorkspaceData } from "./types";

export const EMPTY_WORKSPACE_DATA: WorkspaceData = {
  referenceSeries: [],
  resultSeries: [],
  anomalySeries: [],
  timeseriesDisplayInfo: null,
  tsDiagramPoints: [],
  tsDiagramTemperatureLabel: "Temperature",
  tsDiagramSalinityLabel: "Salinity",
  tsDiagramColorLabel: null,
  tsDiagramColorRange: null,
  tsDiagramPointClasses: [],
  tsDiagramClassColorMap: {},
  tsDiagramWatermassBins: [],
  seriesLabels: {},
  profileSeries: [],
  profileMarkers: [],
  hovmollerRows: [],
  hovmollerTimeLabels: [],
  hovmollerDisplayInfo: null,
  hovmollerDepthIntegratedSeries: [],
  sectionRows: [],
  sectionDistanceKm: [],
  sectionAxisTitle: "",
  sectionSliceLabel: "",
  overlaySeries: [],
  histogramBins: [],
  eofVariance: [],
  eofPcSeries: [],
  eofModes: [],
  compositeFields: [],
  mapField: null,
  eventOverlays: [],
};

type SnakeCardFields = {
  owner_step_id?: string;
  workspace_data?: Partial<WorkspaceData>;
  detail_sections?: ResultDetailSection[];
};

export function normalizeWorkspaceData(workspaceData?: Partial<WorkspaceData>): WorkspaceData {
  return {
    ...EMPTY_WORKSPACE_DATA,
    ...workspaceData,
    referenceSeries: workspaceData?.referenceSeries ?? [],
    resultSeries: workspaceData?.resultSeries ?? [],
    anomalySeries: workspaceData?.anomalySeries ?? [],
    timeseriesDisplayInfo: workspaceData?.timeseriesDisplayInfo ?? null,
    tsDiagramPoints: workspaceData?.tsDiagramPoints ?? [],
    tsDiagramTemperatureLabel: workspaceData?.tsDiagramTemperatureLabel ?? "Temperature",
    tsDiagramSalinityLabel: workspaceData?.tsDiagramSalinityLabel ?? "Salinity",
    tsDiagramColorLabel: workspaceData?.tsDiagramColorLabel ?? null,
    tsDiagramColorRange: workspaceData?.tsDiagramColorRange ?? null,
    tsDiagramPointClasses: workspaceData?.tsDiagramPointClasses ?? [],
    tsDiagramClassColorMap: workspaceData?.tsDiagramClassColorMap ?? {},
    tsDiagramWatermassBins: workspaceData?.tsDiagramWatermassBins ?? [],
    seriesLabels: workspaceData?.seriesLabels ?? {},
    profileSeries: workspaceData?.profileSeries ?? [],
    profileMarkers: workspaceData?.profileMarkers ?? [],
    hovmollerRows: workspaceData?.hovmollerRows ?? [],
    hovmollerTimeLabels: workspaceData?.hovmollerTimeLabels ?? [],
    hovmollerDisplayInfo: workspaceData?.hovmollerDisplayInfo ?? null,
    hovmollerDepthIntegratedSeries: workspaceData?.hovmollerDepthIntegratedSeries ?? [],
    sectionRows: workspaceData?.sectionRows ?? [],
    sectionDistanceKm: workspaceData?.sectionDistanceKm ?? [],
    sectionAxisTitle: workspaceData?.sectionAxisTitle ?? "",
    sectionSliceLabel: workspaceData?.sectionSliceLabel ?? "",
    overlaySeries: workspaceData?.overlaySeries ?? [],
    histogramBins: workspaceData?.histogramBins ?? [],
    eofVariance: workspaceData?.eofVariance ?? [],
    eofPcSeries: workspaceData?.eofPcSeries ?? [],
    eofModes: workspaceData?.eofModes ?? [],
    compositeFields: workspaceData?.compositeFields ?? [],
    mapField: workspaceData?.mapField ?? null,
    eventOverlays: workspaceData?.eventOverlays ?? [],
  };
}

export function normalizeWorkspaceDataByResult(
  workspaceDataByResult?: Record<string, Partial<WorkspaceData>>,
): Record<string, WorkspaceData> {
  const normalized: Record<string, WorkspaceData> = {};
  if (!workspaceDataByResult) {
    return normalized;
  }
  for (const [resultId, value] of Object.entries(workspaceDataByResult)) {
    normalized[resultId] = normalizeWorkspaceData(value);
  }
  return normalized;
}

function normalizeDetailSections(value: ResultDetailSection[] | undefined): ResultDetailSection[] | undefined {
  if (!Array.isArray(value) || value.length === 0) {
    return undefined;
  }
  return value
    .map((section) => ({
      title: section.title,
      items: Array.isArray(section.items) ? section.items.filter((item) => typeof item === "string" && item.trim().length > 0) : [],
    }))
    .filter((section) => section.title.trim().length > 0 && section.items.length > 0);
}

export function hydrateResultCard(
  card: ResultCardSummary,
  workspaceDataByResult: Record<string, WorkspaceData>,
): ResultCardSummary {
  const snakeCard = card as ResultCardSummary & SnakeCardFields;
  const normalizedWorkspace =
    card.workspaceData ??
    (snakeCard.workspace_data ? normalizeWorkspaceData(snakeCard.workspace_data) : undefined) ??
    workspaceDataByResult[card.id];

  return {
    ...card,
    ownerStepId: card.ownerStepId ?? snakeCard.owner_step_id,
    workspaceData: normalizedWorkspace,
    detailSections: normalizeDetailSections(card.detailSections ?? snakeCard.detail_sections),
  };
}

export function hydrateResultCards(
  cards: ResultCardSummary[],
  workspaceDataByResult: Record<string, WorkspaceData>,
): ResultCardSummary[] {
  return cards.map((card) => hydrateResultCard(card, workspaceDataByResult));
}

export function hydrateStepCards(
  cards: StepCard[],
  workspaceDataByResult: Record<string, WorkspaceData>,
): StepCard[] {
  return cards.map((card) => ({
    ...card,
    results: card.results.map((result) => hydrateResultCard(result, workspaceDataByResult)),
  }));
}
