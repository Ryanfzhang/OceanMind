"use client";

import { useEffect, useMemo, useState } from "react";
import { CenterWorkspace } from "@/components/center-workspace";
import { LeftControlPanel } from "@/components/left-control-panel";
import { AnalysisFeed } from "@/components/analysis-feed";
import { ChartTray } from "@/components/chart-tray";
import {
  exportConversationReport,
  getActiveDataset,
  pingApi,
  runDirectVisualization,
  runWorkspaceQueryStream,
  type QueryStreamEnvelope,
} from "@/lib/api";
import {
  availableDatasetVariables,
  hydrateManualViewStateFromDataset,
  isDatasetVariableAvailable,
} from "@/lib/dataset-state";
import { buildQueryExtractedParams, buildWorkspaceSelectionContext } from "@/lib/geometry-tools";
import { manualViewState } from "@/lib/mock-data";
import {
  buildConversationReportPayload,
  buildReportFilename,
} from "@/lib/report-export";
import {
  formatStepProgressText,
  mergeOrAppendStepCard,
  mergeStepCardLists,
  resolveMapResult,
  toggleStepCards,
} from "@/lib/step-card-state";
import {
  EMPTY_WORKSPACE_DATA,
  hydrateResultCards,
  hydrateStepCards,
  normalizeWorkspaceData,
  normalizeWorkspaceDataByResult,
} from "@/lib/workspace-results";
import type {
  AnalysisProposal,
  AssistantMessagePayload,
  ChatMessage,
  DatasetInfo,
  ManualViewState,
  PlanStep,
  IntegratedAssessment,
  PolicyGuidance,
  QueryApiResponse,
  ResultCardSummary,
  ScientificFinding,
  SourceCard,
  StepCard,
  WorkspaceData,
  ChartTrayContent,
} from "@/lib/types";

type QuerySubmitOptions = {
  continuePending?: boolean;
  additionalContext?: Record<string, unknown>;
};

const TOOL_LABELS: Record<string, string> = {
  load_dataset: "Load Data",
  compute_spatial_field: "Build Spatial Field",
  compute_spatial_vorticity_map: "Build Vorticity Map",
  extract_regional_mean: "Extract Regional Mean Time Series",
  extract_timeseries: "Extract Time Series",
  compute_trend: "Compute Linear Trend",
  compute_climatology: "Compute Climatology",
  compute_anomaly: "Compute Anomaly",
  compute_histogram: "Compute Distribution",
  extract_vertical_profile: "Extract Vertical Profile",
  compute_hovmoller: "Build Hovmoller Diagram",
  perform_eof_analysis: "Run EOF Analysis",
  compute_density: "Compute Density",
  compute_derived_field: "Compute Derived Field",
  detect_eddies: "Detect Eddies",
};

function createMessageId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function describeTool(tool?: string | null) {
  if (!tool) {
    return "Execution Step";
  }
  return TOOL_LABELS[tool] ?? tool.replace(/_/g, " ");
}

function mapPlanSteps(response: QueryApiResponse): PlanStep[] {
  return response.plan_steps.map((step, index) => ({
    id: step.step_id ?? `step_${index + 1}`,
    tool: describeTool(step.tool),
    humanLabel: typeof step.human_label === "string" ? step.human_label : describeTool(step.tool),
    technicalLabel: typeof step.technical_label === "string" ? step.technical_label : step.tool ?? "",
    status:
      response.status === "completed"
        ? "completed"
        : response.status === "clarification_needed"
          ? "pending"
          : "active",
  }));
}

function buildPlanStepsFromPlan(plan?: Record<string, unknown> | null): PlanStep[] {
  const steps = Array.isArray(plan?.steps) ? (plan.steps as Array<Record<string, unknown>>) : [];
  return steps.map((step, index) => ({
    id: String(step.step_id ?? `step_${index + 1}`),
    tool: describeTool(typeof step.tool === "string" ? step.tool : null),
    humanLabel:
      typeof step.human_label === "string"
        ? step.human_label
        : describeTool(typeof step.tool === "string" ? step.tool : null),
    technicalLabel: typeof step.technical_label === "string" ? step.technical_label : String(step.tool ?? ""),
    status: "pending",
  }));
}

function updatePlanStepStatus(steps: PlanStep[], stepId: string, status: PlanStep["status"]): PlanStep[] {
  return steps.map((step) => (step.id === stepId ? { ...step, status } : step));
}

function buildConversationContext(messages: ChatMessage[]) {
  const recentMessages = messages.slice(-16);
  const recentQueries = recentMessages
    .filter((message) => message.role === "user")
    .slice(-4)
    .map((message) => message.text);

  const conclusions = recentMessages
    .filter((message) => message.role === "assistant" && message.payload?.state === "completed")
    .slice(-1)
    .map((message) => message.payload?.summary ?? message.text)
    .filter(Boolean);

  return {
    recent_queries: recentQueries,
    conclusions: conclusions.slice(0, 1),
    guidance:
      "Use this as lightweight conversation memory. Prefer the current user request and clarification_context when present.",
  };
}

function buildAdditionalContext(state: ManualViewState, messages: ChatMessage[]) {
  const workspaceSelection = buildWorkspaceSelectionContext(state);
  const selectedRegionBounds = workspaceSelection.selected_region?.region_bounds ?? state.regionBounds;
  const selectedRegionLabel =
    workspaceSelection.selected_region?.type === "polygon" ? "Polygon selected region" : state.regionLabel;
  const selectedPolygonPoints =
    workspaceSelection.selected_region?.type === "polygon" ? workspaceSelection.selected_region.points : [];
  const selectedTransectPoints = workspaceSelection.selected_transect?.points ?? [];
  const currentDepth =
    state.depthMode === "fixed"
      ? state.depthRange[0]
      : state.depthMode === "layer_mean"
        ? null
        : null;

  const currentVerticalSelection =
    state.depthMode === "fixed"
      ? `${state.depthRange[0]} m`
      : state.depthMode === "feature"
        ? state.feature
        : state.layerMeanLabel;

  return {
    workspace_context: {
      dataset: state.dataset,
      variable: state.variable,
      current_day: state.timeRange[0],
      current_time_range: state.timeRange,
      time_range: state.timeRange,
      time_label: state.timeLabel,
      current_region_bounds: selectedRegionBounds,
      current_region_label: selectedRegionLabel,
      region_bounds: selectedRegionBounds,
      region_label: selectedRegionLabel,
      region_selection_type: workspaceSelection.selected_region?.type ?? null,
      selected_point: workspaceSelection.selected_point
        ? {
            lat: workspaceSelection.selected_point.lat,
            lon: workspaceSelection.selected_point.lon,
          }
        : null,
      region_center: {
        lat: state.selectedPoint[0],
        lon: state.selectedPoint[1],
      },
      selection_mode: state.selectionMode,
      workspace_selection: workspaceSelection,
      transect_points: selectedTransectPoints,
      mask_polygon: selectedPolygonPoints,
      drawn_transect_points: selectedTransectPoints,
      drawn_polygon_points: selectedPolygonPoints,
      depth_mode: state.depthMode,
      current_depth: currentDepth,
      current_depth_range: state.depthRange,
      current_vertical_selection: currentVerticalSelection,
      depth_range: state.depthRange,
      vertical_feature: state.feature,
      layer_mean_label: state.layerMeanLabel,
    },
    conversation_context: buildConversationContext(messages),
    parameter_resolution_policy: {
      priority: "explicit user query overrides workspace defaults",
      use_workspace_defaults_for_missing_values_only: true,
    },
  };
}

function buildVisualizationPayload(state: ManualViewState) {
  const selection = buildWorkspaceSelectionContext(state);
  const point = selection.selected_point;
  const region = selection.selected_region?.region
    ?? selection.selected_transect?.padded_region
    ?? (point
      ? {
          lon_range: [point.lon - 1, point.lon + 1] as [number, number],
          lat_range: [Math.max(-90, point.lat - 1), Math.min(90, point.lat + 1)] as [number, number],
        }
      : {
          lon_range: [state.regionBounds.lonMin, state.regionBounds.lonMax] as [number, number],
          lat_range: [state.regionBounds.latMin, state.regionBounds.latMax] as [number, number],
        });
  return {
    dataset: state.dataset,
    variable: state.variable,
    time_range: state.timeRange,
    region: {
      lon_min: region.lon_range[0],
      lon_max: region.lon_range[1],
      lat_min: region.lat_range[0],
      lat_max: region.lat_range[1],
    },
    selection_mode: selection.active_selection,
    polygon_points: selection.selected_region?.type === "polygon" ? selection.selected_region.points : undefined,
    transect_points: selection.selected_transect?.points,
    depth_mode: state.depthMode,
    depth_range: state.depthRange,
    feature: state.feature,
    layer_mean_label: state.layerMeanLabel,
    selected_point: point ? { lat: point.lat, lon: point.lon } : undefined,
  };
}

function configuredVariables(datasetInfo?: DatasetInfo | null): string[] {
  return availableDatasetVariables(datasetInfo);
}

function configuredDepthLevels(datasetInfo?: DatasetInfo | null): number[] {
  if (!Array.isArray(datasetInfo?.depth_levels)) {
    return [];
  }
  return datasetInfo.depth_levels.filter(
    (value): value is number => typeof value === "number" && Number.isFinite(value),
  );
}

function canRunQuickVisualization(state: ManualViewState, datasetInfo?: DatasetInfo | null) {
  const selection = buildWorkspaceSelectionContext(state);
  if (state.selectionMode === "transect" && !selection.selected_transect) return false;
  if (state.selectionMode === "polygon" && selection.selected_region?.type !== "polygon") return false;
  if (configuredVariables(datasetInfo).length === 0) {
    return false;
  }
  if (!isDatasetVariableAvailable(datasetInfo, state.variable)) {
    return false;
  }
  if (state.depthMode === "feature" && !isDatasetVariableAvailable(datasetInfo, "temp")) {
    return false;
  }
  if (state.depthMode === "feature" && state.feature !== "thermocline"
      && !isDatasetVariableAvailable(datasetInfo, "salt")) {
    return false;
  }
  if (
    state.depthMode === "layer_mean" &&
    (!isDatasetVariableAvailable(datasetInfo, state.variable) ||
      !isDatasetVariableAvailable(datasetInfo, "temp") ||
      !isDatasetVariableAvailable(datasetInfo, "salt"))
  ) {
    return false;
  }
  if (state.depthMode === "fixed") {
    return configuredDepthLevels(datasetInfo).length > 0;
  }
  return true;
}

function buildAssistantPayload(response: QueryApiResponse, workspaceData: WorkspaceData) {
  const workspaceDataByResult = normalizeWorkspaceDataByResult(response.workspace_data_by_result);
  const resultCards = hydrateResultCards(response.result_cards, workspaceDataByResult);
  const stepCards = hydrateStepCards(response.step_cards, workspaceDataByResult);
  const synthesisWarnings = Array.isArray(response.synthesis?.synthesis_warnings)
    ? response.synthesis.synthesis_warnings.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    : [];
  const stateMap = {
    completed: "completed",
    clarification_needed: "clarification",
    failed: "failed",
  } as const;
  const findings = buildScientificFindings(response.synthesis);
  const failureSummary = response.error ?? "The query failed.";
  const missingFields = Array.isArray(response.missing_fields)
    ? response.missing_fields.filter((item) => typeof item === "string" && item.trim().length > 0)
    : [];
  const missingFieldsText =
    missingFields.length > 0
      ? `Missing fields: ${missingFields.join(", ")}`
      : "Missing fields: none reported by planner.";
  const clarificationSummary =
    response.analysis_proposal?.approval_prompt ??
    [response.clarification_question ?? "More information is needed.", missingFieldsText].filter(Boolean).join(" ");
  const timingText = formatTimings(response.timings);
  const failureResponse =
    response.status === "failed"
      ? buildFailureResponseCopy({
          stage:
            response.failure_kind === "execution"
              ? "execution"
              : response.failure_kind === "synthesis"
                ? "synthesis"
                : response.failure_kind === "transport"
                  ? "transport"
                  : "planning",
          detail: failureSummary,
        })
      : null;

  return {
    state: stateMap[response.status],
    preferredLanguage: "en" as const,
    summary:
      response.status === "completed"
        ? response.synthesis?.summary ?? response.plan_summary ?? "Analysis completed."
        : response.status === "clarification_needed"
          ? clarificationSummary
          : failureResponse?.summary ?? failureSummary,
    note:
      response.status === "completed"
        ? [response.plan_summary ?? response.router_reason ?? "", timingText].filter(Boolean).join(" ")
        : response.status === "clarification_needed"
          ? [missingFieldsText, timingText].filter(Boolean).join(" ")
          : failureResponse?.note ?? timingText,
    routingMode: response.routing_mode ?? undefined,
    routerConfidence: typeof response.router_confidence === "number" ? response.router_confidence : undefined,
    routerReason: response.router_reason ?? undefined,
    datasetInfo: response.dataset_info,
    skillsUsed: response.skills_used,
    planSummary: response.plan_summary ?? undefined,
    planSteps: mapPlanSteps(response),
    resultCards,
    stepCards,
    findings,
    policyGuidance: response.synthesis?.policy_guidance,
    integratedAssessment: response.synthesis?.integrated_assessment,
    synthesisWarnings,
    analysisProposal: response.analysis_proposal ?? undefined,
    sourceCards: response.source_cards,
    workspaceData,
    workspaceDataByResult,
    attachments: response.attachments,
    activeResultId: response.active_result_id ?? undefined,
    failureKind: response.failure_kind ?? undefined,
    recoverable: response.recoverable ?? undefined,
    timings: response.timings,
  };
}

function formatTimings(timings?: Record<string, number>) {
  if (!timings) {
    return "";
  }
  const entries = Object.entries(timings).filter(([, value]) => typeof value === "number" && Number.isFinite(value));
  if (entries.length === 0) {
    return "";
  }
  return `Timings: ${entries.map(([name, value]) => `${name} ${value.toFixed(2)}s`).join(" · ")}`;
}

function mergeTiming(current: AssistantMessagePayload, payload: Record<string, unknown>): AssistantMessagePayload {
  const name = typeof payload.name === "string" ? payload.name : "";
  const elapsed = typeof payload.elapsed_s === "number" ? payload.elapsed_s : undefined;
  if (!name || elapsed === undefined || !Number.isFinite(elapsed)) {
    return current;
  }
  const timings = { ...(current.timings ?? {}), [name]: elapsed };
  return {
    ...current,
    timings,
    note: formatTimings(timings) || current.note,
  };
}

function buildClientFailureMessage(message: string) {
  const lowered = message.toLowerCase();
  if (
    lowered.includes("valid json") ||
    lowered.includes("jsondecodeerror") ||
    lowered.includes("llm response") ||
    lowered.includes("traceback") ||
    lowered.includes("error_type=")
  ) {
    return "The system could not reliably format the model output for display. Please try again or make the request more specific.";
  }
  if (lowered.includes("network") || lowered.includes("fetch") || lowered.includes("terminated") || lowered.includes("econnreset")) {
    return "The live connection did not return a complete result. Please try again.";
  }
  return message.trim() || "The request could not be completed. Please try again.";
}

type FailureResponseStage = "planning" | "execution" | "synthesis" | "transport" | "visualization";

function normalizeFailureDetail(detail?: string | null) {
  const cleaned = (detail ?? "").trim();
  if (cleaned === "deadline_exceeded") return "The analysis time limit was reached before an answer was ready.";
  if (cleaned === "max_rounds_exceeded") return "The model-decision limit was reached before an answer was ready.";
  return buildClientFailureMessage(cleaned || "The backend did not provide a specific reason.");
}

function buildFailureResponseCopy({
  stage,
  detail,
}: {
  stage: FailureResponseStage;
  detail?: string | null;
}) {
  const summaryByStage: Record<FailureResponseStage, string> = {
    planning: "I could not prepare the analysis plan.",
    execution: "I could not complete the analysis.",
    synthesis: "I could not prepare the final answer.",
    transport: "I could not complete the request.",
    visualization: "I could not create the visualization.",
  };
  return {
    summary: `${summaryByStage[stage]} Reason: ${normalizeFailureDetail(detail)}`,
    note: "",
  };
}


function buildScientificFindings(
  synthesis: QueryApiResponse["synthesis"] | null | undefined,
): ScientificFinding[] {
  const items = synthesis?.scientific_findings;
  if (!Array.isArray(items)) {
    return [];
  }
  return items
    .filter((item): item is NonNullable<typeof item> => Boolean(item && typeof item.finding === "string"))
    .map((item) => ({
      title: item.finding,
      evidence: Array.isArray(item.evidence)
        ? item.evidence.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0)
        : [],
    }));
}

function updateAssistantMessage(
  messages: ChatMessage[],
  messageId: string,
  updater: (payload: NonNullable<ChatMessage["payload"]>) => NonNullable<ChatMessage["payload"]>,
) {
  return messages.map((message) => {
    if (message.id !== messageId || !message.payload) {
      return message;
    }
    const payload = updater(message.payload);
    return {
      ...message,
      text: payload.summary,
      payload,
    };
  });
}

export function OceanWorkspaceApp() {
  const [conversationId, setConversationId] = useState(() => createMessageId("conversation"));
  const [hasPendingClarification, setHasPendingClarification] = useState(false);
  const [manualState, setManualState] = useState(manualViewState);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [queryText, setQueryText] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isQuerying, setIsQuerying] = useState(false);
  const [isVisualizing, setIsVisualizing] = useState(false);
  const [isExportingReport, setIsExportingReport] = useState(false);
  const [chartTrayContent, setChartTrayContent] = useState<ChartTrayContent | null>(null);
  const [activeMapResult, setActiveMapResult] = useState<ResultCardSummary | null>(null);
  const [activeMapWorkspaceData, setActiveMapWorkspaceData] = useState<WorkspaceData>(EMPTY_WORKSPACE_DATA);
  const quickVisualizeEnabled = canRunQuickVisualization(manualState, datasetInfo);

  useEffect(() => {
    pingApi().catch(() => undefined);
    getActiveDataset()
      .then((info) => {
        setDatasetInfo(info);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!datasetInfo) {
      return;
    }
    setManualState((current) => hydrateManualViewStateFromDataset(current, datasetInfo));
  }, [datasetInfo?.id, datasetInfo?.name, datasetInfo?.data_path]);

  const handleExpandDetail = (card: ResultCardSummary, data: WorkspaceData) => {
    const nextWorkspace = normalizeWorkspaceData(data);
    setChartTrayContent({
      resultCard: {
        ...card,
        workspaceData: nextWorkspace,
      },
      workspaceData: nextWorkspace,
    });
  };

  const focusMapResult = (card: ResultCardSummary, data: WorkspaceData) => {
    const nextWorkspace = normalizeWorkspaceData(data);
    if (!nextWorkspace.mapField && nextWorkspace.eventOverlays.length === 0) {
      return;
    }
    setActiveMapResult(card);
    setActiveMapWorkspaceData(nextWorkspace);
  };

  const handlePromoteMapField = (card: ResultCardSummary, data: WorkspaceData, field: WorkspaceData["mapField"]) => {
    if (!field) {
      return;
    }
    const nextWorkspace = normalizeWorkspaceData({
      ...data,
      mapField: field,
    });
    setActiveMapResult({
      ...card,
      workspaceData: nextWorkspace,
    });
    setActiveMapWorkspaceData(nextWorkspace);
  };

  const handleResultAction = (card: ResultCardSummary, data: WorkspaceData, actionId: string) => {
    if (actionId === "focus_map") {
      focusMapResult(card, data);
      return;
    }
    if (actionId === "open_detail" || actionId === "open_modal") {
      handleExpandDetail(card, data);
    }
  };

  const handleCloseChartTray = () => {
    setChartTrayContent(null);
  };

  const handleQuickVisualize = async () => {
    if (!quickVisualizeEnabled) {
      return;
    }
    setIsVisualizing(true);
    try {
      const response = await runDirectVisualization(buildVisualizationPayload(manualState));
      if (response.status === "completed") {
        const nextWorkspace = normalizeWorkspaceData(response.workspace_data);
        setActiveMapWorkspaceData(nextWorkspace);
        setActiveMapResult(response.result_cards[0] ?? null);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown visualization error";
      const responseCopy = buildFailureResponseCopy({
        stage: "visualization",
        detail: message,
      });
      setMessages((previous) => [
        ...previous,
        {
          id: createMessageId("assistant"),
          role: "assistant",
          text: responseCopy.summary,
          payload: {
            state: "failed",
            summary: responseCopy.summary,
            note: responseCopy.note,
            planSteps: [],
            resultCards: [],
            stepCards: [],
            findings: [],
            sourceCards: [],
            workspaceData: EMPTY_WORKSPACE_DATA,
            activeResultId: undefined,
            failureKind: "execution",
            recoverable: true,
          },
        },
      ]);
    } finally {
      setIsVisualizing(false);
    }
  };

  const handleExportReport = async () => {
    if (messages.length === 0) {
      return;
    }

    setIsExportingReport(true);
    try {
      const payload = buildConversationReportPayload({
        conversationId,
        datasetInfo,
        exportedAt: new Date().toISOString(),
        messages,
      });
      const { blob, filename } = await exportConversationReport(payload);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename ?? buildReportFilename(payload.exported_at);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } finally {
      setIsExportingReport(false);
    }
  };

  const handleToggleStepCard = (messageId: string, stepId: string) => {
    const targetMessage = messages.find((message) => message.id === messageId);
    const currentStepCards = targetMessage?.payload?.stepCards ?? [];
    const { nextStepCards, nextMapCard, clearMap } = toggleStepCards(currentStepCards, stepId);

    setMessages((previous) =>
      previous.map((message) => {
        if (message.id !== messageId || !message.payload?.stepCards) {
          return message;
        }

        return {
          ...message,
          payload: {
            ...message.payload,
            stepCards: nextStepCards,
          },
        };
      })
    );

    if (nextMapCard !== null && nextMapCard.workspaceData) {
      setActiveMapResult(nextMapCard);
      setActiveMapWorkspaceData(normalizeWorkspaceData(nextMapCard.workspaceData));
    } else if (clearMap) {
      setActiveMapResult(null);
      setActiveMapWorkspaceData(EMPTY_WORKSPACE_DATA);
    }
  };

  const handleQuery = async (queryOverride?: string, options: QuerySubmitOptions = {}) => {
    const trimmedQuery = (queryOverride ?? queryText).trim();
    if (!trimmedQuery) {
      return;
    }

    if (queryOverride) {
      setQueryText(trimmedQuery);
    }

    setQueryText("");
    setIsQuerying(true);
    const pendingAssistantId = createMessageId("assistant");
    setMessages((previous) => [
      ...previous,
      { id: createMessageId("user"), role: "user", text: trimmedQuery },
      {
        id: pendingAssistantId,
        role: "assistant",
        text: "Thinking",
        payload: {
          state: "planning",
          preferredLanguage: "en" as const,
          summary: "Thinking",
          note: "Routing and planning are starting.",
          routingMode: undefined,
          datasetInfo: datasetInfo ?? undefined,
          skillsUsed: [],
          planSummary: undefined,
          planSteps: [],
          resultCards: [],
          stepCards: [],
          findings: [],
          sourceCards: [],
          workspaceData: EMPTY_WORKSPACE_DATA,
          activeResultId: undefined,
        },
      },
    ]);

    let receivedFinalEvent = false;
    let streamErrorDetail: string | null = null;
    const finishTransportFailure = (detail: string) => {
      const responseCopy = buildFailureResponseCopy({ stage: "transport", detail });
      setMessages((previous) =>
        updateAssistantMessage(previous, pendingAssistantId, (current) => ({
          ...current,
          state: "failed",
          summary: responseCopy.summary,
          note: responseCopy.note,
          failureKind: "transport",
          recoverable: true,
        }))
      );
    };

    try {
      await runWorkspaceQueryStream(
        {
          query: trimmedQuery,
          conversation_id: conversationId,
          continue_pending: options.continuePending ?? hasPendingClarification,
          extracted_params: buildQueryExtractedParams(manualState, trimmedQuery),
          additional_context: {
            ...buildAdditionalContext(manualState, messages),
            ...(options.additionalContext ?? {}),
          },
          synthesize: true,
          trust_env: false,
        },
        (streamEvent: QueryStreamEnvelope) => {
          if (streamEvent.event === "execution_event") {
            const payload = streamEvent.payload;
            const eventType = typeof payload.type === "string" ? payload.type : "";

            if (eventType === "heartbeat") {
              return;
            }

            if (eventType === "timing") {
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => mergeTiming(current, payload))
              );
              return;
            }

            if (eventType === "planning_phase") {
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  state: current.state === "completed" || current.state === "failed" ? current.state : "planning",
                  summary: "Thinking",
                  note: formatTimings(current.timings),
                }))
              );
              return;
            }

            if (eventType === "planning_started") {
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  state: "planning",
                  summary: "Thinking",
                  note: "Deciding whether this needs the active dataset.",
                }))
              );
              return;
            }

            if (eventType === "routing_decision") {
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  summary: current.state === "planning" ? "Thinking" : current.summary,
                  note: typeof payload.reason === "string" ? payload.reason : current.note,
                  routingMode:
                    payload.routing_mode === "dataset_analysis" ||
                    payload.routing_mode === "general_answer"
                      ? payload.routing_mode
                      : current.routingMode,
                  routerConfidence: typeof payload.confidence === "number" ? payload.confidence : current.routerConfidence,
                  routerReason: typeof payload.reason === "string" ? payload.reason : current.routerReason,
                }))
              );
              return;
            }

            if (eventType === "analysis_proposal_ready") {
              const proposal =
                payload.analysis_proposal && typeof payload.analysis_proposal === "object"
                  ? (payload.analysis_proposal as AnalysisProposal)
                  : undefined;
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  state: "clarification",
                  summary: proposal?.approval_prompt ?? "Review the suggested analysis plan before I run it.",
                  note: "A broad request was converted into a concrete analysis plan. Nothing has been executed yet.",
                  analysisProposal: proposal,
                }))
              );
              return;
            }

            if (eventType === "plan_ready") {
              const planSteps = buildPlanStepsFromPlan((payload.plan as Record<string, unknown> | undefined) ?? null);
              const initialStepCards = Array.isArray(payload.initial_step_cards)
                ? (payload.initial_step_cards as StepCard[])
                : [];
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  state: "running",
                  summary:
                    typeof payload.plan_summary === "string" && payload.plan_summary
                      ? payload.plan_summary
                      : "Plan ready. Execution is starting.",
                  note: "Execution is starting. Each step will appear below.",
                  skillsUsed: Array.isArray(payload.skills_used)
                    ? payload.skills_used.filter((item): item is string => typeof item === "string")
                    : current.skillsUsed,
                  planSteps,
                  stepCards: initialStepCards.length > 0 ? initialStepCards : current.stepCards,
                }))
              );
              return;
            }

            if (eventType === "plan_replanned") {
              const planSteps = buildPlanStepsFromPlan((payload.plan as Record<string, unknown> | undefined) ?? null);
              const initialStepCards = Array.isArray(payload.initial_step_cards)
                ? (payload.initial_step_cards as StepCard[])
                : [];
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  state: "running",
                  summary: "Continuing the analysis...",
                  note: "The workflow was updated and execution is continuing.",
                  skillsUsed: Array.isArray(payload.skills_used)
                    ? payload.skills_used.filter((item): item is string => typeof item === "string")
                    : current.skillsUsed,
                  planSteps,
                  stepCards: initialStepCards.length > 0 ? initialStepCards : current.stepCards,
                }))
              );
              return;
            }

            if (eventType === "planning_failed") {
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  state: "running",
                  summary: "Continuing the analysis...",
                  note: "",
                }))
              );
              return;
            }

            if (eventType === "step_started") {
              const stepId = String(payload.step_id ?? "");
              const stepCard = payload.step_card as StepCard | undefined;
              const stepLabel =
                typeof stepCard?.human_label === "string"
                  ? stepCard.human_label
                  : describeTool(typeof payload.tool === "string" ? payload.tool : null);
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  state: "running",
                  summary: `Running: ${stepLabel}`,
                  note: `${stepLabel} is now running.`,
                  planSteps: updatePlanStepStatus(current.planSteps, stepId, "active"),
                  stepCards: stepCard ? mergeOrAppendStepCard(current.stepCards ?? [], stepCard) : current.stepCards,
                }))
              );
              return;
            }

            if (eventType === "step_progress") {
              const stepId = String(payload.step_id ?? "");
              const stepCard = payload.step_card as StepCard | undefined;
              const progress = stepCard?.progress;
              const stepLabel =
                typeof stepCard?.human_label === "string"
                  ? stepCard.human_label
                  : describeTool(typeof payload.tool === "string" ? payload.tool : null);
              const percentLabel =
                typeof progress?.percent === "number" && Number.isFinite(progress.percent)
                  ? `${Math.round(progress.percent * 100)}%`
                  : "";
              const progressNote = [formatStepProgressText(progress), percentLabel].filter(Boolean).join(" · ");
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  state: "running",
                  summary: `Running: ${stepLabel}`,
                  note: progressNote || `${stepLabel} is running.`,
                  planSteps: updatePlanStepStatus(current.planSteps, stepId, "active"),
                  stepCards: stepCard ? mergeOrAppendStepCard(current.stepCards ?? [], stepCard) : current.stepCards,
                }))
              );
              return;
            }

            if (eventType === "step_reflection_started") {
              const stepId = String(payload.step_id ?? "");
              const stepCard = payload.step_card as StepCard | undefined;
              const stepLabel =
                typeof stepCard?.human_label === "string"
                  ? stepCard.human_label
                  : describeTool(typeof payload.tool === "string" ? payload.tool : null);
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  state: "running",
                  summary: `Updating workflow: ${stepLabel}`,
                  note: "Updating the workflow before continuing execution.",
                  planSteps: updatePlanStepStatus(current.planSteps, stepId, "active"),
                  stepCards: stepCard ? mergeOrAppendStepCard(current.stepCards ?? [], stepCard) : current.stepCards,
                }))
              );
              return;
            }

            if (eventType === "step_completed") {
              const stepId = String(payload.step_id ?? "");
              const stepCard = payload.step_card as StepCard | undefined;
              const stepLabel =
                typeof stepCard?.human_label === "string"
                  ? stepCard.human_label
                  : describeTool(typeof payload.tool === "string" ? payload.tool : null);
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  state: "running",
                  summary: `Completed: ${stepLabel}`,
                  note: `${stepLabel} completed.`,
                  planSteps: updatePlanStepStatus(current.planSteps, stepId, "completed"),
                  stepCards: stepCard ? mergeOrAppendStepCard(current.stepCards ?? [], stepCard) : current.stepCards,
                }))
              );
              const nextMapCard = stepCard ? resolveMapResult(stepCard) : null;
              if (nextMapCard?.workspaceData) {
                focusMapResult(nextMapCard, nextMapCard.workspaceData as WorkspaceData);
              }
              return;
            }

            if (eventType === "step_result_attached") {
              const stepCard = payload.step_card as StepCard | undefined;
              if (!stepCard) {
                return;
              }
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => {
                  const existing = current.stepCards?.find((item) => item.step_id === stepCard.step_id);
                  const known = new Set(existing?.results.map((item) => item.id) ?? []);
                  const results = [...(existing?.results ?? []),
                    ...stepCard.results.filter((item) => !known.has(item.id))];
                  return { ...current, stepCards: mergeOrAppendStepCard(
                    current.stepCards ?? [], { ...stepCard, results },
                  ) };
                })
              );
              const nextMapCard = resolveMapResult(stepCard);
              if (nextMapCard?.workspaceData) {
                focusMapResult(nextMapCard, nextMapCard.workspaceData as WorkspaceData);
              }
              return;
            }

            if (eventType === "attempt_result_attached") {
              const card = payload.result_card as ResultCardSummary | undefined;
              if (!card) return;
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  resultCards: current.resultCards.some((item) => item.id === card.id)
                    ? current.resultCards
                    : [...current.resultCards, card],
                }))
              );
              if (card.surface === "map" && card.workspaceData) {
                focusMapResult(card, card.workspaceData as WorkspaceData);
              }
              return;
            }

            if (eventType === "step_failed") {
              const stepId = String(payload.step_id ?? "");
              const stepCard = payload.step_card as StepCard | undefined;
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                    ...current,
                    state: "running",
                    summary: "Continuing the analysis...",
                    note: "",
                    planSteps: updatePlanStepStatus(current.planSteps, stepId, "active"),
                    stepCards: stepCard
                      ? mergeOrAppendStepCard(current.stepCards ?? [], {
                          ...stepCard,
                          status: "running",
                          error: undefined,
                          is_expanded: false,
                          progress: { phase: "reflection", message: "Continuing analysis" },
                        })
                      : current.stepCards,
                }))
              );
              return;
            }

            if (eventType === "clarification_needed") {
              const question =
                typeof payload.question === "string" ? payload.question : "More information is needed to continue.";
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  state: "clarification",
                  summary: question,
                  note: "The planner needs additional context before continuing.",
                }))
              );
            }

            if (eventType === "results_ready") {
              const workspaceData = normalizeWorkspaceData(payload.workspace_data as Partial<WorkspaceData> | undefined);
              const workspaceDataByResult = normalizeWorkspaceDataByResult(
                payload.workspace_data_by_result as Record<string, Partial<WorkspaceData>> | undefined,
              );
              const resultCards = hydrateResultCards(
                Array.isArray(payload.result_cards) ? (payload.result_cards as ResultCardSummary[]) : [],
                workspaceDataByResult,
              );

              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  state: "running",
                  summary:
                    current.skillsUsed?.includes("ocean_environment_health_assessment")
                      ? "Results are ready. Generating environmental assessment..."
                      : "Results are ready. Generating summary...",
                  note: current.planSummary ?? "Waiting for LLM synthesis.",
                  stepCards: hydrateStepCards(current.stepCards ?? [], workspaceDataByResult),
                  resultCards,
                  workspaceData,
                  workspaceDataByResult,
                  activeResultId:
                    typeof payload.active_result_id === "string" ? payload.active_result_id : current.activeResultId,
                }))
              );

              const activeResultId = typeof payload.active_result_id === "string" ? payload.active_result_id : null;
              const activeResultCard = activeResultId
                ? resultCards.find((card) => card.id === activeResultId) ?? null
                : null;
              if (activeResultCard?.workspaceData) {
                const activeWorkspace = activeResultCard.workspaceData as WorkspaceData;
                if (activeWorkspace.mapField || activeWorkspace.eventOverlays.length > 0) {
                  focusMapResult(activeResultCard, activeWorkspace);
                }
              }
              return;
            }

            if (eventType === "synthesis_started" || eventType === "synthesis_retry_started") {
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  state: "running",
                  summary:
                    current.skillsUsed?.includes("ocean_environment_health_assessment")
                      ? "Generating environmental assessment..."
                      : "Generating result synthesis...",
                  note: "Preparing the final answer.",
                }))
              );
              return;
            }

            if (eventType === "synthesis_attempt_failed" || eventType === "synthesis_failed") {
              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => ({
                  ...current,
                  state: "running",
                  summary: "Preparing the final answer...",
                  note: "",
                }))
              );
              return;
            }

            if (eventType === "synthesis_ready") {
              const synthesis = payload.synthesis as Record<string, unknown> | null | undefined;
              const findings = buildScientificFindings(synthesis as QueryApiResponse["synthesis"]);
              const synthesisWarnings = Array.isArray(synthesis?.synthesis_warnings)
                ? synthesis.synthesis_warnings.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
                : [];

              setMessages((previous) =>
                updateAssistantMessage(previous, pendingAssistantId, (current) => {
                  const summary =
                    (typeof synthesis?.summary === "string" ? synthesis.summary : null)
                    ?? current.planSummary
                    ?? "Analysis completed.";

                  // Merge interpretations from synthesis into existing step cards
                  let updatedStepCards = current.stepCards ?? [];
                  if (Array.isArray(payload.step_cards) && payload.step_cards.length > 0) {
                    const newCardsById = new Map(
                      (payload.step_cards as StepCard[]).map((card) => [card.step_id, card]),
                    );
                    updatedStepCards = updatedStepCards.map((card) => {
                      const updated = newCardsById.get(card.step_id);
                      return updated ? { ...card, interpretation: updated.interpretation } : card;
                    });
                  }

                  return {
                    ...current,
                    state: "completed",
                    summary,
                    note: "Summary ready.",
                    findings,
                    policyGuidance: synthesis?.policy_guidance as PolicyGuidance | undefined,
                    integratedAssessment: synthesis?.integrated_assessment as IntegratedAssessment | undefined,
                    synthesisWarnings,
                    stepCards: updatedStepCards,
                  };
                })
              );
              return;
            }

            return;
          }

          if (streamEvent.event === "error") {
            streamErrorDetail =
              typeof streamEvent.payload.detail === "string"
                ? streamEvent.payload.detail
                : "The live query could not return a complete result. Please try again.";
            return;
          }

          if (streamEvent.event === "final") {
            receivedFinalEvent = true;
            const response = streamEvent.payload as unknown as QueryApiResponse;
            if (response.conversation_id) {
              setConversationId(response.conversation_id);
            }

            if (response.dataset_info) {
              setDatasetInfo(response.dataset_info);
            }

            if (response.status === "completed") {
              setHasPendingClarification(false);
            } else {
              setHasPendingClarification(response.status === "clarification_needed");
            }

            const workspace = normalizeWorkspaceData(response.workspace_data);
            const assistantPayload = buildAssistantPayload(response, workspace);
            const activeResultCard =
              response.active_result_id
                ? assistantPayload.resultCards.find((card) => card.id === response.active_result_id) ?? null
                : null;
            if (activeResultCard?.workspaceData) {
              const activeWorkspace = activeResultCard.workspaceData as WorkspaceData;
              if (activeWorkspace.mapField || activeWorkspace.eventOverlays.length > 0) {
                focusMapResult(activeResultCard, activeWorkspace);
              }
            }
            setMessages((previous) =>
              previous.map((msg) => {
                if (msg.id !== pendingAssistantId) return msg;
                // Preserve streaming stepCards; use final data only as fallback
                const existingStepCards = msg.payload?.stepCards ?? [];
                const finalStepCards = assistantPayload.stepCards;
                const mergedStepCards =
                  finalStepCards && finalStepCards.length > 0
                    ? mergeStepCardLists(existingStepCards, finalStepCards)
                    : existingStepCards;
                return {
                  ...msg,
                  text: assistantPayload.summary,
                  payload: { ...assistantPayload, stepCards: mergedStepCards },
                };
              })
            );
          }
        },
      );
      if (!receivedFinalEvent) {
        finishTransportFailure(streamErrorDetail ?? "The response stream ended before a final result was received.");
      }
    } catch (error) {
      if (!receivedFinalEvent) {
        finishTransportFailure(error instanceof Error ? error.message : "Unknown API error");
      }
    } finally {
      setIsQuerying(false);
    }
  };

  const mergedWorkspaceData = useMemo(() => activeMapWorkspaceData, [activeMapWorkspaceData]);
  return (
    <div className="workspace-viewport">
      <div className="workspace-stage">
        <header className="app-banner">
          <h1>OceanMind</h1>
        </header>
        <main className="app-shell">
          <LeftControlPanel
            canQuickVisualize={quickVisualizeEnabled}
            datasetInfo={datasetInfo}
            isBusy={isVisualizing || isQuerying}
            onChange={setManualState}
            onQuickVisualize={handleQuickVisualize}
            state={manualState}
          />
          <div className="map-column">
            <CenterWorkspace
              activeResult={activeMapResult}
              activeEventId={null}
              manualState={manualState}
              onManualStateChange={setManualState}
              workspaceData={mergedWorkspaceData}
            />
            <ChartTray content={chartTrayContent} onClose={handleCloseChartTray} onPromoteMapField={handlePromoteMapField} />
          </div>
          <div className="feed-column">
            <AnalysisFeed
              conversationId={conversationId}
              canExportReport={conversationId.startsWith("conv_") && messages.some(
                (message) => message.payload?.attachments?.some((item) => item.kind === "code")
              )}
              isExportingReport={isExportingReport}
              isBusy={isQuerying}
              isLocked={isQuerying || isVisualizing || isExportingReport}
              messages={messages}
              onOpenDetail={handleExpandDetail}
              onExportReport={handleExportReport}
              onPromoteMapField={handlePromoteMapField}
              onResultAction={handleResultAction}
              onQueryChange={setQueryText}
              onSubmitQuery={handleQuery}
              onToggleStepCard={handleToggleStepCard}
              queryText={queryText}
            />
          </div>
        </main>
      </div>
    </div>
  );
}
