export type VerticalFeature = "mixed_layer" | "thermocline" | "pycnocline";
export type DepthMode = "fixed" | "feature" | "layer_mean";
export type SelectionMode = "box" | "point" | "transect" | "polygon" | "none";
export type RendererType =
  | "reference"
  | "summary"
  | "timeseries"
  | "ts_diagram"
  | "profile"
  | "hovmoller"
  | "section"
  | "histogram"
  | "eof"
  | "composite"
  | "event";

export type RegionBounds = {
  lonMin: number;
  lonMax: number;
  latMin: number;
  latMax: number;
};

export type GeoPoint = {
  lat: number;
  lon: number;
};

export type ManualViewState = {
  dataset: string;
  variable: string;
  timeRange: [string, string];
  timeLabel: string;
  regionBounds: RegionBounds;
  regionLabel: string;
  selectedPoint: [number, number];
  selectionMode: SelectionMode;
  transectPoints: GeoPoint[];
  polygonPoints: GeoPoint[];
  depthMode: DepthMode;
  availableDepths: number[];
  depthRange: [number, number];
  feature: VerticalFeature;
  layerMeanLabel: string;
};

export type ResultMetric = {
  label: string;
  value: string;
};

export type ResultDetailSection = {
  title: string;
  items: string[];
};

export type ResultCardSummary = {
  id: string;
  title: string;
  type: string;
  headline: string;
  description: string;
  renderer: RendererType;
  metrics: ResultMetric[];
  surface?: "inline" | "map" | "drawer" | "modal" | "summary";
  ownerStepId?: string;
  workspaceData?: Partial<WorkspaceData>;
  actions?: StepAction[];
  interpretation?: string;
  detailSections?: ResultDetailSection[];
  finiteCount?: number;
  hasFiniteValues?: boolean;
};

export type PlanStep = {
  id: string;
  tool: string;
  status: "completed" | "active" | "pending" | "failed";
  humanLabel?: string;
  technicalLabel?: string;
};

export type StepAction = {
  id: string;
  label: string;
};

export type StepProgress = {
  phase: string;
  message?: string;
  percent?: number;
  completed_units?: number;
  total_units?: number;
  unit_label?: string;
  current_unit?: string;
  storage_backend?: string;
  compute_backend?: string;
  chunks?: unknown;
};

export type StepCard = {
  step_id: string;
  human_label: string;
  technical_label: string;
  status: "pending" | "running" | "completed" | "failed";
  results_hidden_by_default: boolean;
  results: ResultCardSummary[];
  interpretation?: string;
  actions: StepAction[];
  is_map_bound: boolean;
  is_expanded: boolean;
  progress?: StepProgress;
  error?: string;
};

export type SourceCard = {
  title: string;
  source: string;
  url: string;
  short_snippet: string;
  why_it_matters: string;
  provider?: string | null;
  search_query?: string | null;
  rank?: number | null;
};

export type DatasetInfo = {
  id: string;
  name: string;
  data_path?: string;
  data_path_redacted?: boolean;
  data_path_policy?: string;
  backend?: "netcdf" | "zarr" | string;
  chunks?: Record<string, number | "auto"> | null;
  zarr_store_pattern?: string | null;
  data_stores?: Record<string, { store?: string | null; exists?: boolean; error?: string }>;
  description: string;
  variables?: string[];
  spatial_extent?: Record<string, unknown> | null;
  temporal_extent?: Record<string, unknown> | null;
  depth_levels?: number[] | null;
  depth_range?: number[] | null;
  resolution?: string | Record<string, unknown> | null;
};

export type ScientificFinding = {
  title: string;
  evidence: string[];
};

export type AnalysisSeries = {
  label: string;
  value: number;
};

export type ProfilePoint = {
  depth: number;
  value: number;
};

export type HistogramBin = {
  label: string;
  value: number;
};

export type TsDiagramPoint = {
  temperature: number;
  salinity: number;
  colorValue?: number;
  pointClass?: string;
};

export type TsDiagramWatermassBin = {
  id: string;
  name: string;
  short_name?: string;
  color: string;
  sigma0_range?: number[];
  temp_range?: number[];
  salt_range?: number[];
};

export type HovmollerRow = {
  depthLabel: string;
  depthValue?: number;
  values: Array<number | null>;
};

export type HovmollerDisplayInfo = {
  aggregation: "none" | "daily_climatology" | string;
  aggregationLabel: string;
  originalColumns: number;
  displayColumns: number;
  variable?: string;
  units?: string;
  depthIntegratedUnits?: string;
  depthAxisSource?: string;
  depthLevels?: number[];
  smoothingWindowDays?: number;
  smoothingMinPeriods?: number;
  climatologyInput?: string;
  leapDayPolicy?: string;
};

export type TimeseriesDisplayInfo = {
  aggregation: "none" | "daily_climatology" | string;
  aggregationLabel: string;
  originalPoints: number;
  displayPoints: number;
  variable?: string;
  units?: string;
  finiteCount?: number;
  hasFiniteValues?: boolean;
};

export type SectionRow = {
  label: string;
  coordValue?: number;
  values: number[];
};

export type EofPoint = {
  day: string;
  value: number;
};

export type EofModePreview = {
  id: string;
  title: string;
  varianceLabel: string;
  mapField: MapFieldData;
  pcSeries: AnalysisSeries[];
};

export type GeoBounds = {
  lonMin: number;
  lonMax: number;
  latMin: number;
  latMax: number;
};

export type SubregionGridCell = {
  subregionId: string;
  label: string;
  shortLabel?: string;
  bounds?: GeoBounds | null;
  dominantMechanism?: string | null;
  dominantScore?: number | null;
  runnerUpMechanism?: string | null;
  runnerUpScore?: number | null;
  claimStrength?: string | null;
  category?: string | null;
  categoryLabel?: string | null;
  categoryShortLabel?: string | null;
  color?: string | null;
  value?: number | null;
  valueLabel?: string | null;
  details?: string[];
  status: string;
};

export type SubregionGridData = {
  gridShape: [number, number];
  cells: SubregionGridCell[];
  bounds?: GeoBounds;
  validCount?: number;
  skippedCount?: number;
  dominantCounts?: Record<string, number>;
};

export type DiscreteLegendItem = {
  value: number;
  category?: string;
  label: string;
  short_label?: string;
  color: string;
};

export type MapColorScale = {
  min: number;
  max: number;
  rawMin?: number;
  rawMax?: number;
  colormap?: string;
  units?: string;
  label?: string;
  symmetric?: boolean;
  scaleStrategy?: string;
  renderMode?: "filled" | "contours" | string;
  lineColor?: string;
};

export type TransportFilledRegionData = {
  id?: string;
  region?: string;
  label?: string;
  scaleStrategy?: string;
  mask: boolean[][];
};

export type TransportRenderingData = {
  mode?: string;
  filledRegion?: string;
  contourRegion?: string;
  filledMask?: boolean[][];
  contourMask?: boolean[][];
  contourLevels?: number[];
  filledRegions?: TransportFilledRegionData[];
  filledColormap?: string;
  contourColor?: string;
  zeroContourColor?: string;
};

export type MapFieldData = {
  lon: number[];
  lat: number[];
  values: number[][];
  label: string;
  variable: string;
  units?: string;
  statistics?: Record<string, number>;
  depthLabel?: string;
  timeLabel?: string;
  contourImage?: string;
  colorScale?: MapColorScale;
  regionalColorScales?: MapColorScale[];
  transportRendering?: TransportRenderingData;
  tileMapKind?: "event_hotspot" | "dominant_watermass" | string;
  bounds?: [[number, number], [number, number]];
  subregionGrid?: SubregionGridData;
  discreteLegend?: DiscreteLegendItem[];
};

export type CompositeFieldPreview = {
  id: string;
  title: string;
  mapField: MapFieldData;
};

export type EventOverlayBounds = GeoBounds;

export type EventOverlayPoint = {
  lat: number;
  lon: number;
};

export type EventOverlay = {
  id: string;
  eventType: string;
  title: string;
  center: EventOverlayPoint;
  bounds?: EventOverlayBounds;
  shape?: "rectangle" | "circle" | "polyline" | "point";
  path?: EventOverlayPoint[];
  radiusKm?: number;
  symbol?: "diamond" | "triangle" | "square";
  details: string[];
  severity?: string;
  timestamp?: string;
  endTimestamp?: string;
  occurrenceCount?: number;
};

export type WorkspaceData = {
  referenceSeries: AnalysisSeries[];
  resultSeries: AnalysisSeries[];
  anomalySeries: AnalysisSeries[];
  timeseriesDisplayInfo?: TimeseriesDisplayInfo | null;
  tsDiagramPoints: TsDiagramPoint[];
  tsDiagramTemperatureLabel?: string;
  tsDiagramSalinityLabel?: string;
  tsDiagramColorLabel?: string | null;
  tsDiagramColorRange?: number[] | null;
  tsDiagramPointClasses: string[];
  tsDiagramClassColorMap: Record<string, string>;
  tsDiagramWatermassBins: TsDiagramWatermassBin[];
  seriesLabels?: {
    reference?: string;
    result?: string;
    compare?: string;
  };
  profileSeries: ProfilePoint[];
  profileMarkers: { label: string; depth: number }[];
  hovmollerRows: HovmollerRow[];
  hovmollerTimeLabels?: string[];
  hovmollerDisplayInfo?: HovmollerDisplayInfo | null;
  hovmollerDepthIntegratedSeries: AnalysisSeries[];
  sectionRows: SectionRow[];
  sectionDistanceKm?: number[];
  sectionAxisTitle?: string;
  sectionSliceLabel?: string;
  overlaySeries: { day: string; depth: number }[];
  histogramBins: HistogramBin[];
  eofVariance: ResultMetric[];
  eofPcSeries: EofPoint[];
  eofModes: EofModePreview[];
  compositeFields: CompositeFieldPreview[];
  mapField?: MapFieldData | null;
  eventOverlays: EventOverlay[];
};

export type ApiQueryStatus = "completed" | "clarification_needed" | "failed";

export type ApiPlanStep = {
  step_id?: string;
  tool?: string;
  save_as?: string;
  status?: string;
  human_label?: string;
  technical_label?: string;
};

export type ApiScientificFinding = {
  finding: string;
  evidence: string[];
  confidence?: "high" | "medium" | "low";
  result_ids?: string[];
};

export type PolicyGuidanceAction = {
  priority: "high" | "medium" | "low" | "screening";
  action_type:
    | "monitoring"
    | "source_control"
    | "discharge_outlet"
    | "river_estuary"
    | "seasonal_management"
    | "coastal_planning"
    | "economic_assessment"
    | "governance";
  target: string;
  where_when: string;
  evidence_basis: string;
  recommendation: string;
  guardrail: string;
  evidence_strength: "supported" | "limited" | "screening" | "not_supported";
};

export type PolicyGuidance = {
  should_include: boolean;
  headline?: string;
  place_based_policy_brief?: string;
  evidence_action_matrix?: PolicyGuidanceAction[];
  evidence_limits?: string[];
};

export type PolicySynthesisDecisionRow = {
  decision_unit: string;
  action_group?:
    | "spatial_priority"
    | "oxygen_response"
    | "seasonal_operations"
    | "driver_adaptation"
    | "source_pathway_screening"
    | "economic_data_assessment"
    | "validation_gap";
  target?: string;
  policy_lever: string;
  where_when?: string;
  evidence_basis?: string;
  trigger_evidence: string;
  recommended_action: string;
  guardrail?: string;
  rationale?: string;
  confidence: "supported" | "limited" | "screening" | "not_supported";
  evidence_result_ids?: string[];
};

export type PolicySynthesisRecommendation = {
  policy_title: string;
  recommended_action: string;
  priority_places?: string[];
  evidence_status?: "computed" | "indirect" | "data_gap";
  evidence_note?: string;
  evidence_result_ids?: string[];
  supporting_evidence?: string[];
  why_this_policy?: string;
  guardrail?: string;
};

export type PolicySynthesis = {
  one_sentence_judgment?: string;
  policy_narrative?: string;
  policy_recommendations?: PolicySynthesisRecommendation[];
  policy_frame?: string;
  decision_rows?: PolicySynthesisDecisionRow[];
};

export type IntegratedEvidenceThread = {
  theme: "warming/heatwave" | "bottom_oxygen/hypoxia" | "stratification" | "chlorophyll/bloom" | "data_gap";
  status: "computed" | "indirect" | "data_gap";
  evidence_summary: string;
  evidence_result_ids?: string[];
};

export type AnalysisProposal = {
  title: string;
  public_question: string;
  proposed_query: string;
  analysis_steps: string[];
  expected_outputs: string[];
  limitations: string[];
  approval_prompt: string;
  executable?: boolean;
  requires_revision?: boolean;
  synthesis_profile_id?: string;
  selected_skills?: string[];
  skill_plan?: {
    primary_skill?: string;
    skills_used?: string[];
    synthesis_profile_id?: string;
    planned_tools?: string[];
    planned_steps?: Array<{
      label?: string;
      tool?: string;
      purpose?: string;
    }>;
  };
  plan?: {
    steps?: unknown[];
  };
};

export type IntegratedAssessment = {
  profile_id?: string;
  direct_answer?: string;
  assessment_narrative?: string;
  evidence_threads?: IntegratedEvidenceThread[];
  higher_risk_regions?: Array<{
    region?: string;
    major_environmental_risks?: string;
    evidence?: string;
    evidence_result_ids?: string[];
  }>;
  suitability?: string;
  risk_hotspots?: string[];
  environmental_drivers?: string[];
  economic_implications?: string;
  environmental_protection_implications?: string;
  management_guidance?: string | string[];
  future_outlook?: string;
  uncertainty_and_data_gaps?: string[];
  evidence_boundary_notes?: string[];
  evidence_result_ids?: string[];
  policy_synthesis?: PolicySynthesis;
};

export type ApiSynthesis = {
  summary: string;
  scientific_findings?: ApiScientificFinding[];
  policy_guidance?: PolicyGuidance;
  integrated_assessment?: IntegratedAssessment;
  notable_patterns?: string[];
  anomalies?: { description: string; evidence: string[]; result_id?: string }[];
  significance_assessment?: { target: string; assessment: string; evidence: string[] }[];
  uncertainties?: string[];
  synthesis_warnings?: string[];
};

export type QueryApiResponse = {
  status: ApiQueryStatus;
  query: string;
  language?: "en" | null;
  conversation_id?: string | null;
  routing_mode?: "dataset_analysis" | "general_answer" | null;
  router_confidence?: number | null;
  router_reason?: string | null;
  skill_id?: string | null;
  skills_used: string[];
  clarification_question?: string | null;
  missing_fields: string[];
  analysis_proposal?: AnalysisProposal | null;
  dataset_info?: DatasetInfo;
  plan_summary?: string | null;
  plan_steps: ApiPlanStep[];
  step_cards: StepCard[];
  result_cards: ResultCardSummary[];
  result_summaries: Record<string, Record<string, unknown>>;
  synthesis?: ApiSynthesis | null;
  summary_status?: "pending" | "completed" | "failed";
  source_cards: SourceCard[];
  active_result_id?: string | null;
  active_map_step_id?: string | null;
  workspace_data?: Partial<WorkspaceData>;
  workspace_data_by_result?: Record<string, Partial<WorkspaceData>>;
  error?: string | null;
  failure_kind?: "capability_boundary" | "llm_format" | "planning" | "execution" | "synthesis" | "transport" | null;
  recoverable?: boolean | null;
  timings?: Record<string, number>;
};

export type VisualizationApiResponse = {
  status: "completed" | "failed";
  result_cards: ResultCardSummary[];
  active_result_id?: string | null;
  workspace_data?: Partial<WorkspaceData>;
  error?: string | null;
};

export type ChatAssistantState = "info" | "running" | "completed" | "clarification" | "failed";

export type AssistantMessagePayload = {
  state: ChatAssistantState | "planning";
  preferredLanguage?: "en";
  summary: string;
  note?: string;
  routingMode?: QueryApiResponse["routing_mode"];
  routerConfidence?: number;
  routerReason?: string;
  datasetInfo?: DatasetInfo;
  skillsUsed?: string[];
  planSummary?: string;
  planSteps: PlanStep[];
  resultCards: ResultCardSummary[];
  stepCards?: StepCard[];
  findings: ScientificFinding[];
  policyGuidance?: PolicyGuidance;
  integratedAssessment?: IntegratedAssessment;
  synthesisWarnings?: string[];
  analysisProposal?: AnalysisProposal;
  sourceCards?: SourceCard[];
  workspaceData?: WorkspaceData;
  workspaceDataByResult?: Record<string, WorkspaceData>;
  activeResultId?: string;
  failureKind?: QueryApiResponse["failure_kind"];
  recoverable?: boolean;
  timings?: Record<string, number>;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  payload?: AssistantMessagePayload;
};

export type ContextTagKind = "variable" | "region" | "time" | "depth";

export type ContextTag = {
  id: string;
  kind: ContextTagKind;
  label: string;
  value: string;
};

export type ChartTrayContent = {
  resultCard: ResultCardSummary;
  workspaceData: WorkspaceData;
};
