import type {
  ManualViewState,
  PlanStep,
  ResultCardSummary,
  ScientificFinding,
  WorkspaceData
} from "@/lib/types";

export const manualViewState: ManualViewState = {
  dataset: "CMOMS",
  variable: "temp",
  timeRange: ["2011-01-01", "2011-01-10"],
  timeLabel: "2011-01-01 ~ 2011-01-10",
  regionBounds: {
    lonMin: 99.9,
    lonMax: 150.1,
    latMin: -0.1,
    latMax: 52.0
  },
  regionLabel: "99.9-150.1E · -0.1-52N",
  selectedPoint: [25.95, 125],
  selectionMode: "none",
  transectPoints: [],
  polygonPoints: [],
  depthMode: "fixed",
  availableDepths: [],
  depthRange: [0, 0],
  feature: "thermocline",
  layerMeanLabel: "surface -> thermocline"
};

export const planSteps: PlanStep[] = [
  { id: "step_1", tool: "load_dataset", status: "completed" },
  { id: "step_2", tool: "identify_thermocline_depth", status: "completed" },
  { id: "step_3", tool: "compute_layer_mean", status: "completed" },
  { id: "step_4", tool: "extract_regional_mean", status: "completed" },
  { id: "step_5", tool: "compute_trend", status: "active" }
];

export const resultCards: ResultCardSummary[] = [
  {
    id: "reference",
    title: "Reference Surface Temp",
    type: "data_container_result",
    headline: "Original variable for direct visualization",
    description: "Use this tab to keep the base field visible while comparing analysis outputs.",
    renderer: "reference",
    metrics: [
      { label: "Variable", value: "temp" },
      { label: "Depth", value: "surface" },
      { label: "Mean", value: "23.1 C" }
    ]
  },
  {
    id: "timeseries-layer-mean",
    title: "Surface-to-Thermocline Mean Temp",
    type: "trend_result",
    headline: "Cooling trend with anomaly context",
    description: "Regional layer-mean temperature timeseries built from the surface to the diagnosed thermocline.",
    renderer: "timeseries",
    metrics: [
      { label: "Trend", value: "-0.12 C / 10d" },
      { label: "p-value", value: "0.03" },
      { label: "R²", value: "0.81" }
    ]
  },
  {
    id: "profile-overlay",
    title: "Profile with Thermocline Marker",
    type: "profile_result",
    headline: "Point profile with vertical structure marker",
    description: "Vertical temperature profile at 111E, 19N with a diagnosed thermocline overlay.",
    renderer: "profile",
    metrics: [
      { label: "Point", value: "111E · 19N" },
      { label: "Thermocline", value: "-42 m" },
      { label: "Surface", value: "24.6 C" }
    ]
  },
  {
    id: "hovmoller-overlay",
    title: "Time-Depth Hovmoller",
    type: "hovmoller_result",
    headline: "Upper-ocean evolution with MLD overlay",
    description: "Time-depth temperature section with mixed-layer depth plotted as an overlay line.",
    renderer: "hovmoller",
    metrics: [
      { label: "Overlay", value: "MLD" },
      { label: "Window", value: "10 days" },
      { label: "Depth", value: "0 to -100 m" }
    ]
  },
  {
    id: "histogram-pycnocline",
    title: "Pycnocline Depth Distribution",
    type: "histogram_result",
    headline: "Distribution of diagnosed pycnocline depths",
    description: "Histogram summarizing where density-gradient maxima cluster in the region.",
    renderer: "histogram",
    metrics: [
      { label: "Mode", value: "-35 m" },
      { label: "Spread", value: "12 bins" },
      { label: "p95", value: "-15 m" }
    ]
  },
  {
    id: "eof-thermocline",
    title: "Thermocline EOF",
    type: "eof_result",
    headline: "Leading mode of thermocline-depth variability",
    description: "Spatial EOF mode on thermocline depth with a companion PC timeseries.",
    renderer: "eof",
    metrics: [
      { label: "Mode 1", value: "48%" },
      { label: "Mode 2", value: "21%" },
      { label: "PC Peak", value: "Jan 08" }
    ]
  }
];

export const findings: ScientificFinding[] = [
  {
    title: "Surface-to-thermocline mean temperature cools over the analysis window.",
    evidence: ["Trend is negative over 10 days", "p-value remains below 0.05", "Cooling aligns with the cold anomaly on Jan 10"]
  },
  {
    title: "Thermocline depth varies enough to justify using layer semantics instead of a fixed depth range.",
    evidence: ["Diagnosed thermocline ranges from -34 m to -48 m", "Layer-mean path differs from a fixed -50 m average"]
  },
  {
    title: "Map-first layout keeps all non-map charts anchored to a visible region or point selection.",
    evidence: ["Timeseries uses the highlighted region", "Profile and hovmoller share a point marker on the map"]
  }
];

export const workspaceData: WorkspaceData = {
  referenceSeries: [
    { label: "Jan 01", value: 23.6 },
    { label: "Jan 02", value: 23.4 },
    { label: "Jan 03", value: 23.1 },
    { label: "Jan 04", value: 22.9 },
    { label: "Jan 05", value: 22.7 },
    { label: "Jan 06", value: 22.6 },
    { label: "Jan 07", value: 22.5 }
  ],
  resultSeries: [
    { label: "Jan 01", value: 22.9 },
    { label: "Jan 02", value: 22.8 },
    { label: "Jan 03", value: 22.7 },
    { label: "Jan 04", value: 22.6 },
    { label: "Jan 05", value: 22.4 },
    { label: "Jan 06", value: 22.2 },
    { label: "Jan 07", value: 22.1 }
  ],
  anomalySeries: [
    { label: "Jan 01", value: 0.28 },
    { label: "Jan 02", value: 0.16 },
    { label: "Jan 03", value: 0.07 },
    { label: "Jan 04", value: -0.03 },
    { label: "Jan 05", value: -0.08 },
    { label: "Jan 06", value: -0.15 },
    { label: "Jan 07", value: -0.21 }
  ],
  tsDiagramPoints: [],
  tsDiagramTemperatureLabel: "Temperature",
  tsDiagramSalinityLabel: "Salinity",
  tsDiagramColorLabel: null,
  tsDiagramColorRange: null,
  tsDiagramPointClasses: [],
  tsDiagramClassColorMap: {},
  tsDiagramWatermassBins: [],
  profileSeries: [
    { depth: 0, value: 24.6 },
    { depth: -10, value: 24.1 },
    { depth: -20, value: 23.5 },
    { depth: -30, value: 22.7 },
    { depth: -40, value: 21.2 },
    { depth: -50, value: 19.6 },
    { depth: -60, value: 18.9 },
    { depth: -80, value: 18.1 },
    { depth: -100, value: 17.6 }
  ],
  profileMarkers: [
    { label: "Thermocline", depth: -42 },
    { label: "MLD", depth: -28 }
  ],
  hovmollerRows: [
    { depthLabel: "0 m", depthValue: 0, values: [8, 8, 7, 6, 5, 5, 4] },
    { depthLabel: "-20 m", depthValue: -20, values: [7, 7, 6, 5, 5, 4, 4] },
    { depthLabel: "-40 m", depthValue: -40, values: [6, 6, 5, 5, 4, 4, 3] },
    { depthLabel: "-60 m", depthValue: -60, values: [5, 5, 4, 4, 3, 3, 2] },
    { depthLabel: "-80 m", depthValue: -80, values: [4, 4, 3, 3, 3, 2, 2] },
    { depthLabel: "-100 m", depthValue: -100, values: [3, 3, 3, 2, 2, 2, 1] }
  ],
  hovmollerTimeLabels: ["Jan 01", "Jan 02", "Jan 03", "Jan 04", "Jan 05", "Jan 06", "Jan 07"],
  hovmollerDepthIntegratedSeries: [
    { label: "Jan 01", value: 560 },
    { label: "Jan 02", value: 560 },
    { label: "Jan 03", value: 460 },
    { label: "Jan 04", value: 390 },
    { label: "Jan 05", value: 350 },
    { label: "Jan 06", value: 300 },
    { label: "Jan 07", value: 260 }
  ],
  sectionRows: [],
  sectionDistanceKm: [],
  sectionAxisTitle: "",
  sectionSliceLabel: "",
  overlaySeries: [
    { day: "Jan 01", depth: -38 },
    { day: "Jan 02", depth: -41 },
    { day: "Jan 03", depth: -36 },
    { day: "Jan 04", depth: -33 },
    { day: "Jan 05", depth: -35 },
    { day: "Jan 06", depth: -39 },
    { day: "Jan 07", depth: -42 }
  ],
  histogramBins: [
    { label: "-60", value: 3 },
    { label: "-55", value: 5 },
    { label: "-50", value: 9 },
    { label: "-45", value: 13 },
    { label: "-40", value: 16 },
    { label: "-35", value: 19 },
    { label: "-30", value: 15 },
    { label: "-25", value: 10 },
    { label: "-20", value: 6 },
    { label: "-15", value: 4 }
  ],
  eofVariance: [
    { label: "Mode 1", value: "48%" },
    { label: "Mode 2", value: "21%" },
    { label: "Mode 3", value: "11%" }
  ],
  eofPcSeries: [
    { day: "Jan 01", value: 0.22 },
    { day: "Jan 03", value: 0.48 },
    { day: "Jan 05", value: 0.11 },
    { day: "Jan 08", value: 0.79 },
    { day: "Jan 10", value: 0.31 }
  ],
  eofModes: [],
  compositeFields: [],
  eventOverlays: [],
  mapField: {
    lon: [110, 110.25, 110.5, 110.75, 111, 111.25, 111.5, 111.75, 112],
    lat: [20, 19.75, 19.5, 19.25, 19, 18.75, 18.5, 18.25, 18],
    values: [
      [23.1, 23.2, 23.3, 23.4, 23.5, 23.6, 23.5, 23.4, 23.2],
      [22.9, 23.0, 23.1, 23.3, 23.4, 23.5, 23.4, 23.2, 23.0],
      [22.7, 22.8, 23.0, 23.1, 23.3, 23.4, 23.2, 23.0, 22.8],
      [22.5, 22.7, 22.8, 23.0, 23.1, 23.2, 23.0, 22.8, 22.7],
      [22.3, 22.5, 22.7, 22.9, 23.0, 23.1, 22.9, 22.7, 22.5],
      [22.1, 22.3, 22.5, 22.7, 22.9, 23.0, 22.8, 22.6, 22.4],
      [21.9, 22.1, 22.3, 22.6, 22.7, 22.8, 22.6, 22.4, 22.2],
      [21.7, 21.9, 22.1, 22.3, 22.5, 22.6, 22.4, 22.2, 22.0],
      [21.6, 21.7, 21.9, 22.1, 22.3, 22.4, 22.2, 22.0, 21.8]
    ],
    label: "Surface temperature",
    variable: "temp",
    units: "C",
    statistics: {
      mean: 22.67,
      min: 21.6,
      max: 23.6,
      std: 0.49
    },
    depthLabel: "0 to -100 m mean",
    timeLabel: "2011-01-01 ~ 2011-01-10"
  }
};
