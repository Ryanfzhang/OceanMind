"use client";

import type { DatasetInfo, ManualViewState, VerticalFeature } from "@/lib/types";
import {
  formatDatasetVariables,
  formatDepthCoverage,
  formatResolution,
  formatSpatialCoverage,
  formatTemporalCoverage,
} from "@/lib/dataset-display";
import { availableDatasetVariables } from "@/lib/dataset-state";
import { SelectionPanel } from "@/components/selection-panel";

type LeftControlPanelProps = {
  datasetInfo?: DatasetInfo | null;
  isBusy?: boolean;
  state: ManualViewState;
  onChange: (next: ManualViewState) => void;
  onQuickVisualize?: () => void;
  canQuickVisualize?: boolean;
};

const features: VerticalFeature[] = ["mixed_layer", "thermocline", "pycnocline"];
const layerMeanPresets = [
  "surface -> mixed_layer",
  "surface -> thermocline",
  "surface -> pycnocline",
  "mixed_layer -> thermocline"
];

export function LeftControlPanel({
  datasetInfo = null,
  isBusy = false,
  state,
  onChange,
  onQuickVisualize,
  canQuickVisualize = false,
}: LeftControlPanelProps) {
  const availableVariables = availableDatasetVariables(datasetInfo);
  const availableDepths = Array.isArray(datasetInfo?.depth_levels)
    ? datasetInfo.depth_levels.filter((value): value is number => typeof value === "number" && Number.isFinite(value))
    : [];
  const isConfigLoading = datasetInfo === null;
  const hasAvailableVariables = availableVariables.length > 0;
  const hasAvailableDepths = availableDepths.length > 0;
  const selectedDepth = state.depthRange[0];
  const datasetName = datasetInfo?.name ?? "CMOMS";
  const datasetDescription =
    datasetInfo?.description ??
    "CMOMS provides gridded upper-ocean state variables for map-first exploration and agent-driven scientific analysis.";
  const updateTimeRange = (index: 0 | 1, value: string) => {
    if (!value) {
      return;
    }
    const nextTimeRange: [string, string] =
      index === 0 ? [value, state.timeRange[1]] : [state.timeRange[0], value];
    onChange({
      ...state,
      timeRange: nextTimeRange,
      timeLabel: `${nextTimeRange[0]} ~ ${nextTimeRange[1]}`
    });
  };

  return (
    <aside className="left-panel">
      <section className="panel-section ui-card left-control-card">
        <div className="ui-card-header">
          <h2 className="ui-card-title">Dataset</h2>
        </div>
        <p className="ui-card-body dataset-summary">
          <strong>{datasetName}</strong>
        </p>
        <details className="ui-details dataset-details">
          <summary>Details</summary>
          <p className="ui-card-body">{datasetDescription}</p>
          <div className="dataset-meta-list">
            <div className="dataset-meta-row">
              <span className="mini-label">Variables</span>
              <span className="dataset-meta-value">{formatDatasetVariables(datasetInfo)}</span>
            </div>
            <div className="dataset-meta-row">
              <span className="mini-label">Spatial Coverage</span>
              <span className="dataset-meta-value">{formatSpatialCoverage(datasetInfo)}</span>
            </div>
            <div className="dataset-meta-row">
              <span className="mini-label">Temporal Coverage</span>
              <span className="dataset-meta-value">{formatTemporalCoverage(datasetInfo)}</span>
            </div>
            <div className="dataset-meta-row">
              <span className="mini-label">Depth Range</span>
              <span className="dataset-meta-value">{formatDepthCoverage(datasetInfo)}</span>
            </div>
            <div className="dataset-meta-row">
              <span className="mini-label">Resolution</span>
              <span className="dataset-meta-value">{formatResolution(datasetInfo)}</span>
            </div>
          </div>
        </details>
      </section>

      <SelectionPanel state={state} onChange={onChange} />

      <section className="panel-section ui-card left-control-card">
        <div className="ui-card-header">
          <h2 className="ui-card-title">Variable</h2>
        </div>
        {isConfigLoading ? (
          <p className="ui-card-body">Loading dataset configuration.</p>
        ) : hasAvailableVariables ? (
          <div className="token-grid">
            {availableVariables.map((variable) => (
              <button
                key={variable}
                className={`token-button ${state.variable === variable ? "is-active" : ""}`}
                onClick={() => onChange({ ...state, variable })}
                type="button"
              >
                {variable}
              </button>
            ))}
          </div>
        ) : (
          <p className="ui-card-body">No variables are declared in the dataset configuration.</p>
        )}
      </section>

      <section className="panel-section ui-card left-control-card">
        <div className="ui-card-header">
          <h2 className="ui-card-title">Time Range</h2>
        </div>
        <div className="date-input-grid">
          <label className="bound-input">
            <span className="mini-label">Start</span>
            <input
              onChange={(event) => updateTimeRange(0, event.target.value)}
              type="date"
              value={state.timeRange[0]}
            />
          </label>
          <label className="bound-input">
            <span className="mini-label">End</span>
            <input
              onChange={(event) => updateTimeRange(1, event.target.value)}
              type="date"
              value={state.timeRange[1]}
            />
          </label>
        </div>
      </section>

      <section className="panel-section ui-card left-control-card">
        <div className="ui-card-header">
          <h2 className="ui-card-title">Depth</h2>
        </div>
        <div className="mode-switch">
          {[
            { id: "fixed", label: "Fixed", disabled: !hasAvailableDepths },
            { id: "feature", label: "Feature", disabled: isConfigLoading || !hasAvailableVariables },
            { id: "layer_mean", label: "Layer Mean", disabled: isConfigLoading || !hasAvailableVariables }
          ].map((mode) => (
            <button
              key={mode.id}
              className={`segment-button ${state.depthMode === mode.id ? "is-active" : ""}`}
              disabled={mode.disabled}
              onClick={() => onChange({ ...state, depthMode: mode.id as ManualViewState["depthMode"] })}
              type="button"
            >
              {mode.label}
            </button>
          ))}
        </div>

        {state.depthMode === "fixed" && (
          isConfigLoading ? (
            <p className="ui-card-body">Loading dataset configuration.</p>
          ) : hasAvailableDepths ? (
            <div className="depth-level-list" role="listbox" aria-label="Available depth levels">
              {availableDepths.map((depth) => (
                <button
                  key={depth}
                  className={`depth-level-button ${selectedDepth === depth ? "is-active" : ""}`}
                  onClick={() => onChange({ ...state, depthRange: [depth, depth] })}
                  type="button"
                >
                  {depth} m
                </button>
              ))}
            </div>
          ) : (
            <p className="ui-card-body">No fixed depth levels are declared in the dataset configuration.</p>
          )
        )}

        {state.depthMode === "feature" && (
          <div className="token-grid">
            {features.map((feature) => (
              <button
                key={feature}
                className={`token-button ${state.feature === feature ? "is-active" : ""}`}
                onClick={() => onChange({ ...state, feature })}
                type="button"
              >
                {feature}
              </button>
            ))}
          </div>
        )}

        {state.depthMode === "layer_mean" && (
          <div className="stack-list">
            {layerMeanPresets.map((label) => (
              <button
                key={label}
                className={`list-button ${state.layerMeanLabel === label ? "is-active" : ""}`}
                onClick={() => onChange({ ...state, layerMeanLabel: label })}
                type="button"
              >
                {label}
              </button>
            ))}
          </div>
        )}
      </section>

      <section className="panel-section left-actions">
        <button
          className="primary-button"
          disabled={isBusy || !canQuickVisualize}
          onClick={onQuickVisualize}
          type="button"
        >
          {isBusy ? "Running..." : canQuickVisualize ? "Quick Visualize" : "Config Required"}
        </button>
      </section>
    </aside>
  );
}
