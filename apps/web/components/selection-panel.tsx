"use client";

import { useState } from "react";
import type { ManualViewState } from "@/lib/types";
import { clearGeometry, geometrySummary, nextSelectionMode, reverseTransect } from "@/lib/geometry-tools";
import { withRegionBounds } from "@/lib/manual-state";

type SelectionPanelProps = {
  state: ManualViewState;
  onChange: (next: ManualViewState) => void;
};

export function SelectionPanel({ state, onChange }: SelectionPanelProps) {
  const [isMoreGeometryOpen, setIsMoreGeometryOpen] = useState(false);
  const pointLabel = `${state.selectedPoint[1].toFixed(2)}E · ${state.selectedPoint[0].toFixed(2)}N`;
  const transectSummary = geometrySummary(state.transectPoints, "transect");
  const polygonSummary = geometrySummary(state.polygonPoints, "polygon");
  const isGeometryMode = state.selectionMode === "transect" || state.selectionMode === "polygon";
  const activeGeometrySummary =
    state.selectionMode === "transect"
      ? transectSummary
      : state.selectionMode === "polygon"
        ? polygonSummary
        : "None";

  const updateRegionBound = (key: keyof ManualViewState["regionBounds"], rawValue: string) => {
    const nextValue = Number.parseFloat(rawValue);
    if (!Number.isFinite(nextValue)) {
      return;
    }
    onChange(
      withRegionBounds(state, {
        ...state.regionBounds,
        [key]: nextValue,
      }),
    );
  };

  return (
    <section className="selection-panel ui-card">
      <div className="selection-panel-header">
        <div>
          <h2 className="ui-card-title">Spatial Selection</h2>
          <p className="ui-card-subtitle">Working region for analysis.</p>
        </div>
      </div>

      <div className="selection-panel-grid">
        <div className="selection-section selection-mode-section">
          <h3 className="ui-card-title">Mode</h3>
          <div className="mode-switch selection-mode-switch">
            {[
              { id: "box", label: "Box" },
              { id: "point", label: "Point" },
              { id: "transect", label: "Transect" },
              { id: "polygon", label: "Polygon" },
            ].map((mode) => (
              <button
                key={mode.id}
                className={`segment-button ${state.selectionMode === mode.id ? "is-active" : ""}`}
                onClick={() =>
                  onChange({
                    ...state,
                    selectionMode: nextSelectionMode(state.selectionMode, mode.id as ManualViewState["selectionMode"]),
                  })
                }
                type="button"
              >
                {mode.label}
              </button>
            ))}
          </div>
        </div>

        <div className="selection-section selection-region-card">
          <h3 className="ui-card-title">Region</h3>
          <div className="selection-region-inputs">
            <label className="bound-input">
              <span className="mini-label">Lon Min</span>
              <input
                onChange={(event) => updateRegionBound("lonMin", event.target.value)}
                step="0.1"
                type="number"
                value={state.regionBounds.lonMin}
              />
            </label>
            <label className="bound-input">
              <span className="mini-label">Lon Max</span>
              <input
                onChange={(event) => updateRegionBound("lonMax", event.target.value)}
                step="0.1"
                type="number"
                value={state.regionBounds.lonMax}
              />
            </label>
            <label className="bound-input">
              <span className="mini-label">Lat Min</span>
              <input
                onChange={(event) => updateRegionBound("latMin", event.target.value)}
                step="0.1"
                type="number"
                value={state.regionBounds.latMin}
              />
            </label>
            <label className="bound-input">
              <span className="mini-label">Lat Max</span>
              <input
                onChange={(event) => updateRegionBound("latMax", event.target.value)}
                step="0.1"
                type="number"
                value={state.regionBounds.latMax}
              />
            </label>
          </div>
        </div>

        <details className="selection-subcard selection-current-card selection-compact-details">
          <summary>
            <span className="ui-card-title">Current Selection</span>
            <span className="ui-card-subtitle">{state.regionLabel}</span>
          </summary>
          <div className="selection-summary-grid">
            <div>
              <span className="mini-label">Region</span>
              <strong>{state.regionLabel}</strong>
            </div>
            <div>
              <span className="mini-label">
                {state.selectionMode === "point"
                  ? "Point"
                  : state.selectionMode === "transect"
                    ? "Transect Center"
                    : state.selectionMode === "polygon"
                      ? "Polygon Center"
                      : "Region Center"}
              </span>
              <strong>{pointLabel}</strong>
            </div>
            <div>
              <span className="mini-label">{isGeometryMode ? state.selectionMode : "Geometry"}</span>
              <strong>{activeGeometrySummary}</strong>
            </div>
          </div>
        </details>

        <details
          className="selection-subcard selection-more-card"
          onToggle={(event) => setIsMoreGeometryOpen(event.currentTarget.open)}
          open={isMoreGeometryOpen}
        >
          <summary>
            <span className="ui-card-title">More Geometry</span>
            <span className="ui-card-subtitle">Transect / Polygon</span>
          </summary>
          <div className="geometry-tool-grid">
            <div className="geometry-tool-card">
              <span className="mini-label">Transect</span>
              <strong>{transectSummary}</strong>
              <div className="selection-action-row">
                <button className="result-expand-btn" onClick={() => onChange(reverseTransect(state))} type="button">
                  Reverse
                </button>
                <button className="result-expand-btn" onClick={() => onChange(clearGeometry(state, "transect"))} type="button">
                  Clear
                </button>
              </div>
            </div>
            <div className="geometry-tool-card">
              <span className="mini-label">Polygon</span>
              <strong>{polygonSummary}</strong>
              <div className="selection-action-row">
                <button className="result-expand-btn" onClick={() => onChange(clearGeometry(state, "polygon"))} type="button">
                  Clear
                </button>
              </div>
            </div>
          </div>
        </details>
      </div>
    </section>
  );
}
