"use client";

import { useState } from "react";
import type { ChartTrayContent, MapFieldData, ResultCardSummary, WorkspaceData } from "@/lib/types";
import { renderDockPanel } from "@/components/renderers";
import { normalizeWorkspaceData } from "@/lib/workspace-results";

type ChartTrayProps = {
  content: ChartTrayContent | null;
  onClose: () => void;
  onPromoteMapField?: (card: ResultCardSummary, data: WorkspaceData, field: MapFieldData) => void;
};

export function ChartTray({ content, onClose, onPromoteMapField }: ChartTrayProps) {
  const [height, setHeight] = useState(50);
  const isOpen = content !== null;

  if (!isOpen || !content) {
    return null;
  }

  const workspaceData = normalizeWorkspaceData(content.workspaceData);
  const resultCard = {
    ...content.resultCard,
    workspaceData,
  };

  return (
    <>
      <div className="chart-tray-backdrop" onClick={onClose} />
      <div className="chart-tray" style={{ height: `${height}%` }}>
        <div className="chart-tray-header">
          <h3>{resultCard.title}</h3>
          <button className="chart-tray-close" onClick={onClose} type="button">
            ×
          </button>
        </div>
        <div className="chart-tray-body">
          {renderDockPanel(resultCard, workspaceData, {
            onPromoteMapField: (field) => onPromoteMapField?.(resultCard, workspaceData, field),
          })}
        </div>
        <div
          className="chart-tray-handle"
          onMouseDown={(e) => {
            e.preventDefault();
            const startY = e.clientY;
            const startHeight = height;
            const onMove = (me: MouseEvent) => {
              const delta = ((startY - me.clientY) / window.innerHeight) * 100;
              setHeight(Math.max(30, Math.min(80, startHeight + delta)));
            };
            const onUp = () => {
              document.removeEventListener("mousemove", onMove);
              document.removeEventListener("mouseup", onUp);
            };
            document.addEventListener("mousemove", onMove);
            document.addEventListener("mouseup", onUp);
          }}
        />
      </div>
    </>
  );
}
