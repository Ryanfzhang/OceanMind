"use client";

import dynamic from "next/dynamic";
import type { ManualViewState, ResultCardSummary, WorkspaceData } from "@/lib/types";

type MainMapProps = {
  activeResult: ResultCardSummary | null;
  activeEventId?: string | null;
  manualState: ManualViewState;
  onManualStateChange: (next: ManualViewState) => void;
  onSelectEvent?: (eventId: string) => void;
  workspaceData: WorkspaceData;
};

const LeafletMapStage = dynamic(() => import("@/components/main-map-leaflet"), {
  ssr: false,
  loading: () => (
    <div className="map-stage">
      <div className="map-canvas map-loading-shell">
        <div className="empty-state">Preparing interactive map tiles and geographic overlays.</div>
      </div>
    </div>
  )
});

export function MainMap({
  activeResult,
  activeEventId = null,
  manualState,
  onManualStateChange,
  onSelectEvent,
  workspaceData
}: MainMapProps) {
  return (
    <LeafletMapStage
      activeResult={activeResult}
      activeEventId={activeEventId}
      manualState={manualState}
      onManualStateChange={onManualStateChange}
      onSelectEvent={onSelectEvent}
      workspaceData={workspaceData}
    />
  );
}
