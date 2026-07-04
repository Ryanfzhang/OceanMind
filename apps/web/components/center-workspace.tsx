import { MainMap } from "@/components/main-map";
import type { ManualViewState, ResultCardSummary, WorkspaceData } from "@/lib/types";

type CenterWorkspaceProps = {
  activeResult: ResultCardSummary | null;
  activeEventId?: string | null;
  manualState: ManualViewState;
  onManualStateChange: (next: ManualViewState) => void;
  onSelectEvent?: (eventId: string) => void;
  workspaceData: WorkspaceData;
};

export function CenterWorkspace({
  activeResult,
  activeEventId = null,
  manualState,
  onManualStateChange,
  onSelectEvent,
  workspaceData
}: CenterWorkspaceProps) {
  return (
    <section className="center-workspace panel-surface">
      <div className="workspace-body">
        <div className="map-region">
          <MainMap
            activeResult={activeResult}
            activeEventId={activeEventId}
            manualState={manualState}
            onManualStateChange={onManualStateChange}
            onSelectEvent={onSelectEvent}
            workspaceData={workspaceData}
          />
        </div>
      </div>
    </section>
  );
}
