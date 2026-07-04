import { renderDockPanel } from "@/components/renderers";
import type { ResultCardSummary, WorkspaceData } from "@/lib/types";

type AnalysisDockProps = {
  activeResult?: ResultCardSummary | null;
  activeEventId?: string | null;
  workspaceData: WorkspaceData;
  onSelectEvent?: (eventId: string) => void;
  title?: string;
  emptyMessage?: string;
};

export function AnalysisDock({
  activeResult,
  activeEventId = null,
  workspaceData,
  onSelectEvent,
  title = "Results",
  emptyMessage = "Run a query or open a result to see the visualization here."
}: AnalysisDockProps) {
  return (
    <aside className="analysis-dock">
      <div className="analysis-dock-header">
        <div className="analysis-dock-title">
          <h3>{title}</h3>
        </div>
      </div>
      <div className="analysis-dock-body">
        {activeResult ? (
          renderDockPanel(activeResult, workspaceData, {
            activeEventId,
            onSelectEvent,
          })
        ) : (
          <div className="empty-state">{emptyMessage}</div>
        )}
      </div>
    </aside>
  );
}
