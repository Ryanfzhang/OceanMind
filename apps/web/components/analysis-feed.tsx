"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type {
  AnalysisProposal,
  ChatMessage,
  IntegratedAssessment,
  MapFieldData,
  PolicyGuidance,
  PolicyGuidanceAction,
  ResultCardSummary,
  SourceCard,
  StepCard,
  WorkspaceData,
} from "@/lib/types";
import { renderDockPanel, renderInlineChart } from "@/components/renderers";
import { shouldShowExecutionProgress } from "@/lib/assistant-display";
import { buildCompletedSummaryContent } from "@/lib/assistant-summary";
import { evidenceLinkedPolicyCardsForDisplay } from "@/lib/policy-guidance-display";
import { displayLooseResults, displayStepDetails, displayStepTimeline, latestFigureResultIds, shouldShowStepFallbackInterpretation } from "@/lib/step-card-state";
import { normalizeWorkspaceData } from "@/lib/workspace-results";

type QuerySubmitOptions = {
  continuePending?: boolean;
  additionalContext?: Record<string, unknown>;
};

type AnalysisFeedProps = {
  conversationId?: string | null;
  canExportReport?: boolean;
  isExportingReport?: boolean;
  messages: ChatMessage[];
  isBusy?: boolean;
  isLocked?: boolean;
  onExportReport?: () => void;
  queryText: string;
  onOpenDetail?: (card: ResultCardSummary, data: WorkspaceData) => void;
  onPromoteMapField?: (card: ResultCardSummary, data: WorkspaceData, field: MapFieldData) => void;
  onResultAction?: (card: ResultCardSummary, data: WorkspaceData, actionId: string) => void;
  onQueryChange: (text: string) => void;
  onSubmitQuery: (queryOverride?: string, options?: QuerySubmitOptions) => void;
  onToggleStepCard?: (messageId: string, stepId: string) => void;
};

function MarkdownSummary({ text }: { text: string }) {
  return <div className="markdown-summary"><ReactMarkdown
    remarkPlugins={[remarkGfm]}
    skipHtml
    components={{table: ({ children }) => <div className="markdown-table-wrap"><table>{children}</table></div>}}
  >{text}</ReactMarkdown></div>;
}

function statusIcon(status: StepCard["status"]) {
  switch (status) {
    case "completed":
      return "✓";
    case "pending":
      return "○";
    case "failed":
    case "running":
    default:
      return "⠿";
  }
}

function assistantProgressIconState(state: ChatMessage["payload"] | undefined) {
  return assistantProgressIcon(state?.state);
}

function assistantProgressIcon(
  state: "info" | "running" | "completed" | "clarification" | "failed" | "planning" | undefined,
) {
  switch (state) {
    case "planning":
      return "⋯";
    case "completed":
      return "✓";
    case "failed":
      return "!";
    case "clarification":
      return "?";
    case "running":
    case "info":
    default:
      return "⠿";
  }
}

function buildSourceSummary(source: SourceCard) {
  const parts = [source.short_snippet]
    .map((item) => item?.trim())
    .filter((item): item is string => Boolean(item));
  if (parts.length === 0) {
    return "";
  }
  return parts.join(" ");
}

function buildWebSearchHeader(sourceCards: SourceCard[], chinese: boolean) {
  const query = sourceCards.find((source) => source.search_query?.trim())?.search_query?.trim();
  return {
    title: chinese ? "外部来源" : "External Sources",
    query,
  };
}

function isNoUsableExternalSourcesCard(source: SourceCard) {
  return source.title.trim().toLowerCase() === "no usable external sources found";
}

function hasUsableSourceUrl(source: SourceCard) {
  return /^https?:\/\//i.test(source.url?.trim() ?? "");
}

function formatElapsedTime(seconds: number) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function resultImageUrl(conversationId: string, artifactId: string) {
  return `/api/results?${new URLSearchParams({ conversation: conversationId, mode: "image", artifact: artifactId })}`;
}

function ResultImage({ url, title, mapSupplement = false }: {
  url: string;
  title: string;
  mapSupplement?: boolean;
}) {
  const [imageExpanded, setImageExpanded] = useState(false);
  useEffect(() => {
    if (!imageExpanded) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setImageExpanded(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [imageExpanded]);
  return <>
    <button className="result-inline-image-button" type="button"
      aria-label={`Enlarge ${title}`} onClick={() => setImageExpanded(true)}>
      <img className={`result-inline-image${mapSupplement ? " is-map-supplement" : ""}`}
        src={url} alt={title} />
      <span className="result-image-expand-hint">Click to enlarge</span>
    </button>
    {imageExpanded ? createPortal(
      <div className="result-image-dialog" role="dialog" aria-modal="true"
        aria-label={title} onClick={(event) => {
          if (event.target === event.currentTarget) setImageExpanded(false);
        }}>
        <button className="result-image-dialog-close" type="button"
          aria-label="Close enlarged image" onClick={() => setImageExpanded(false)} autoFocus>×</button>
        <img className="result-image-dialog-image" src={url} alt={title} />
      </div>, document.body,
    ) : null}
  </>;
}

function StepResultContent({
  card,
  conversationId,
  onOpenDetail,
  onPromoteMapField,
  onResultAction,
  showFullVisualization = false,
  preferredLanguage,
}: {
  card: ResultCardSummary;
  conversationId?: string | null;
  onOpenDetail?: (card: ResultCardSummary, data: WorkspaceData) => void;
  onPromoteMapField?: (card: ResultCardSummary, data: WorkspaceData, field: MapFieldData) => void;
  onResultAction?: (card: ResultCardSummary, data: WorkspaceData, actionId: string) => void;
  showFullVisualization?: boolean;
  preferredLanguage?: "en";
}) {
  const chinese = false;
  const imageUrl = conversationId && card.type === "image_png"
    ? resultImageUrl(conversationId, card.id)
    : null;
  const workspaceData = card.workspaceData ? normalizeWorkspaceData(card.workspaceData) : undefined;
  const inlineChart =
    workspaceData && !showFullVisualization && card.type !== "image_png"
      ? renderInlineChart(card, workspaceData, {
          onPromoteMapField: (field) => onPromoteMapField?.(card, workspaceData, field),
        })
      : null;
  const usesInlineMapPreview = card.surface === "map" && card.renderer === "summary" && Boolean(workspaceData?.mapField);
  const hasLoadedMapPayload = Boolean(workspaceData?.mapField || workspaceData?.eventOverlays?.length);
  const usesWideInlineChart = card.renderer === "hovmoller" || card.renderer === "eof";
  const fullVisualization =
    showFullVisualization && workspaceData && (card.renderer !== "reference" || card.surface === "inline") ? (
      <div className="result-inline-full-visualization">
        {renderDockPanel(card, workspaceData, {
          onPromoteMapField: (field) => onPromoteMapField?.(card, workspaceData, field),
          showSummary: false,
        })}
      </div>
    ) : null;
  const showExpandButton =
    !showFullVisualization &&
    (card.surface === "drawer" || card.surface === "modal") &&
    onOpenDetail &&
    workspaceData;

  return (
    <div className={`result-inline-card result-surface-${card.surface ?? "summary"} ${showFullVisualization ? "show-full-visualization" : ""}`}>
      <div className="result-inline-header">
        <h4>{card.title}</h4>
        {card.surface === "map" && hasLoadedMapPayload ? <span className="result-map-badge">{chinese ? "📍 已加载到地图" : "📍 Loaded on map"}</span> : null}
      </div>
      {imageUrl ? <ResultImage url={imageUrl} title={card.title}
        mapSupplement={card.surface === "map"} /> : null}
      {inlineChart ? <div className={`result-inline-chart ${usesWideInlineChart ? "is-wide" : ""}`}>{inlineChart}</div> : null}
      <p className="result-inline-headline">{card.headline}</p>
      {card.description ? <p className="result-inline-description">{card.description}</p> : null}
      {card.metrics.length > 0 ? (
        <div className="result-inline-metrics">
          {card.metrics.map((metric) => (
            <span key={`${card.id}-${metric.label}`} className="metric-inline">
              {metric.label}: <strong>{metric.value}</strong>
            </span>
          ))}
        </div>
      ) : null}
      {card.detailSections && card.detailSections.length > 0 ? (
        <div className="result-detail-section-list">
          {card.detailSections.map((section) => (
            <div key={`${card.id}-${section.title}`} className="result-detail-section">
              <span className="mini-label">{section.title}</span>
              {section.items.map((item) => (
                <p key={`${card.id}-${section.title}-${item}`}>{item}</p>
              ))}
            </div>
          ))}
        </div>
      ) : null}
      {card.surface === "map" && !usesInlineMapPreview ? (
        <div className="step-card-action-row">
          {(card.actions ?? []).map((action) => (
            <button
              key={`${card.id}-${action.id}`}
              className="result-expand-btn"
              onClick={() => workspaceData && onResultAction?.(card, workspaceData, action.id)}
              type="button"
            >
              {action.label}
            </button>
          ))}
        </div>
      ) : null}
      {showExpandButton ? (
        <button className="result-expand-btn" onClick={() => onOpenDetail(card, workspaceData)} type="button">
          {card.surface === "drawer" ? (chinese ? "展开详情" : "Expand details") : (chinese ? "查看完整分析" : "View full analysis")}
        </button>
      ) : null}
      {fullVisualization}
      {card.interpretation ? (
        <div className="result-interpretation-card">
          <span className="mini-label">{chinese ? "科学解释" : "Scientific Interpretation"}</span>
          <p>{card.interpretation}</p>
        </div>
      ) : null}
    </div>
  );
}

function StepCardBlock({
  conversationId,
  messageId,
  step,
  onOpenDetail,
  onPromoteMapField,
  onResultAction,
  onToggleStepCard,
  preferredLanguage,
}: {
  conversationId?: string | null;
  messageId: string;
  step: StepCard;
  onOpenDetail?: (card: ResultCardSummary, data: WorkspaceData) => void;
  onPromoteMapField?: (card: ResultCardSummary, data: WorkspaceData, field: MapFieldData) => void;
  onResultAction?: (card: ResultCardSummary, data: WorkspaceData, actionId: string) => void;
  onToggleStepCard?: (messageId: string, stepId: string) => void;
  preferredLanguage?: "en";
}) {
  const chinese = false;
  const canExpand = step.status !== "running" && step.status !== "pending" && step.results.length > 0;
  const showFallbackInterpretation = shouldShowStepFallbackInterpretation(step);
  const displayStatus = step.status;

  return (
    <div className={`step-card status-${displayStatus} ${step.is_expanded ? "is-expanded" : ""}`}>
      <button
        className="step-card-header"
        onClick={() => canExpand && onToggleStepCard?.(messageId, step.step_id)}
        type="button"
      >
        <div className="step-card-heading">
          <span className="step-card-icon">{statusIcon(displayStatus)}</span>
          <div>
            <strong>{step.human_label}</strong>
            <p>{step.technical_label}</p>
          </div>
        </div>
        <span className="step-card-toggle">{canExpand ? (step.is_expanded ? "▾" : "▸") : ""}</span>
      </button>

      {step.is_expanded ? (
        <div className="step-card-body">
          {step.results.map((card) => (
            <StepResultContent
              key={card.id}
              card={card}
              conversationId={conversationId}
              onOpenDetail={onOpenDetail}
              onPromoteMapField={onPromoteMapField}
              onResultAction={onResultAction}
              preferredLanguage={preferredLanguage}
            />
          ))}

          {showFallbackInterpretation ? (
            <div className="step-interpretation-card">
              <span className="mini-label">{chinese ? "科学解释" : "Scientific Interpretation"}</span>
              <p>{step.interpretation}</p>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function AssistantBlock({
  conversationId,
  message,
  onOpenDetail,
  onPromoteMapField,
  onResultAction,
  onQueryChange,
  onSubmitQuery,
  onToggleStepCard,
}: {
  conversationId?: string | null;
  message: ChatMessage;
  onOpenDetail?: (card: ResultCardSummary, data: WorkspaceData) => void;
  onPromoteMapField?: (card: ResultCardSummary, data: WorkspaceData, field: MapFieldData) => void;
  onResultAction?: (card: ResultCardSummary, data: WorkspaceData, actionId: string) => void;
  onQueryChange: (text: string) => void;
  onSubmitQuery: (queryOverride?: string, options?: QuerySubmitOptions) => void;
  onToggleStepCard?: (messageId: string, stepId: string) => void;
}) {
  const [planExpanded, setPlanExpanded] = useState(false);
  const payload = message.payload;
  const chinese = false;
  const showExecutionProgress = shouldShowExecutionProgress(payload);
  const [clockNow, setClockNow] = useState(() => Date.now());
  useEffect(() => {
    if (!showExecutionProgress) return;
    setClockNow(Date.now());
    const timer = window.setInterval(() => setClockNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [showExecutionProgress]);
  const elapsedSeconds = typeof payload?.workflowStartedAt === "number"
    ? Math.max(0, Math.floor(((payload.workflowFinishedAt ?? clockNow) - payload.workflowStartedAt) / 1000))
    : null;
  const allStepCards = payload?.stepCards ?? [];
  const allResultCards = payload?.resultCards ?? [];
  const currentFigures = latestFigureResultIds(
    allResultCards, allStepCards, payload?.state === "completed",
  );
  const keepResult = (card: ResultCardSummary) =>
    card.type !== "image_png" || currentFigures.has(card.id);
  const stepCards = allStepCards.map((step) => ({
    ...step, results: step.results.filter(keepResult),
  }));
  const resultCards = allResultCards.filter(keepResult);
  const figureAttachments = resultCards.filter((card) => card.type === "image_png");
  const terminal = payload?.state === "completed" || payload?.state === "failed";
  const visibleSteps = displayStepTimeline(stepCards, terminal);
  const detailSteps = displayStepDetails(stepCards, visibleSteps);
  const looseResults = displayLooseResults(resultCards, stepCards);
  const planSteps = payload?.planSteps ?? [];
  const visibleStepIds = new Set(visibleSteps.map((step) => step.step_id));
  const visiblePlanSteps = planSteps.filter((step) => visibleStepIds.has(step.id));
  const sourceCards = payload?.sourceCards ?? [];
  const displaySourceCards = sourceCards.filter((source) => !isNoUsableExternalSourcesCard(source) && hasUsableSourceUrl(source));
  const webSearchHeader = buildWebSearchHeader(displaySourceCards, chinese);
  const summaryContent = buildCompletedSummaryContent(message);
  const completedSteps = visibleSteps.length + detailSteps.length;
  const progressLabel = completedSteps > 0 ? `${completedSteps} completed` : "";
  return (
    <div className="analysis-block">
      {showExecutionProgress || visibleSteps.length > 0 || looseResults.length > 0 ? (
        <section className="workflow-card ui-card">
          <div className="workflow-card-header">
            <div>
              <h4 className="ui-card-title">{chinese ? "工作流" : "Workflow"}</h4>
              <p className="ui-card-subtitle">{completedSteps > 0 ? `${completedSteps} completed steps` : "Execution steps"}</p>
            </div>
            <span>
              {elapsedSeconds !== null ? `${formatElapsedTime(elapsedSeconds)} · ` : ""}
              {payload?.state === "completed"
                ? "Complete"
                : payload?.state === "running" || payload?.state === "planning"
                  ? "Running"
                  : payload?.state === "failed"
                    ? "Finished"
                    : "Pending"}
            </span>
          </div>

          {showExecutionProgress ? (
            <div className="analysis-progress">
              <div className="progress-line">
                <span className="progress-spinner">{assistantProgressIconState(payload)}</span>
                <span>{payload?.summary ?? message.text}</span>
                <span>{progressLabel}</span>
                {visiblePlanSteps.length > 0 ? (
                  <button className="progress-expand" onClick={() => setPlanExpanded((current) => !current)} type="button">
                    {planExpanded ? (chinese ? "收起 ▴" : "Collapse ▴") : (chinese ? "展开 ▾" : "Expand ▾")}
                  </button>
                ) : null}
              </div>

              {planExpanded && visiblePlanSteps.length > 0 ? (
                <div className="plan-steps-list">
                  {visiblePlanSteps.map((step, index) => (
                    <div key={step.id} className="plan-step-item">
                      <span className="step-icon">{index + 1}.</span>
                      <span>{step.humanLabel ?? step.tool}</span>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}

          {visibleSteps.length > 0 || looseResults.length > 0 || detailSteps.length > 0 ? (
            <div className="step-card-timeline">
              {visibleSteps.map((step) => (
                <StepCardBlock
                  key={step.step_id}
                  conversationId={conversationId}
                  messageId={message.id}
                  onOpenDetail={onOpenDetail}
                  onPromoteMapField={onPromoteMapField}
                  onResultAction={onResultAction}
                  onToggleStepCard={onToggleStepCard}
                  preferredLanguage={payload?.preferredLanguage}
                  step={step}
                />
              ))}
              {looseResults.map((card) => (
                <StepResultContent
                  key={card.id}
                  card={card}
                  conversationId={conversationId}
                  onOpenDetail={onOpenDetail}
                  onPromoteMapField={onPromoteMapField}
                  onResultAction={onResultAction}
                  preferredLanguage={payload?.preferredLanguage}
                />
              ))}
              {detailSteps.length > 0 ? (
                <details className="workflow-details">
                  <summary>{`Show ${detailSteps.length} analysis ${detailSteps.length === 1 ? "step" : "steps"}`}</summary>
                  <div className="workflow-details-list">
                    {detailSteps.map((step) => (
                      <StepCardBlock
                        key={step.step_id}
                        conversationId={conversationId}
                        messageId={message.id}
                        onOpenDetail={onOpenDetail}
                        onPromoteMapField={onPromoteMapField}
                        onResultAction={onResultAction}
                        onToggleStepCard={onToggleStepCard}
                        preferredLanguage={payload?.preferredLanguage}
                        step={step}
                      />
                    ))}
                  </div>
                </details>
              ) : null}
            </div>
          ) : null}
        </section>
      ) : null}

      {payload?.state === "completed" && payload.summary ? (
        <div className="findings-section ui-card">
          <h4 className="findings-title ui-card-title">{chinese ? "回复" : "Response"}</h4>
          {conversationId && figureAttachments.length > 0 ? (
            <div className="response-figures">
              {figureAttachments.map((card) => (
                <figure key={card.id} className="response-figure">
                  <ResultImage url={resultImageUrl(conversationId, card.id)} title={card.title} />
                  <figcaption>{card.title}</figcaption>
                </figure>
              ))}
            </div>
          ) : null}
          <MarkdownSummary text={payload.summary} />

          {summaryContent.evidence.length > 0 ? (
            <div className="findings-evidence-list">
              <strong className="findings-evidence-title">{chinese ? "科学发现" : "Scientific Findings"}</strong>
              <ul>
                {summaryContent.evidence.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          ) : null}

        </div>
      ) : null}

      {payload?.state === "completed" ? (
        <PolicyGuidanceSection
          guidance={summaryContent.policyGuidance}
          integratedAssessment={summaryContent.integratedAssessment}
          synthesisWarnings={summaryContent.synthesisWarnings}
        />
      ) : null}

      {displaySourceCards.length > 0 ? (
        <div className="web-search-card ui-card">
          <div className="web-search-card-header">
            <div>
              <h4 className="ui-card-title">{webSearchHeader.title}</h4>
              {webSearchHeader.query ? <p className="ui-card-subtitle">{webSearchHeader.query}</p> : null}
            </div>
          </div>
          <div className="web-search-result-list">
            {displaySourceCards.map((source, index) => {
              const summary = buildSourceSummary(source);
              return (
                <article key={`${source.source}-${source.url}-${index}`} className="web-search-result-row">
                  <div className="web-search-result-body">
                    <a href={source.url} rel="noreferrer" target="_blank" className="web-search-result-title">
                      {source.title || source.source || (chinese ? "搜索结果" : "Search result")}
                    </a>
                    {summary ? <p>{summary}</p> : null}
                  </div>
                </article>
              );
            })}
          </div>
        </div>
      ) : null}

      {payload?.state === "clarification" && payload.analysisProposal ? (
        <AnalysisProposalCard
          proposal={payload.analysisProposal}
          onRevise={() => onQueryChange(`Revise the proposed analysis: ${payload.analysisProposal?.proposed_query ?? ""}`)}
          onRun={() =>
            onSubmitQuery("Run proposed analysis", {
              continuePending: true,
              additionalContext: {
                approved_analysis_proposal: payload.analysisProposal,
              },
            })
          }
        />
      ) : payload?.state === "clarification" ? (
        <div className="clarification-card ui-card">
          <strong className="ui-card-title">{chinese ? "需要补充信息" : "Clarification needed"}</strong>
          <p className="ui-card-body">{payload.summary}</p>
          {payload.note ? <p className="ui-card-body">{payload.note}</p> : null}
        </div>
      ) : null}

      {payload?.state === "failed" ? (
        <div className="failure-response-card ui-card">
          <strong className="ui-card-title">{chinese ? "回复" : "Response"}</strong>
          <p className="ui-card-body">{payload.summary}</p>
        </div>
      ) : null}
    </div>
  );
}

function formatPolicyLabel(value: string) {
  return value.replace(/_/g, " ");
}

function AnalysisProposalCard({
  proposal,
  onRun,
  onRevise,
}: {
  proposal: AnalysisProposal;
  onRun: () => void;
  onRevise: () => void;
}) {
  const steps = Array.isArray(proposal.analysis_steps) ? proposal.analysis_steps.filter(Boolean) : [];
  const outputs = Array.isArray(proposal.expected_outputs) ? proposal.expected_outputs.filter(Boolean) : [];
  const limitations = Array.isArray(proposal.limitations) ? proposal.limitations.filter(Boolean) : [];
  const selectedSkills = Array.isArray(proposal.selected_skills) ? proposal.selected_skills.filter(Boolean) : [];
  const plannedSteps = Array.isArray(proposal.skill_plan?.planned_steps)
    ? proposal.skill_plan.planned_steps.filter(
        (item) => Boolean(item?.label || item?.tool || item?.purpose),
      )
    : [];
  const plannedTools = Array.isArray(proposal.skill_plan?.planned_tools)
    ? proposal.skill_plan.planned_tools.filter(Boolean)
    : [];
  const hasExecutablePlan = Array.isArray(proposal.plan?.steps) && proposal.plan.steps.length > 0;
  const canRunProposal = proposal.executable !== false && proposal.requires_revision !== true && hasExecutablePlan;

  return (
    <div className="analysis-proposal-card">
      <div className="analysis-proposal-header">
        <span className="mini-label">Suggested Analysis Plan</span>
        <h4>{proposal.title || "Suggested analysis"}</h4>
      </div>
      {proposal.public_question ? <p className="analysis-proposal-question">{proposal.public_question}</p> : null}
      <div className="analysis-proposal-query">
        <span className="mini-label">Executable Query</span>
        <p>{proposal.proposed_query}</p>
      </div>
      {selectedSkills.length > 0 || plannedTools.length > 0 ? (
        <div className="analysis-proposal-skillbacking">
          <span className="mini-label">Skill-backed plan</span>
          {selectedSkills.length > 0 ? (
            <p>
              Skill: <strong>{selectedSkills.join(", ")}</strong>
            </p>
          ) : null}
          {plannedTools.length > 0 ? <p>Tools: {plannedTools.join(" -> ")}</p> : null}
        </div>
      ) : null}
      {steps.length > 0 ? (
        <div className="analysis-proposal-grid">
          <div>
            <span className="mini-label">Analysis Steps</span>
            <ul>
              {steps.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
          {outputs.length > 0 ? (
            <div>
              <span className="mini-label">Expected Outputs</span>
              <ul>
                {outputs.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : null}
      {plannedSteps.length > 0 ? (
        <div className="analysis-proposal-planned-steps">
          <span className="mini-label">Execution Backbone</span>
          <ol>
            {plannedSteps.map((item, index) => (
              <li key={`${item.label ?? item.tool ?? "step"}-${index}`}>
                <strong>{item.label || item.tool || `Step ${index + 1}`}</strong>
                {item.tool ? <span> · {item.tool}</span> : null}
                {item.purpose ? <p>{item.purpose}</p> : null}
              </li>
            ))}
          </ol>
        </div>
      ) : null}
      {limitations.length > 0 ? (
        <div className="analysis-proposal-limitations">
          <span className="mini-label">Limitations</span>
          <ul>
            {limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      <p className="analysis-proposal-approval">{proposal.approval_prompt}</p>
      <div className="analysis-proposal-actions">
        {canRunProposal ? (
          <button className="result-expand-btn primary" onClick={onRun} type="button">
            Run proposed analysis
          </button>
        ) : null}
        <button className="result-expand-btn" onClick={onRevise} type="button">
          Revise plan
        </button>
      </div>
    </div>
  );
}

function PolicyGuidanceSection({
  guidance,
  integratedAssessment,
  synthesisWarnings,
}: {
  guidance?: PolicyGuidance;
  integratedAssessment?: IntegratedAssessment;
  synthesisWarnings?: string[];
}) {
  const matrix = Array.isArray(guidance?.evidence_action_matrix)
    ? guidance.evidence_action_matrix.filter(isPolicyGuidanceAction)
    : [];
  const placeBrief = typeof guidance?.place_based_policy_brief === "string"
    ? guidance.place_based_policy_brief.trim()
    : "";
  const limits = Array.isArray(guidance?.evidence_limits)
    ? guidance.evidence_limits.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    : [];
  const managementGuidance = cleanStringOrList(integratedAssessment?.management_guidance);
  const uncertaintyAndGaps = cleanStringList(integratedAssessment?.uncertainty_and_data_gaps);
  const supportNotes = uniqueStrings([
    ...cleanStringList(integratedAssessment?.evidence_boundary_notes),
    ...cleanStringList(synthesisWarnings),
  ]);
  const economicImplications = cleanString(integratedAssessment?.economic_implications);
  const protectionImplications = cleanString(integratedAssessment?.environmental_protection_implications);
  const policySynthesis = integratedAssessment?.policy_synthesis;
  const oneSentenceJudgment = cleanString(policySynthesis?.one_sentence_judgment);
  const policyNarrative = cleanString(policySynthesis?.policy_narrative);
  const directAnswer = cleanString(integratedAssessment?.direct_answer) || oneSentenceJudgment;
  const assessmentNarrative = cleanString(integratedAssessment?.assessment_narrative) || policyNarrative;
  const evidenceThreads = Array.isArray(integratedAssessment?.evidence_threads)
    ? integratedAssessment.evidence_threads.filter(isIntegratedEvidenceThread)
    : [];
  const higherRiskRegions = Array.isArray(integratedAssessment?.higher_risk_regions)
    ? integratedAssessment.higher_risk_regions.filter(isIntegratedHigherRiskRegion)
    : [];
  const policyCards = evidenceLinkedPolicyCardsForDisplay(policySynthesis);
  const guidanceAllowed = guidance?.should_include !== false;
  const hasPolicyGuidance = Boolean(
    guidanceAllowed
      && guidance
      && (
        placeBrief
        || matrix.length > 0
        || limits.length > 0
        || (typeof guidance.headline === "string" && guidance.headline.trim().length > 0)
      ),
  );
  const hasIntegratedPolicyContent = Boolean(
    managementGuidance.length > 0
      || uncertaintyAndGaps.length > 0
      || economicImplications
      || protectionImplications
      || directAnswer
      || assessmentNarrative
      || higherRiskRegions.length > 0
      || evidenceThreads.length > 0
      || policyCards.length > 0,
  );

  if (!hasPolicyGuidance && !hasIntegratedPolicyContent) {
    return null;
  }

  return (
    <div className="policy-guidance-section ui-card">
      <h4 className="ui-card-title">{integratedAssessment ? "Integrated Assessment" : "Policy / Management Guidance"}</h4>
      {guidance?.headline ? <p className="policy-guidance-headline">{guidance.headline}</p> : null}
      {placeBrief ? <p className="policy-guidance-brief">{placeBrief}</p> : null}
      {directAnswer ? (
        <div className="policy-guidance-subsection">
          <span className="policy-guidance-subtitle">Overall judgment</span>
          <p className="policy-guidance-brief">{directAnswer}</p>
        </div>
      ) : null}
      {assessmentNarrative ? (
        <div className="policy-guidance-subsection">
          <span className="policy-guidance-subtitle">Integrated narrative</span>
          {splitParagraphs(assessmentNarrative).map((paragraph) => (
            <p className="policy-guidance-brief" key={paragraph}>{paragraph}</p>
          ))}
        </div>
      ) : null}
      {higherRiskRegions.length > 0 ? (
        <div className="policy-guidance-subsection">
          <span className="policy-guidance-subtitle">Higher-risk coastal regions</span>
          <div className="policy-guidance-table-wrap">
            <table className="policy-guidance-table">
              <thead>
                <tr>
                  <th>Region</th>
                  <th>Major environmental risks</th>
                  <th>Evidence</th>
                </tr>
              </thead>
              <tbody>
                {higherRiskRegions.map((item, index) => (
                  <tr key={`${item.region}-${index}`}>
                    <td>{item.region}</td>
                    <td>{item.major_environmental_risks}</td>
                    <td>{formatHigherRiskRegionEvidence(item)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
      {evidenceThreads.length > 0 ? (
        <div className="policy-guidance-subsection">
          <span className="policy-guidance-subtitle">Evidence threads</span>
          <div className="policy-guidance-list">
            {evidenceThreads.map((item, index) => (
              <div className="policy-guidance-item" key={`${item.theme}-${index}`}>
                <div className="policy-guidance-line">
                  <span className="policy-guidance-tag">{formatEvidenceThreadTheme(item.theme)}</span>
                  <span className="policy-guidance-tag">{formatEvidenceStatus(item.status)}</span>
                </div>
                <p className="policy-guidance-rationale">
                  {formatEvidenceThreadText(item)}
                </p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
      {policyCards.length > 0 ? (
        <div className="policy-guidance-subsection">
          <span className="policy-guidance-subtitle">Policy recommendations</span>
          <div className="policy-guidance-list">
            {policyCards.map((item, index) => (
              <div className="policy-guidance-item" key={`${item.title}-${index}`}>
                <div className="policy-guidance-line">
                  <span className="policy-guidance-tag">Policy {index + 1}</span>
                  {item.evidenceStatus ? (
                    <span className="policy-guidance-tag">{formatEvidenceStatus(item.evidenceStatus)}</span>
                  ) : null}
                </div>
                <p className="policy-guidance-rationale">{item.text}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
      {economicImplications || protectionImplications ? (
        <div className="policy-guidance-subsection">
          <span className="policy-guidance-subtitle">Implications</span>
          <div className="policy-guidance-grid">
            {economicImplications ? (
              <div className="policy-guidance-block">
                <span className="policy-guidance-kicker">Economic-development</span>
                <p>{economicImplications}</p>
              </div>
            ) : null}
            {protectionImplications ? (
              <div className="policy-guidance-block">
                <span className="policy-guidance-kicker">Environmental-protection</span>
                <p>{protectionImplications}</p>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
      {managementGuidance ? (
        <div className="policy-guidance-subsection">
          <span className="policy-guidance-subtitle">Management summary</span>
          <p className="policy-guidance-brief">{managementGuidance}</p>
        </div>
      ) : null}
      {matrix.length > 0 ? (
        <div className="policy-guidance-list">
          {matrix.map((item, index) => (
            <div className="policy-guidance-item" key={`${item.action_type}-${item.target}-${index}`}>
              <div className="policy-guidance-line">
                <span className="policy-guidance-tag">
                  {formatPolicyLabel(item.priority)} / {formatPolicyLabel(item.action_type)}
                </span>
              </div>
              <p className="policy-guidance-rationale">{formatLegacyPolicyAction(item)}</p>
            </div>
          ))}
        </div>
      ) : null}
      {limits.length > 0 || uncertaintyAndGaps.length > 0 ? (
        <ul className="policy-guidance-limits">
          {[...limits, ...uncertaintyAndGaps].map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : null}
      {supportNotes.length > 0 ? (
        <div className="policy-guidance-subsection">
          <span className="policy-guidance-subtitle">Data support notes</span>
          <ul className="policy-guidance-limits">
            {supportNotes.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function cleanString(value: unknown) {
  return typeof value === "string" ? value.trim() : "";
}

function cleanStringOrList(value: unknown) {
  if (typeof value === "string") {
    return value.trim();
  }
  if (Array.isArray(value)) {
    return value
      .filter((item): item is string => typeof item === "string" && item.trim().length > 0)
      .map((item) => item.trim())
      .join(" ");
  }
  return "";
}

function cleanStringList(value: unknown) {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && item.trim().length > 0).map((item) => item.trim())
    : [];
}

function uniqueStrings(values: string[]) {
  const seen = new Set<string>();
  return values.filter((item) => {
    const key = item.toLowerCase();
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function splitParagraphs(value: string) {
  const paragraphs = value
    .split(/\n{2,}/)
    .map((item) => item.trim())
    .filter(Boolean);
  return paragraphs.length > 0 ? paragraphs : [value];
}

function formatEvidenceStatus(value: "computed" | "indirect" | "data_gap") {
  return value === "data_gap" ? "data gap" : value;
}

type IntegratedEvidenceThreadItem = NonNullable<IntegratedAssessment["evidence_threads"]>[number];
type IntegratedHigherRiskRegionItem = {
  region: string;
  major_environmental_risks: string;
  evidence: string;
  evidence_result_ids?: string[];
};

function formatEvidenceThreadTheme(value: IntegratedEvidenceThreadItem["theme"]) {
  return value.replace("/", " / ").replace(/_/g, " ");
}

function formatEvidenceThreadText(item: IntegratedEvidenceThreadItem) {
  const summary = item.evidence_summary.trim().replace(/[.;]\s*$/, "");
  const ids = cleanStringList(item.evidence_result_ids);
  if (ids.length === 0) {
    return `${summary}.`;
  }
  return `${summary}. Supporting tool results: ${ids.join(", ")}.`;
}

function formatHigherRiskRegionEvidence(item: IntegratedHigherRiskRegionItem) {
  const evidence = (item.evidence ?? "").trim().replace(/[.;]\s*$/, "");
  const ids = cleanStringList(item.evidence_result_ids);
  if (ids.length === 0) {
    return `${evidence}.`;
  }
  const evidenceLower = evidence.toLowerCase();
  const missingIds = ids.filter((id) => !evidenceLower.includes(id.toLowerCase()));
  if (missingIds.length === 0) {
    return `${evidence}.`;
  }
  return `${evidence}. Results: ${missingIds.join(", ")}.`;
}

function formatLegacyPolicyAction(item: PolicyGuidanceAction) {
  const parts = [
    item.recommendation.trim().replace(/[.;]\s*$/, ""),
    `Focus this first on ${item.where_when.trim().replace(/[.;]\s*$/, "")}`,
    `This is grounded in ${item.evidence_basis.trim().replace(/[.;]\s*$/, "")}`,
    `The evidence strength is ${formatPolicyLabel(item.evidence_strength)}`,
  ];
  return `${parts.join(". ")}.`;
}

function isIntegratedEvidenceThread(value: unknown): value is IntegratedEvidenceThreadItem {
  return Boolean(
    value
      && typeof value === "object"
      && typeof (value as IntegratedEvidenceThreadItem).theme === "string"
      && typeof (value as IntegratedEvidenceThreadItem).status === "string"
      && typeof (value as IntegratedEvidenceThreadItem).evidence_summary === "string",
  );
}

function isIntegratedHigherRiskRegion(value: unknown): value is IntegratedHigherRiskRegionItem {
  return Boolean(
    value
      && typeof value === "object"
      && typeof (value as IntegratedHigherRiskRegionItem).region === "string"
      && typeof (value as IntegratedHigherRiskRegionItem).major_environmental_risks === "string"
      && typeof (value as IntegratedHigherRiskRegionItem).evidence === "string",
  );
}

function isPolicyGuidanceAction(value: unknown): value is PolicyGuidanceAction {
  return Boolean(
    value
      && typeof value === "object"
      && typeof (value as PolicyGuidanceAction).target === "string"
      && typeof (value as PolicyGuidanceAction).recommendation === "string"
      && typeof (value as PolicyGuidanceAction).evidence_basis === "string"
      && typeof (value as PolicyGuidanceAction).guardrail === "string",
  );
}

export function AnalysisFeed({
  conversationId,
  canExportReport = false,
  isExportingReport = false,
  messages,
  isBusy = false,
  isLocked = false,
  onExportReport,
  queryText,
  onOpenDetail,
  onPromoteMapField,
  onResultAction,
  onQueryChange,
  onSubmitQuery,
  onToggleStepCard,
}: AnalysisFeedProps) {
  const threadRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const thread = threadRef.current;
    if (!thread) {
      return;
    }
    thread.scrollTo({
      top: thread.scrollHeight,
      behavior: "smooth",
    });
  }, [messages.length]);

  return (
    <div className="analysis-feed">
      <div className="feed-toolbar">
        <button
          className="result-expand-btn"
          disabled={!canExportReport || isLocked}
          onClick={() => onExportReport?.()}
          type="button"
        >
          {isExportingReport ? "Exporting notebook..." : "Export notebook"}
        </button>
      </div>
      <div className="feed-thread" ref={threadRef}>
        {messages.map((message) =>
          message.role === "user" ? (
            <div key={message.id} className="user-message">
              <p>{message.text}</p>
            </div>
          ) : (
            <AssistantBlock
              conversationId={conversationId}
              key={message.id}
              message={message}
              onOpenDetail={onOpenDetail}
              onPromoteMapField={onPromoteMapField}
              onResultAction={onResultAction}
              onQueryChange={onQueryChange}
              onSubmitQuery={onSubmitQuery}
              onToggleStepCard={onToggleStepCard}
            />
          )
        )}
      </div>

      <div className="feed-input">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            onSubmitQuery();
          }}
        >
          <input
            disabled={isLocked}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Ask for analysis, trends, or visualizations..."
            value={queryText}
          />
          <button disabled={isLocked} type="submit">
            {isBusy ? "Running..." : "Send"}
          </button>
        </form>
      </div>
    </div>
  );
}
