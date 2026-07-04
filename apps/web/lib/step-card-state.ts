import type { PlanStep, ResultCardSummary, StepCard, StepProgress } from "./types";

function rankStatus(status: StepCard["status"]) {
  return {
    pending: 0,
    running: 1,
    completed: 2,
    failed: 3,
  }[status];
}

function mergeStatus(existing: StepCard["status"], incoming: StepCard["status"]) {
  if (incoming === "completed" || existing === "completed") {
    return "completed";
  }
  if (incoming === "failed") {
    return "failed";
  }
  if (existing === "failed") {
    return incoming;
  }
  return rankStatus(incoming) >= rankStatus(existing) ? incoming : existing;
}

function mergeStepCard(existing: StepCard, incoming: StepCard): StepCard {
  const nextStatus = mergeStatus(existing.status, incoming.status);
  const isRecoveryFromFailure = existing.status === "failed" && incoming.status !== "failed";
  const progress = nextStatus === "completed"
    ? undefined
    : isRecoveryFromFailure
      ? incoming.progress
      : mergeStepProgress(existing.progress, incoming.progress);
  const error = nextStatus === "failed" ? incoming.error ?? existing.error : undefined;

  return {
    ...existing,
    ...incoming,
    human_label: incoming.human_label || existing.human_label,
    technical_label: incoming.technical_label || existing.technical_label,
    status: nextStatus,
    results: incoming.results.length > 0 ? incoming.results : existing.results,
    actions: incoming.actions.length > 0 ? incoming.actions : existing.actions,
    interpretation: incoming.interpretation?.trim() ? incoming.interpretation : existing.interpretation,
    is_map_bound: existing.is_map_bound || incoming.is_map_bound,
    is_expanded: existing.is_expanded || incoming.is_expanded,
    progress,
    error,
  };
}

function mergeStepProgress(existing?: StepProgress, incoming?: StepProgress): StepProgress | undefined {
  if (!incoming) {
    return existing;
  }
  if (!existing) {
    return incoming;
  }
  const next = { ...incoming };
  if (
    typeof existing.percent === "number" &&
    Number.isFinite(existing.percent) &&
    typeof incoming.percent === "number" &&
    Number.isFinite(incoming.percent) &&
    incoming.percent < existing.percent
  ) {
    next.percent = existing.percent;
  }
  return next;
}

export function mergeOrAppendStepCard(cards: StepCard[], incoming: StepCard): StepCard[] {
  const index = cards.findIndex((card) => card.step_id === incoming.step_id);
  if (index === -1) {
    return [...cards, incoming];
  }
  const next = [...cards];
  next[index] = mergeStepCard(next[index], incoming);
  return next;
}

export function mergeStepCardLists(existingCards: StepCard[], incomingCards: StepCard[]): StepCard[] {
  return incomingCards.reduce(
    (merged, incoming) => mergeOrAppendStepCard(merged, incoming),
    existingCards,
  );
}

export function resolveMapResult(stepCard: StepCard): ResultCardSummary | null {
  return stepCard.results.find((result) => result.surface === "map") ?? null;
}

export function getProgressCounts(planSteps: PlanStep[], stepCards: StepCard[]) {
  return {
    totalSteps: Math.max(planSteps.length, stepCards.length),
    completedSteps: stepCards.filter((step) => step.status === "completed").length,
  };
}

function pluralizeUnit(label: string, count: number) {
  if (count === 1) {
    return label;
  }
  if (label.endsWith("s")) {
    return label;
  }
  return `${label}s`;
}

export function formatStepProgressText(progress?: StepProgress, chinese = false) {
  if (!progress) {
    return "";
  }
  const phaseLabels: Record<string, { en: string; zh: string }> = {
    resolving_sources: { en: "Resolving data source", zh: "正在定位数据源" },
    opening_source: { en: "Opening data source", zh: "正在打开数据源" },
    subset_prepared: { en: "Preparing data subset", zh: "正在准备数据子集" },
    complete: { en: "Data ready", zh: "数据已准备" },
    preparing_compute: { en: "Preparing compute", zh: "正在准备计算" },
    lazy_result_prepared: { en: "Lazy result ready", zh: "惰性结果已准备" },
    compute_graph_prepared: { en: "Preparing compute graph", zh: "正在准备计算图" },
    computing: { en: "Computing", zh: "正在计算" },
    compute_complete: { en: "Compute complete", zh: "计算完成" },
    building_mask: { en: "Building mask", zh: "正在构建掩膜" },
    solving_streamfunction: { en: "Solving streamfunction", zh: "正在反演流函数" },
    applying_regional_gauge: { en: "Applying regional gauge", zh: "正在应用区域定标" },
    preparing_map_payload: { en: "Preparing map payload", zh: "正在准备地图数据" },
    compute_failed: { en: "Compute failed", zh: "计算失败" },
    partition_started: { en: "Computing partition", zh: "正在计算分区" },
    partition_complete: { en: "Partition complete", zh: "分区完成" },
    reflection: { en: "Updating workflow", zh: "正在更新工作流" },
  };
  const phase = phaseLabels[progress.phase];
  const phaseLabel = phase ? (chinese ? phase.zh : phase.en) : "";
  const unitLabel =
    typeof progress.completed_units === "number" &&
    typeof progress.total_units === "number" &&
    progress.total_units > 0
      ? `${progress.completed_units}/${progress.total_units} ${pluralizeUnit(progress.unit_label ?? "data source", progress.total_units)}`
      : "";
  const currentUnit = formatVisibleCurrentUnit(progress);
  return [phaseLabel, progress.message, unitLabel, currentUnit].filter(Boolean).join(" · ");
}

function formatVisibleCurrentUnit(progress: StepProgress) {
  const currentUnit = progress.current_unit?.trim();
  if (!currentUnit) {
    return "";
  }
  if (progress.unit_label?.toLowerCase() === "task") {
    return "";
  }
  return currentUnit.length > 80 ? `${currentUnit.slice(0, 77)}...` : currentUnit;
}

export function shouldShowStepFallbackInterpretation(step: StepCard): boolean {
  if (!step.interpretation?.trim()) {
    return false;
  }
  return !step.results.some((result) => Boolean(result.interpretation?.trim()));
}

export function toggleStepCards(stepCards: StepCard[], stepId: string) {
  const targetStep = stepCards.find((step) => step.step_id === stepId);
  if (!targetStep) {
    return {
      nextStepCards: stepCards,
      nextMapCard: null as ResultCardSummary | null,
      clearMap: false,
    };
  }

  const targetIsMapBound = Boolean(targetStep.is_map_bound);
  const willExpand = !targetStep.is_expanded;

  const nextStepCards = stepCards.map((step) => {
    if (step.step_id === stepId) {
      return { ...step, is_expanded: !step.is_expanded };
    }

    if (targetIsMapBound && step.is_map_bound) {
      return { ...step, is_expanded: false };
    }

    return step;
  });

  return {
    nextStepCards,
    nextMapCard: targetIsMapBound && willExpand ? resolveMapResult(targetStep) : null,
    clearMap: targetIsMapBound && !willExpand,
  };
}
