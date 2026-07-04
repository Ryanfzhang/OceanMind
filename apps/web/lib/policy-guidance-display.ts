import type {
  PolicySynthesis,
  PolicySynthesisDecisionRow,
  PolicySynthesisRecommendation,
} from "./types";

export type PolicyActionGroupKey = NonNullable<PolicySynthesisDecisionRow["action_group"]>;
export type PolicyActionDisplayGroupKey = PolicyActionGroupKey | "other";

export const POLICY_ACTION_GROUP_ORDER: PolicyActionDisplayGroupKey[] = [
  "spatial_priority",
  "oxygen_response",
  "seasonal_operations",
  "driver_adaptation",
  "source_pathway_screening",
  "economic_data_assessment",
  "validation_gap",
  "other",
];

export const POLICY_ACTION_GROUP_LABELS: Record<PolicyActionDisplayGroupKey, string> = {
  spatial_priority: "Spatial priorities",
  oxygen_response: "Oxygen response",
  seasonal_operations: "Seasonal operations",
  driver_adaptation: "Driver adaptation",
  source_pathway_screening: "Source/pathway screening",
  economic_data_assessment: "Economic/data assessment",
  validation_gap: "Validation gaps",
  other: "Other evidence-bounded actions",
};

const POLICY_ACTION_GROUP_KEYS = new Set<PolicyActionDisplayGroupKey>(POLICY_ACTION_GROUP_ORDER);

export type GroupedPolicyDecisionRows = {
  group: PolicyActionDisplayGroupKey;
  label: string;
  rows: PolicySynthesisDecisionRow[];
};

export type EvidenceLinkedPolicyCard = {
  title: string;
  action: string;
  priorityPlaces: string[];
  evidenceStatus?: "computed" | "indirect" | "data_gap";
  evidenceNote?: string;
  text: string;
  evidenceResultIds: string[];
};

export function normalizePolicyActionGroup(value: unknown): PolicyActionDisplayGroupKey {
  if (typeof value !== "string") {
    return "other";
  }
  const normalized = value.trim() as PolicyActionDisplayGroupKey;
  return POLICY_ACTION_GROUP_KEYS.has(normalized) && normalized !== "other" ? normalized : "other";
}

export function groupPolicyDecisionRowsForDisplay(
  rows: PolicySynthesisDecisionRow[],
): GroupedPolicyDecisionRows[] {
  const grouped = new Map<PolicyActionDisplayGroupKey, PolicySynthesisDecisionRow[]>();
  for (const row of rows) {
    const group = normalizePolicyActionGroup(row.action_group);
    const existing = grouped.get(group);
    if (existing) {
      existing.push(row);
    } else {
      grouped.set(group, [row]);
    }
  }

  return POLICY_ACTION_GROUP_ORDER
    .map((group) => ({
      group,
      label: POLICY_ACTION_GROUP_LABELS[group],
      rows: grouped.get(group) ?? [],
    }))
    .filter((item) => item.rows.length > 0);
}

export function evidenceLinkedPolicyCardsForDisplay(
  policySynthesis?: PolicySynthesis,
): EvidenceLinkedPolicyCard[] {
  const recommendations = Array.isArray(policySynthesis?.policy_recommendations)
    ? policySynthesis.policy_recommendations.filter(isPolicySynthesisRecommendation)
    : [];
  if (recommendations.length > 0) {
    return recommendations.map((item) => {
      const priorityPlaces = cleanStringList(item.priority_places);
      const evidenceNote = cleanString(item.evidence_note)
        ?? cleanString(cleanStringList(item.supporting_evidence).join("; "))
        ?? cleanString(item.why_this_policy);
      const evidenceStatus = normalizeEvidenceStatus(item.evidence_status);
      return {
        title: item.policy_title.trim(),
        action: item.recommended_action.trim(),
        priorityPlaces,
        evidenceStatus,
        evidenceNote,
        text: buildNaturalPolicyText({
          title: item.policy_title,
          action: item.recommended_action,
          priorityPlaces,
          evidenceNote,
          evidenceStatus,
          evidenceResultIds: cleanStringList(item.evidence_result_ids),
        }),
        evidenceResultIds: cleanStringList(item.evidence_result_ids),
      };
    });
  }

  const rows = Array.isArray(policySynthesis?.decision_rows)
    ? policySynthesis.decision_rows.filter(isPolicySynthesisDecisionRow)
    : [];
  return rows.map((item) => ({
    title: item.decision_unit.trim(),
    action: item.recommended_action.trim(),
    priorityPlaces: cleanStringList([item.where_when, item.target]),
    evidenceStatus: legacyConfidenceToEvidenceStatus(item.confidence),
    evidenceNote: cleanString(item.trigger_evidence) ?? cleanString(item.evidence_basis),
    text: buildNaturalPolicyText({
      title: item.decision_unit,
      action: item.recommended_action,
      priorityPlaces: cleanStringList([item.where_when, item.target]),
      evidenceNote: cleanString(item.trigger_evidence) ?? cleanString(item.evidence_basis),
      evidenceStatus: legacyConfidenceToEvidenceStatus(item.confidence),
      evidenceResultIds: cleanStringList(item.evidence_result_ids),
    }),
    evidenceResultIds: cleanStringList(item.evidence_result_ids),
  }));
}

function buildNaturalPolicyText({
  title,
  action,
  priorityPlaces,
  evidenceNote,
  evidenceStatus,
  evidenceResultIds,
}: {
  title?: string;
  action: string;
  priorityPlaces: string[];
  evidenceNote?: string;
  evidenceStatus?: EvidenceLinkedPolicyCard["evidenceStatus"];
  evidenceResultIds?: string[];
}) {
  const cleanTitle = cleanString(title);
  const cleanAction = action.trim().replace(/[.;]\s*$/, "");
  const parts = [cleanTitle ? `${cleanTitle}: ${cleanAction}` : cleanAction];
  if (priorityPlaces.length > 0) {
    parts.push(`Focus this first on ${priorityPlaces.join("; ")}`);
  }
  if (evidenceNote) {
    const prefix = evidenceStatus === "data_gap"
      ? "Current data are incomplete: "
      : evidenceStatus === "indirect"
        ? "The current evidence supports this indirectly: "
        : "The analysis directly supports this: ";
    parts.push(`${prefix}${evidenceNote.replace(/[.;]\s*$/, "")}`);
  }
  if (evidenceResultIds && evidenceResultIds.length > 0) {
    parts.push(`The supporting tool results are ${evidenceResultIds.join(", ")}`);
  }
  return `${parts.join(". ")}.`;
}

function isPolicySynthesisRecommendation(value: unknown): value is PolicySynthesisRecommendation {
  return Boolean(
    value
      && typeof value === "object"
      && typeof (value as PolicySynthesisRecommendation).policy_title === "string"
      && typeof (value as PolicySynthesisRecommendation).recommended_action === "string",
  );
}

function isPolicySynthesisDecisionRow(value: unknown): value is PolicySynthesisDecisionRow {
  return Boolean(
    value
      && typeof value === "object"
      && typeof (value as PolicySynthesisDecisionRow).decision_unit === "string"
      && typeof (value as PolicySynthesisDecisionRow).recommended_action === "string",
  );
}

function cleanString(value: unknown) {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function normalizeEvidenceStatus(value: unknown): EvidenceLinkedPolicyCard["evidenceStatus"] {
  return value === "computed" || value === "indirect" || value === "data_gap" ? value : undefined;
}

function legacyConfidenceToEvidenceStatus(value: unknown): EvidenceLinkedPolicyCard["evidenceStatus"] {
  if (value === "supported") {
    return "computed";
  }
  if (value === "limited" || value === "screening") {
    return "indirect";
  }
  if (value === "not_supported") {
    return "data_gap";
  }
  return undefined;
}

function cleanStringList(value: unknown) {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && item.trim().length > 0).map((item) => item.trim())
    : [];
}
