import type { AssistantMessagePayload, ChatMessage } from "./types";

export function collectSynthesisEvidenceFromPayload(payload: Pick<AssistantMessagePayload, "findings"> | undefined): string[] {
  const findings = payload?.findings ?? [];
  const seen = new Set<string>();
  const evidence: string[] = [];

  for (const finding of findings) {
    for (const item of finding.evidence) {
      const normalized = item.trim();
      if (!normalized || seen.has(normalized)) {
        continue;
      }
      seen.add(normalized);
      evidence.push(normalized);
      if (evidence.length >= 4) {
        return evidence;
      }
    }
  }

  return evidence;
}

export function buildCompletedSummaryContent(message: ChatMessage) {
  return {
    summary: message.payload?.summary ?? message.text,
    evidence: collectSynthesisEvidenceFromPayload(message.payload),
    policyGuidance: message.payload?.policyGuidance,
    integratedAssessment: message.payload?.integratedAssessment,
    synthesisWarnings: message.payload?.synthesisWarnings ?? [],
  };
}
