import type { AssistantMessagePayload } from "./types";

export function shouldShowExecutionProgress(payload: Partial<AssistantMessagePayload> | undefined) {
  if (!payload) {
    return false;
  }
  return payload.state === "planning" || payload.state === "running";
}
