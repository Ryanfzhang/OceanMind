import type { ConversationReportPayload } from "./report-export";
import type { DatasetInfo, VisualizationApiResponse } from "./types";

export type QueryStreamEnvelope = {
  event: "execution_event" | "final" | "error";
  payload: Record<string, unknown>;
};

export function isLikelyStreamTransportError(error: unknown) {
  if (error instanceof TypeError) {
    return true;
  }

  const message = error instanceof Error ? error.message : String(error ?? "");
  const normalized = message.toLowerCase();
  return [
    "network",
    "fetch",
    "terminated",
    "aborted",
    "abort",
    "econnreset",
    "socket",
    "body stream",
    "incomplete ndjson",
  ].some((pattern) => normalized.includes(pattern));
}

async function yieldToUi() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

export async function consumeNdjsonStream(
  stream: ReadableStream<Uint8Array>,
  onEvent: (event: QueryStreamEnvelope) => void | Promise<void>
) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      let parsed: QueryStreamEnvelope;
      try {
        parsed = JSON.parse(trimmed) as QueryStreamEnvelope;
      } catch {
        buffer = `${trimmed}\n${buffer}`;
        break;
      }
      await onEvent(parsed);
      await yieldToUi();
    }
  }

  const trailing = buffer.trim();
  if (trailing) {
    try {
      await onEvent(JSON.parse(trailing) as QueryStreamEnvelope);
    } catch (error) {
      if (error instanceof SyntaxError) {
        throw new Error("Stream ended with an incomplete NDJSON event.");
      }
      throw error;
    }
  }
}

async function parseError(response: Response) {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail ?? `Request failed with status ${response.status}`;
  } catch {
    return `Request failed with status ${response.status}`;
  }
}

export async function pingApi() {
  const response = await fetch("/api/health", {
    method: "GET",
    cache: "no-store"
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return (await response.json()) as { status: string };
}

export async function getActiveDataset() {
  const response = await fetch("/api/dataset", {
    method: "GET",
    cache: "no-store"
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return (await response.json()) as DatasetInfo;
}

export async function runWorkspaceQueryStream(
  payload: {
    query: string;
    conversation_id?: string;
    continue_pending?: boolean;
    extracted_params?: Record<string, unknown>;
    additional_context?: Record<string, unknown>;
    synthesize?: boolean;
    trust_env?: boolean;
  },
  onEvent: (event: QueryStreamEnvelope) => void | Promise<void>
) {
  const response = await fetch("/api/query/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      query: payload.query,
      conversation_id: payload.conversation_id,
      continue_pending: payload.continue_pending ?? false,
      extracted_params: payload.extracted_params ?? {},
      additional_context: payload.additional_context ?? {},
      synthesize: payload.synthesize ?? true,
      trust_env: payload.trust_env ?? false
    })
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  if (!response.body) {
    throw new Error("Streaming response body is empty.");
  }

  await consumeNdjsonStream(response.body, onEvent);
}

export async function runDirectVisualization(payload: {
  dataset: string;
  variable: string;
  time_range: [string, string];
  region: {
    lon_min: number;
    lon_max: number;
    lat_min: number;
    lat_max: number;
  };
  depth_mode: "fixed" | "feature" | "layer_mean";
  depth_range?: [number, number];
  feature?: "mixed_layer" | "thermocline" | "pycnocline";
  layer_mean_label?: string;
  selected_point?: {
    lat: number;
    lon: number;
  };
}) {
  const response = await fetch("/api/visualize", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return (await response.json()) as VisualizationApiResponse;
}

export async function exportConversationReport(payload: ConversationReportPayload) {
  const response = await fetch("/api/report/export", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  const contentDisposition = response.headers.get("Content-Disposition") ?? "";
  const filenameMatch = /filename=\"?([^\";]+)\"?/i.exec(contentDisposition);

  return {
    blob: await response.blob(),
    filename: filenameMatch?.[1] ?? null,
  };
}
