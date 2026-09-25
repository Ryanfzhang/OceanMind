"use client";

import { useEffect, useState } from "react";

type Attempt = { attempt_id: string; status: string; code_version?: string | null };
type Entry = { artifact_id?: string | null; stage: string; time?: string | null;
  depth?: string | number | null; status: string; summary: string };
type Page = { total: number; offset: number; results: Entry[]; next_offset: number | null };

function url(conversation: string, mode: string, extra: Record<string, string> = {}) {
  const params = new URLSearchParams({ conversation, mode, ...extra });
  return `/api/results?${params.toString()}`;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Result request failed (${response.status})`);
  return response.json() as Promise<T>;
}

export function ResultIndexBrowser({ conversationId }: { conversationId: string }) {
  const [open, setOpen] = useState(false);
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [attemptId, setAttemptId] = useState("");
  const [page, setPage] = useState<Page | null>(null);
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    getJson<{ attempts: Attempt[] }>(url(conversationId, "attempts"))
      .then((value) => {
        if (cancelled) return;
        setAttempts(value.attempts);
        setAttemptId(value.attempts.at(-1)?.attempt_id ?? "");
      })
      .catch((cause) => !cancelled && setError(String(cause)));
    return () => { cancelled = true; };
  }, [open, conversationId]);

  useEffect(() => {
    if (!open || !attemptId) return;
    let cancelled = false;
    setDetail(null);
    getJson<Page>(url(conversationId, "page", { attempt: attemptId, offset: "0" }))
      .then((value) => !cancelled && setPage(value))
      .catch((cause) => !cancelled && setError(String(cause)));
    return () => { cancelled = true; };
  }, [open, conversationId, attemptId]);

  async function loadPage(offset: number) {
    try {
      setError("");
      setPage(await getJson<Page>(url(conversationId, "page", { attempt: attemptId, offset: String(offset) })));
      setDetail(null);
    } catch (cause) { setError(String(cause)); }
  }

  async function inspect(entry: Entry) {
    if (!entry.artifact_id) return;
    try {
      setError("");
      setDetail(await getJson<Record<string, unknown>>(url(conversationId, "artifact", { artifact: entry.artifact_id })));
    } catch (cause) { setError(String(cause)); }
  }

  return (
    <>
      <button className="result-expand-btn" type="button" onClick={() => setOpen(true)}>Browse all results</button>
      {open ? (
        <div role="dialog" aria-modal="true" aria-label="Saved result index" className="result-index-dialog">
          <div className="result-index-panel ui-card">
            <div className="result-inline-header">
              <h3>Saved result index</h3>
              <button type="button" onClick={() => setOpen(false)}>Close</button>
            </div>
            <label>
              Analysis attempt
              <select value={attemptId} onChange={(event) => setAttemptId(event.target.value)}>
                {attempts.map((attempt) => (
                  <option key={attempt.attempt_id} value={attempt.attempt_id}>
                    {attempt.status} · {attempt.attempt_id.slice(-12)}
                  </option>
                ))}
              </select>
            </label>
            {error ? <p role="alert">{error}</p> : null}
            {page ? (
              <>
                <p>{page.total} saved entries · showing {page.offset + 1}–{page.offset + page.results.length}</p>
                <div className="result-index-list">
                  {page.results.map((entry, index) => (
                    <button key={`${entry.artifact_id ?? "failed"}-${page.offset + index}`}
                      type="button" onClick={() => inspect(entry)} disabled={!entry.artifact_id}>
                      {page.offset + index + 1}. {entry.stage} · {entry.time ?? "time unspecified"}
                      {entry.depth != null ? ` · depth ${entry.depth}` : ""} · {entry.status}
                    </button>
                  ))}
                </div>
                <div>
                  <button type="button" disabled={page.offset === 0}
                    onClick={() => loadPage(Math.max(0, page.offset - 20))}>Previous</button>
                  <button type="button" disabled={page.next_offset == null}
                    onClick={() => page.next_offset != null && loadPage(page.next_offset)}>Next</button>
                </div>
              </>
            ) : null}
            {detail ? (
              <div>
                <h4>Result {String(detail.artifact_id ?? "")}</h4>
                {detail.kind === "image_png" ? (
                  // Image URL is an opaque run-owned reference served by the backend.
                  <img alt="Saved analysis figure" style={{ maxWidth: "100%" }}
                    src={url(conversationId, "image", { artifact: String(detail.artifact_id) })} />
                ) : <pre>{String(detail.content ?? JSON.stringify(detail.summary ?? {}, null, 2))}</pre>}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </>
  );
}
