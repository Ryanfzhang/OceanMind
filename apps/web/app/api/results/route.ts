import { NextRequest } from "next/server";
import { getBackendApiBaseUrl } from "@/lib/server-api";

// IDs are opaque server-generated tokens. This proxy accepts no filesystem paths.
export async function GET(request: NextRequest) {
  const query = request.nextUrl.searchParams;
  const conversation = query.get("conversation");
  const attempt = query.get("attempt");
  const artifact = query.get("artifact");
  const mode = query.get("mode") ?? "attempts";
  if (!conversation || !/^conv_[0-9a-f]{32}$/.test(conversation)) {
    return Response.json({ detail: "Invalid conversation ID" }, { status: 400 });
  }
  if (attempt && !/^[a-z0-9_]+$/.test(attempt)) {
    return Response.json({ detail: "Invalid attempt ID" }, { status: 400 });
  }
  if (artifact && !/^[a-z0-9_]+$/.test(artifact)) {
    return Response.json({ detail: "Invalid artifact ID" }, { status: 400 });
  }
  const base = `/results/${conversation}`;
  let path: string;
  if (mode === "attempts") {
    path = `${base}/attempts`;
  } else if (mode === "page" && attempt) {
    const offset = Number(query.get("offset") ?? 0);
    if (!Number.isInteger(offset) || offset < 0) {
      return Response.json({ detail: "Invalid offset" }, { status: 400 });
    }
    path = `${base}/attempts/${attempt}?offset=${offset}&limit=20`;
  } else if (mode === "artifact" && artifact) {
    path = `${base}/artifacts/${artifact}`;
  } else if (mode === "image" && artifact) {
    path = `${base}/images/${artifact}`;
  } else if (mode === "code" && artifact) {
    path = `${base}/code/${artifact}`;
  } else {
    return Response.json({ detail: "Invalid result request" }, { status: 400 });
  }
  try {
    const response = await fetch(`${getBackendApiBaseUrl()}${path}`, { cache: "no-store" });
    const headers = new Headers();
    headers.set("Content-Type", response.headers.get("Content-Type") ?? "application/json");
    headers.set("Cache-Control", "no-store");
    return new Response(response.body, { status: response.status, headers });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Backend unavailable";
    return Response.json({ detail }, { status: 502 });
  }
}
