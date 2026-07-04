import { getBackendApiBaseUrl } from "./server-api";

function buildUnavailableResponse(failureLabel: string, error: unknown) {
  const message = error instanceof Error ? error.message : failureLabel;
  return Response.json(
    { detail: `Backend is unavailable. Start uvicorn on 127.0.0.1:8000. ${message}` },
    { status: 502 }
  );
}

function buildPassthroughResponse(response: Response, text: string, fallbackContentType: string) {
  return new Response(text, {
    status: response.status,
    headers: {
      "Content-Type": response.headers.get("Content-Type") ?? fallbackContentType
    }
  });
}

export async function proxyGet(pathname: string, failureLabel: string) {
  try {
    const response = await fetch(`${getBackendApiBaseUrl()}${pathname}`, {
      method: "GET",
      cache: "no-store"
    });

    const text = await response.text();
    return buildPassthroughResponse(response, text, "application/json");
  } catch (error) {
    return buildUnavailableResponse(failureLabel, error);
  }
}

export async function proxyJsonPost(
  request: Request,
  pathname: string,
  failureLabel: string,
  invalidJsonMessage = "Invalid JSON payload."
) {
  let payload: unknown;

  try {
    payload = await request.json();
  } catch {
    return Response.json({ detail: invalidJsonMessage }, { status: 400 });
  }

  try {
    const response = await fetch(`${getBackendApiBaseUrl()}${pathname}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload),
      cache: "no-store"
    });

    const text = await response.text();
    return buildPassthroughResponse(response, text, "application/json");
  } catch (error) {
    return buildUnavailableResponse(failureLabel, error);
  }
}

export async function proxyBinaryPost(
  request: Request,
  pathname: string,
  failureLabel: string,
  invalidJsonMessage = "Invalid JSON payload.",
) {
  let payload: unknown;

  try {
    payload = await request.json();
  } catch {
    return Response.json({ detail: invalidJsonMessage }, { status: 400 });
  }

  try {
    const response = await fetch(`${getBackendApiBaseUrl()}${pathname}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      cache: "no-store",
    });

    const buffer = await response.arrayBuffer();
    const headers = new Headers();
    headers.set("Content-Type", response.headers.get("Content-Type") ?? "application/octet-stream");

    const contentDisposition = response.headers.get("Content-Disposition");
    if (contentDisposition) {
      headers.set("Content-Disposition", contentDisposition);
    }

    return new Response(buffer, {
      status: response.status,
      headers,
    });
  } catch (error) {
    return buildUnavailableResponse(failureLabel, error);
  }
}

export async function proxyStreamPost(request: Request, pathname: string, failureLabel: string) {
  const rawBody = await request.text();

  try {
    const response = await fetch(`${getBackendApiBaseUrl()}${pathname}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: rawBody,
      cache: "no-store"
    });

    if (!response.body) {
      const text = await response.text();
      return buildPassthroughResponse(response, text, "application/json");
    }

    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type": response.headers.get("Content-Type") ?? "application/x-ndjson",
        "Cache-Control": "no-store, no-transform",
        "X-Accel-Buffering": "no"
      }
    });
  } catch (error) {
    return buildUnavailableResponse(failureLabel, error);
  }
}
