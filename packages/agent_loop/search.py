"""Agent web search through the Exa hosted MCP endpoint used by OceanX."""

from __future__ import annotations

import json
import re
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

EXA_MCP_ENDPOINT = "https://mcp.exa.ai/mcp"
WEB_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search current web sources and return titles, URLs, snippets, authors, and dates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                "search_type": {"type": "string", "enum": ["auto", "fast", "deep"]},
                "live_crawl": {"type": "string", "enum": ["fallback", "preferred"]},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
_RESULT = re.compile(
    r"(?:^|\n---\n)Title:\s*(?P<title>[^\n]+)\n"
    r"URL:\s*(?P<url>[^\n]+)\nPublished:\s*(?P<published>[^\n]+)\n"
    r"Author:\s*(?P<author>[^\n]+)\nHighlights:\s*\n"
    r"(?P<snippet>.*?)(?=\n---\nTitle:|\Z)", re.DOTALL,
)


def _mcp_text(body: str) -> str:
    candidates = [
        body.strip(),
        *(line[5:].strip() for line in body.splitlines() if line.startswith("data:")),
    ]
    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        payload = json.loads(candidate)
        if payload.get("error") is not None:
            raise ValueError(f"Exa MCP error: {payload['error']}")
        content = (payload.get("result") or {}).get("content")
        if isinstance(content, list):
            text = "\n".join(str(item.get("text") or "").strip() for item in content
                             if isinstance(item, dict) and item.get("type") == "text").strip()
            if text:
                return text
    raise ValueError("Exa MCP response contained no text result")


def _sources(body: str, limit: int) -> list[dict[str, Any]]:
    results = []
    for match in _RESULT.finditer(body):
        url = match.group("url").strip()
        parsed = urlparse(url)
        if len(url) > 2048 or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        published = match.group("published").strip()[:64]
        author = match.group("author").strip()[:256]
        results.append({
            "title": match.group("title").strip()[:512], "url": url,
            "snippet": match.group("snippet").strip()[:4000],
            "author": "" if author == "N/A" else author,
            "published_at": None if published == "N/A" else published,
        })
        if len(results) >= limit:
            break
    return results


def make_web_search_tool(
    *, client_factory: Callable[..., Any] | None = None, endpoint: str = EXA_MCP_ENDPOINT
):
    """Return a synchronous callable for the graph's name-to-function registry."""
    client_factory = client_factory or httpx.Client

    def web_search(
        query: str,
        max_results: int = 5,
        search_type: str = "auto",
        live_crawl: str = "fallback",
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 1000:
            raise ValueError("query must contain 1–1000 characters")
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or not 1 <= max_results <= 10
        ):
            raise ValueError("max_results must be an integer from 1 to 10")
        if (
            search_type not in {"auto", "fast", "deep"}
            or live_crawl not in {"fallback", "preferred"}
        ):
            raise ValueError("invalid search_type or live_crawl")
        if timeout is not None and timeout <= 0:
            raise TimeoutError("web_search deadline expired")
        query = query.strip()
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "web_search_exa", "arguments": {"query": query, "type": search_type,
            "numResults": max_results, "livecrawl": live_crawl, "contextMaxCharacters": 10_000},
        }}
        with client_factory(timeout=min(25.0, timeout) if timeout is not None else 25.0,
                            follow_redirects=False) as client:
            response = client.post(endpoint, headers={
                "Accept": "application/json, text/event-stream", "Content-Type": "application/json",
            }, json=body)
            response.raise_for_status()
        results = _sources(_mcp_text(response.text), max_results)
        if not results:
            raise ValueError("web_search returned no parseable results")
        return {"query": query, "provider": "exa", "results": results}

    return web_search
