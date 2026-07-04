from __future__ import annotations

import html
import os
import re
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from packages.llm_gateway.config import load_config_value, load_llm_api_key, load_llm_base_url, load_model_name


class WebSearchService:
    SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        fallback_to_duckduckgo: Optional[bool] = None,
        client_factory: Optional[Callable[..., httpx.Client]] = None,
    ):
        self.provider = (provider or os.getenv("WEB_SEARCH_PROVIDER") or "auto").strip().lower()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.fallback_to_duckduckgo = (
            fallback_to_duckduckgo
            if fallback_to_duckduckgo is not None
            else os.getenv("WEB_SEARCH_FALLBACK", "1").strip().lower() not in {"0", "false", "no", "off"}
        )
        self.client_factory = client_factory or httpx.Client
        self.last_provider: Optional[str] = None

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        trust_env: bool = False,
        timeout_sec: float = 12.0,
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        provider = self.provider
        if provider == "auto" and self._should_prefer_openai_compatible_search():
            provider = "openai_compatible"

        if provider in {"deepseek", "openai_compatible", "openai-compatible", "api"}:
            try:
                results = self._search_openai_compatible(
                    query,
                    max_results=max_results,
                    trust_env=trust_env,
                    timeout_sec=timeout_sec,
                )
                if results or not self.fallback_to_duckduckgo:
                    self.last_provider = self._provider_label()
                    return results
            except Exception:
                if not self.fallback_to_duckduckgo:
                    raise

        self.last_provider = "duckduckgo_html"
        return self._search_duckduckgo_html(
            query,
            max_results=max_results,
            trust_env=trust_env,
            timeout_sec=timeout_sec,
        )

    def _search_openai_compatible(
        self,
        query: str,
        *,
        max_results: int,
        trust_env: bool,
        timeout_sec: float,
    ) -> List[Dict[str, Any]]:
        endpoint = self._chat_completions_endpoint()
        payload: Dict[str, Any] = {
            "model": self._search_model(),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Search the web for the user's query and return a concise answer with citations. "
                        "Prefer primary or official sources when available."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "temperature": 0,
            "max_tokens": 900,
            "stream": False,
            "web_search_options": self._web_search_options(max_results=max_results),
        }

        with self.client_factory(timeout=timeout_sec, trust_env=trust_env, follow_redirects=True) as client:
            response = client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key or load_llm_api_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return self._parse_openai_compatible_search_response(
            data,
            query=query,
            max_results=max_results,
            provider=self._provider_label(),
        )

    def _search_duckduckgo_html(
        self,
        query: str,
        *,
        max_results: int,
        trust_env: bool,
        timeout_sec: float,
    ) -> List[Dict[str, Any]]:
        with self.client_factory(timeout=timeout_sec, trust_env=trust_env, follow_redirects=True) as client:
            response = client.get(
                self.SEARCH_ENDPOINT,
                params={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
                },
            )
            response.raise_for_status()
            results = self._parse_duckduckgo_html(response.text, max_results=max_results)
        for index, result in enumerate(results, start=1):
            result.setdefault("provider", "duckduckgo_html")
            result.setdefault("rank", index)
            result.setdefault("search_query", query)
        return results

    def _should_prefer_openai_compatible_search(self) -> bool:
        try:
            base_url = self.base_url or load_llm_base_url()
        except Exception:
            return False
        if "deepseek" in base_url.lower():
            return True
        return bool(os.getenv("WEB_SEARCH_MODEL"))

    def _search_model(self) -> str:
        if self.model:
            return self.model
        configured = load_config_value("WEB_SEARCH_MODEL")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        return load_model_name("WEB_ANSWER_MODEL")

    def _provider_label(self) -> str:
        base_url = (self.base_url or load_llm_base_url()).lower()
        if "deepseek" in base_url:
            return "deepseek_api"
        return "openai_compatible_api"

    def _chat_completions_endpoint(self) -> str:
        base_url = (self.base_url or load_llm_base_url()).rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def _web_search_options(self, *, max_results: int) -> Dict[str, Any]:
        options: Dict[str, Any] = {}
        context_size = os.getenv("WEB_SEARCH_CONTEXT_SIZE")
        if context_size and context_size.strip().lower() in {"low", "medium", "high"}:
            options["search_context_size"] = context_size.strip().lower()
        return options

    def _parse_openai_compatible_search_response(
        self,
        data: Dict[str, Any],
        *,
        query: str,
        max_results: int,
        provider: str,
    ) -> List[Dict[str, Any]]:
        message = self._first_choice_message(data)
        answer_text = self._message_text(message)
        annotations = self._message_annotations(message)
        cards = self._source_cards_from_annotations(
            annotations,
            answer_text=answer_text,
            query=query,
            max_results=max_results,
            provider=provider,
        )
        if cards:
            return cards

        cards = self._source_cards_from_markdown_links(
            answer_text,
            query=query,
            max_results=max_results,
            provider=provider,
        )
        if cards:
            return cards

        return []

    def _first_choice_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return {}
        first = choices[0]
        if not isinstance(first, dict):
            return {}
        message = first.get("message")
        return message if isinstance(message, dict) else {}

    def _message_text(self, message: Dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part).strip()
        return ""

    def _message_annotations(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        annotations = message.get("annotations")
        if isinstance(annotations, list):
            return [item for item in annotations if isinstance(item, dict)]
        return []

    def _source_cards_from_annotations(
        self,
        annotations: List[Dict[str, Any]],
        *,
        answer_text: str,
        query: str,
        max_results: int,
        provider: str,
    ) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()
        for annotation in annotations:
            citation = annotation.get("url_citation") if isinstance(annotation.get("url_citation"), dict) else annotation
            url = str(citation.get("url") or "").strip()
            title = str(citation.get("title") or "").strip()
            if not self._is_usable_result(title or url, url):
                continue
            normalized_url = self._normalize_result_url(url)
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            snippet = self._annotation_snippet(answer_text, citation)
            cards.append(
                {
                    "title": title or self._extract_domain(normalized_url),
                    "url": normalized_url,
                    "short_snippet": snippet or self._compact_text(answer_text, max_chars=280),
                    "source": self._extract_domain(normalized_url),
                    "why_it_matters": "Cited by the web search API response.",
                    "provider": provider,
                    "rank": len(cards) + 1,
                    "search_query": query,
                    "search_answer": answer_text,
                }
            )
            if len(cards) >= max_results:
                break
        return cards

    def _source_cards_from_markdown_links(
        self,
        answer_text: str,
        *,
        query: str,
        max_results: int,
        provider: str,
    ) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()
        pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
        for match in pattern.finditer(answer_text):
            title = self._clean_html(match.group(1))
            url = self._normalize_result_url(match.group(2))
            if not self._is_usable_result(title, url) or url in seen_urls:
                continue
            seen_urls.add(url)
            cards.append(
                {
                    "title": title,
                    "url": url,
                    "short_snippet": self._compact_text(answer_text, max_chars=280),
                    "source": self._extract_domain(url),
                    "why_it_matters": "Linked by the web search API response.",
                    "provider": provider,
                    "rank": len(cards) + 1,
                    "search_query": query,
                    "search_answer": answer_text,
                }
            )
            if len(cards) >= max_results:
                break
        return cards

    def _annotation_snippet(self, answer_text: str, citation: Dict[str, Any]) -> str:
        try:
            start = int(citation.get("start_index"))
            end = int(citation.get("end_index"))
        except (TypeError, ValueError):
            return ""
        if start < 0 or end <= start or start >= len(answer_text):
            return ""
        context_start = max(0, start - 160)
        context_end = min(len(answer_text), end + 160)
        return self._compact_text(answer_text[context_start:context_end], max_chars=320)

    def _parse_duckduckgo_html(self, text: str, *, max_results: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
            r'(?:<a[^>]*class="result__snippet"[^>]*>|<span[^>]*class="result__snippet"[^>]*>)'
            r'(?P<snippet>.*?)</(?:a|span)>',
            flags=re.IGNORECASE | re.DOTALL,
        )

        for match in pattern.finditer(text):
            url = self._normalize_result_url(match.group("url"))
            title = self._clean_html(match.group("title"))
            snippet = self._clean_html(match.group("snippet"))
            if not self._is_usable_result(title, url):
                continue
            results.append(
                {
                    "title": title,
                    "url": url,
                    "short_snippet": snippet,
                    "source": self._extract_domain(url),
                }
            )
            if len(results) >= max_results:
                break

        if results:
            return results

        fallback_pattern = re.compile(
            r'<a[^>]+href="(?P<url>https?://[^"]+)"[^>]*>(?P<title>.*?)</a>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in fallback_pattern.finditer(text):
            url = self._normalize_result_url(match.group("url"))
            title = self._clean_html(match.group("title"))
            if not self._is_usable_result(title, url):
                continue
            results.append(
                {
                    "title": title,
                    "url": url,
                    "short_snippet": "",
                    "source": self._extract_domain(url),
                }
            )
            if len(results) >= max_results:
                break
        return results

    def _clean_html(self, raw: str) -> str:
        text = re.sub(r"<.*?>", " ", raw)
        text = html.unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    def _compact_text(self, raw: str, *, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", raw or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "..."

    def _extract_domain(self, url: str) -> str:
        if "://" not in url:
            return "web"
        return url.split("://", 1)[1].split("/", 1)[0]

    def _normalize_result_url(self, raw_url: str) -> str:
        cleaned = html.unescape(re.sub(r"\s+", " ", raw_url)).strip()
        if not cleaned:
            return ""
        if cleaned.startswith("//"):
            cleaned = f"https:{cleaned}"
        if cleaned.startswith("/"):
            cleaned = f"https://duckduckgo.com{cleaned}"

        parsed = urlparse(cleaned)
        redirected = parse_qs(parsed.query).get("uddg", [])
        if redirected:
            return unquote(redirected[0]).strip()
        return cleaned

    def _is_usable_result(self, title: str, url: str) -> bool:
        if not title or not url:
            return False

        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if "duckduckgo.com" in domain:
            return False

        if title.strip().lower() in {"here", "more results", "duckduckgo"}:
            return False
        return True
