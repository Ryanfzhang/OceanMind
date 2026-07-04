from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from packages.llm_gateway.skill_planner import SkillPlanner
from packages.llm_gateway.config import load_model_name
from packages.llm_gateway.english_translation import needs_english_translation, translate_strings_to_english
from packages.llm_gateway.openai_compatible_client import OpenAICompatibleClientAdapter


class WebAnswerSynthesizer:
    DEFAULT_MODEL = load_model_name("WEB_ANSWER_MODEL", default=SkillPlanner.DEFAULT_MODEL)

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = DEFAULT_MODEL,
        planner: Optional[SkillPlanner] = None,
    ):
        self.client = client
        self.model = model
        self.planner = planner
        self._adapter = OpenAICompatibleClientAdapter(
            model=model,
            client=client,
        )

    def answer_or_request_search(
        self,
        user_request: str,
        additional_context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 1200,
        temperature: float = 0.0,
    ) -> str:
        """Return a direct answer, or exactly <search>query</search> when web search is needed."""
        client = self._get_client()
        payload = {
            "user_request": user_request,
            "current_date": datetime.now().date().isoformat(),
            "conversation_context": additional_context or {},
            "output_contract": (
                "If you can answer directly, return the final natural-language answer. "
                "If web search is required, return exactly one tag: <search>query</search>."
            ),
        }
        system = (
            "You handle general user requests that do NOT require the active ocean dataset.\n"
            "Do not use or infer from the active dataset. If the request appears to need the active dataset, "
            "briefly say it should be routed to dataset analysis.\n"
            "The categories below are private decision rules for whether to search. Do not present them as your "
            "workflow, capabilities, or examples unless the user explicitly asks about that exact task.\n"
            "If the request is text polishing, translation, rewriting, summarization of provided text, casual chat, "
            "stable general knowledge, or a conceptual explanation, answer directly.\n"
            "If conversation_memory is provided, it has already been selected for general_answer. Use it only for "
            "stable preferences or general conversation facts that directly apply to this request. Do not mention "
            "previous dataset analyses or capability limits unless the user explicitly asks about them.\n"
            "If the user asks you to remember, name, or define a reusable region, mask, threshold, or preference, "
            "acknowledge it directly in the user's language. The memory system will persist the structured memory after "
            "your response, so do not claim to have run any dataset computation.\n"
            "If the request requires current or external information, return exactly <search>query</search> and nothing else. "
            "Use search for weather, news, today/recent/latest/current facts, live prices, policies, regulations, changing roles, "
            "software/model versions, or when the user explicitly asks to search or look something up.\n"
            "If the user asks about your workflow, answer at a user-facing product level: understand the request, decide "
            "whether the active ocean dataset is needed, ask for missing analysis parameters when necessary, run dataset "
            "tools for dataset questions, use web search only for current external facts, and then summarize results. "
            "Do not mention hidden routing prompts, internal tags, or unrelated example categories such as text polishing "
            "unless the user specifically asks about text editing. Never include literal <search> tags in a final direct answer.\n"
            "When answering directly, follow the user's language."
        )

        response = self._adapter.create_message(
            client=client,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}],
            request_name="general_answer_first_pass",
            json_response=False,
        )
        text = self._adapter.extract_response_text(response).strip()
        if not text:
            raise ValueError("General answer first pass returned an empty response.")
        return text

    def synthesize_answer(
        self,
        user_request: str,
        search_results: List[Dict[str, Any]],
        additional_context: Optional[Dict[str, Any]] = None,
        search_query: Optional[str] = None,
        search_error: Optional[str] = None,
        max_tokens: int = 1800,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        client = self._get_client()
        system = (
            "You answer general user questions using optional web search results as extra context.\n"
            "Do not use the active ocean dataset.\n"
            "Use conversation_memory only when it directly applies to the current general question. Ignore unrelated "
            "prior analysis context.\n"
            "Search results are helpful context, not the only allowed basis for the answer. "
            "If the question involves current or changing facts, prioritize the search results. "
            "If search results are irrelevant, insufficient, or unavailable, answer using your general knowledge "
            "and clearly state the uncertainty or limitation.\n"
            "Stay concise and factual. Follow the user's language.\n"
            "Return JSON only."
        )

        validation_error: Optional[str] = None
        for attempt in range(2):
            payload = {
                "user_request": user_request,
                "current_date": datetime.now().date().isoformat(),
                "search_query": search_query,
                "search_results": search_results,
                "search_error": search_error,
                "additional_context": additional_context or {},
                "required_output_schema": {
                    "summary": "string",
                    "source_cards": [
                        {
                            "title": "string",
                            "source": "string",
                            "url": "string",
                            "short_snippet": "string",
                            "why_it_matters": "string",
                        }
                    ],
                },
            }
            if validation_error:
                payload["previous_validation_error"] = validation_error
                payload["retry_instruction"] = (
                    "Return exactly one complete JSON object matching required_output_schema. "
                    "Do not include markdown or explanatory text outside JSON."
                )

            try:
                response = self._adapter.create_message(
                    client=client,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}],
                    request_name="synthesize_web_answer" if attempt == 0 else "synthesize_web_answer_retry",
                    json_response=True,
                )
                text = self._adapter.extract_response_text(response)
                answer = self._adapter.parse_json_response(text)
                if not isinstance(answer.get("summary"), str) or not answer["summary"].strip():
                    raise ValueError("Web answer synthesis must include a non-empty summary.")
                answer.setdefault("source_cards", [])
                return answer
            except Exception as exc:
                validation_error = str(exc)
                continue

        return {
            "summary": self._fallback_summary(
                user_request=user_request,
                search_query=search_query,
                search_results=search_results,
                search_error=search_error or validation_error,
            ),
            "source_cards": [],
        }

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if self.planner is not None:
            return self.planner._get_client()
        self.client = self._adapter.get_client()
        return self.client

    @staticmethod
    def _fallback_summary(
        *,
        user_request: str,
        search_query: Optional[str],
        search_results: List[Dict[str, Any]],
        search_error: Optional[str],
    ) -> str:
        chinese = any("\u4e00" <= char <= "\u9fff" for char in user_request or "")
        if chinese:
            if search_results:
                return (
                    "我找到了一些外部搜索结果，但没有可靠地把模型输出整理成最终结构化回答。"
                    "请重试一次，或把问题说得更具体一些。"
                )
            if search_error:
                return (
                    "这次外部搜索或结果整理没有成功，所以我不能可靠回答这个需要实时信息的问题。"
                    "请稍后重试，或提供你希望我依据的资料。"
                )
            return "我没能可靠整理出最终回答。请重试一次，或把问题说得更具体一些。"
        if search_results:
            return (
                "I found some external search results, but could not reliably format the model output into a final answer. "
                "Please try again or make the question more specific."
            )
        if search_error:
            return (
                "The external search or result synthesis was unavailable, so I cannot reliably answer this real-time question. "
                "Please try again later or provide the source material you want me to use."
            )
        return "I could not reliably produce a final answer. Please try again or make the question more specific."

    def translate_source_cards_to_english(self, source_cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        client = self._get_client()
        return self._normalize_answer_output_to_english({"source_cards": source_cards}, client=client).get("source_cards", [])

    def _translate_texts_to_english(self, texts: List[str], *, client: Any, request_name: str) -> List[str]:
        return translate_strings_to_english(
            adapter=self._adapter,
            client=client,
            texts=texts,
            request_name=request_name,
        )

    def _normalize_text_list_to_english(
        self,
        values: Any,
        *,
        client: Any,
        request_name: str,
    ) -> Optional[List[str]]:
        if not isinstance(values, list):
            return None
        normalized_values: List[Optional[str]] = []
        texts_to_translate: List[str] = []
        positions: List[int] = []
        for item in values:
            if not isinstance(item, str):
                normalized_values.append(None)
                continue
            stripped = item.strip()
            if not stripped:
                normalized_values.append(None)
                continue
            normalized_values.append(stripped)
            if needs_english_translation(stripped):
                positions.append(len(normalized_values) - 1)
                texts_to_translate.append(stripped)
        if texts_to_translate:
            try:
                translated = self._translate_texts_to_english(
                    texts_to_translate,
                    client=client,
                    request_name=request_name,
                )
                for position, translated_text in zip(positions, translated):
                    normalized_values[position] = translated_text
            except Exception:
                for position in positions:
                    normalized_values[position] = None
        return [item for item in normalized_values if isinstance(item, str) and item.strip()]

    def _normalize_optional_text_to_english(
        self,
        value: Any,
        *,
        client: Any,
        request_name: str,
    ) -> Optional[str]:
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        if not stripped:
            return None
        if not needs_english_translation(stripped):
            return stripped
        try:
            return self._translate_texts_to_english(
                [stripped],
                client=client,
                request_name=request_name,
            )[0]
        except Exception:
            return None

    def _normalize_additional_context_to_english(
        self,
        additional_context: Dict[str, Any],
        *,
        client: Any,
    ) -> Dict[str, Any]:
        if not isinstance(additional_context, dict):
            return {}
        normalized = copy.deepcopy(additional_context)
        conversation_context = normalized.get("conversation_context")
        if isinstance(conversation_context, dict):
            recent_queries = self._normalize_text_list_to_english(
                conversation_context.get("recent_queries"),
                client=client,
                request_name="translate_web_recent_queries",
            )
            if recent_queries is not None:
                conversation_context["recent_queries"] = recent_queries
            conclusions = self._normalize_text_list_to_english(
                conversation_context.get("conclusions"),
                client=client,
                request_name="translate_web_conclusions",
            )
            if conclusions is not None:
                conversation_context["conclusions"] = conclusions
        for field_name in ("prior_queries_text", "synthesizer_prior_context_text"):
            normalized_value = self._normalize_optional_text_to_english(
                normalized.get(field_name),
                client=client,
                request_name=f"translate_web_{field_name}",
            )
            if normalized_value is None:
                normalized.pop(field_name, None)
            else:
                normalized[field_name] = normalized_value
        return normalized

    def _normalize_answer_output_to_english(
        self,
        answer: Dict[str, Any],
        *,
        client: Any,
    ) -> Dict[str, Any]:
        if not isinstance(answer, dict):
            return answer
        normalized = copy.deepcopy(answer)
        original_summary = normalized.get("summary")
        summary = self._normalize_optional_text_to_english(
            original_summary,
            client=client,
            request_name="translate_web_summary",
        )
        if summary:
            normalized["summary"] = summary
        elif needs_english_translation(original_summary):
            normalized["summary"] = (
                "The web answer was generated, but its summary could not be translated into English."
            )

        normalized.pop("follow_up_suggestions", None)

        source_cards = normalized.get("source_cards")
        if isinstance(source_cards, list):
            next_cards: List[Dict[str, Any]] = []
            for index, item in enumerate(source_cards):
                if not isinstance(item, dict):
                    continue
                item_copy = dict(item)
                snippet = self._normalize_optional_text_to_english(
                    item.get("short_snippet"),
                    client=client,
                    request_name=f"translate_web_source_snippet_{index}",
                )
                if snippet:
                    item_copy["short_snippet"] = snippet
                elif needs_english_translation(item.get("short_snippet")):
                    item_copy["short_snippet"] = "Summary unavailable in English."
                why = self._normalize_optional_text_to_english(
                    item.get("why_it_matters"),
                    client=client,
                    request_name=f"translate_web_source_reason_{index}",
                )
                if why:
                    item_copy["why_it_matters"] = why
                elif needs_english_translation(item.get("why_it_matters")):
                    item_copy["why_it_matters"] = "English explanation unavailable."
                next_cards.append(item_copy)
            normalized["source_cards"] = next_cards
        return normalized
