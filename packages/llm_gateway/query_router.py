from __future__ import annotations

from typing import Any, Dict, Optional

from packages.llm_gateway.unified_processor import UnifiedQueryProcessor


class QueryRouter:
    DEFAULT_MODEL = UnifiedQueryProcessor.DEFAULT_MODEL

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        model: str = DEFAULT_MODEL,
        planner: Optional[Any] = None,
        skills_root: Optional[str] = None,
        trust_env: bool = False,
        base_url: Optional[str] = None,
        request_retries: int = 2,
    ):
        self.processor = UnifiedQueryProcessor(
            api_key=api_key,
            client=client,
            model=model,
            planner=planner,
            skills_root=skills_root,
            trust_env=trust_env,
            base_url=base_url,
            request_retries=request_retries,
        )

    def route_query(
        self,
        user_request: str,
        additional_context: Optional[Dict[str, Any]] = None,
        dataset_context: Optional[Dict[str, Any]] = None,
        extracted_params: Optional[Dict[str, Any]] = None,
        max_tokens: int = 1200,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        return self.processor.process(
            query=user_request,
            dataset_context=dataset_context or {},
            conversation_context=(additional_context or {}).get("conversation_context", {}),
            extracted_params=extracted_params or {},
            additional_context=additional_context or {},
            max_tokens=max_tokens,
            temperature=temperature,
        )
