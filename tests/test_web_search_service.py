from packages.web_search.service import WebSearchService


def test_openai_compatible_search_annotations_become_source_cards():
    service = WebSearchService(base_url="https://api.deepseek.com")
    response = {
        "choices": [
            {
                "message": {
                    "content": "DeepSeek API uses an OpenAI-compatible format for chat completions.",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "title": "DeepSeek API Docs",
                                "url": "https://api-docs.deepseek.com/",
                                "start_index": 0,
                                "end_index": 12,
                            },
                        }
                    ],
                }
            }
        ]
    }

    cards = service._parse_openai_compatible_search_response(
        response,
        query="DeepSeek API docs",
        max_results=5,
        provider="deepseek_api",
    )

    assert cards == [
        {
            "title": "DeepSeek API Docs",
            "url": "https://api-docs.deepseek.com/",
            "short_snippet": "DeepSeek API uses an OpenAI-compatible format for chat completions.",
            "source": "api-docs.deepseek.com",
            "why_it_matters": "Cited by the web search API response.",
            "provider": "deepseek_api",
            "rank": 1,
            "search_query": "DeepSeek API docs",
            "search_answer": "DeepSeek API uses an OpenAI-compatible format for chat completions.",
        }
    ]


def test_openai_compatible_search_markdown_links_are_parsed_when_annotations_missing():
    service = WebSearchService(base_url="https://api.deepseek.com")
    response = {
        "choices": [
            {
                "message": {
                    "content": "See [DeepSeek Docs](https://api-docs.deepseek.com/) for the official API format.",
                }
            }
        ]
    }

    cards = service._parse_openai_compatible_search_response(
        response,
        query="DeepSeek web search",
        max_results=5,
        provider="deepseek_api",
    )

    assert len(cards) == 1
    assert cards[0]["title"] == "DeepSeek Docs"
    assert cards[0]["url"] == "https://api-docs.deepseek.com/"
    assert cards[0]["provider"] == "deepseek_api"


def test_openai_compatible_search_text_without_urls_is_not_a_source_card():
    service = WebSearchService(base_url="https://api.deepseek.com")
    response = {
        "choices": [
            {
                "message": {
                    "content": "The search API returned an answer, but did not include URL citations.",
                }
            }
        ]
    }

    cards = service._parse_openai_compatible_search_response(
        response,
        query="surface temperature trend",
        max_results=5,
        provider="deepseek_api",
    )

    assert cards == []
