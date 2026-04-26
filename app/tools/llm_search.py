"""LLM 에이전트 검색 (T13) — OpenAI API 기반 웹 검색/추론."""
import logging
from typing import Any, Optional

import httpx

from app.core.config import settings
from app.services.llm_queue import run_with_engine_queue
from app.services.prompt_registry import DEFAULT_PROMPTS, get_prompt_registry

logger = logging.getLogger(__name__)

SEARCH_SYSTEM_PROMPT = DEFAULT_PROMPTS["search"]["system"]


async def llm_search(
    query: str,
    context: Optional[str] = None,
) -> dict[str, Any]:
    """LLM 기반 에이전트 검색 (T13)."""
    api_key = settings.openai_api_key
    if not api_key:
        return {
            "success": False,
            "message": "openai_api_key 미설정",
            "answer": "",
        }

    if context:
        search_input = f"참고 정보:\n{context}\n\n질문: {query}"
    else:
        search_input = query
    messages = get_prompt_registry().render_messages(
        "search",
        search_input=search_input,
    )

    try:
        async def post_search() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.openai_model,
                        "messages": messages,
                        "max_tokens": 1024,
                    },
                )
                resp.raise_for_status()
                return resp.json()

        data = await run_with_engine_queue("search", post_search)

        answer = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return {"success": True, "answer": answer}
    except httpx.HTTPStatusError as e:
        logger.error("LLM Search API error: %s", e.response.status_code)
        return {"success": False, "message": str(e), "answer": ""}
    except Exception as e:
        logger.error("LLM Search error: %s", e)
        return {"success": False, "message": str(e), "answer": ""}
