"""LLM 에이전트 검색 (T13) — frontier provider 기반 웹 검색/추론."""
import logging
import re
from typing import Any, Optional

from app.services.frontier_llm import chat_completion, has_configured_frontier_provider
from app.services.prompt_registry import DEFAULT_PROMPTS, get_prompt_registry

logger = logging.getLogger(__name__)

SEARCH_SYSTEM_PROMPT = DEFAULT_PROMPTS["search"]["system"]


def _strip_reasoning_tags(content: str) -> str:
    if "<think" not in (content or "").lower():
        return content or ""
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>\s*", "", content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think\b[^>]*>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


async def llm_search(
    query: str,
    context: Optional[str] = None,
) -> dict[str, Any]:
    """LLM 기반 에이전트 검색 (T13)."""
    if not has_configured_frontier_provider():
        return {
            "success": False,
            "message": "frontier provider 미설정",
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

    result = await chat_completion(
        task="search",
        messages=messages,
        max_tokens=512,
        temperature=0.1,
    )
    if not result["success"]:
        logger.error("LLM Search error: %s", result.get("message"))
        return {
            "success": False,
            "message": result.get("message", "LLM Search 오류"),
            "answer": "",
        }

    answer = _strip_reasoning_tags(result.get("content") or "")
    return {
        "success": True,
        "answer": answer,
        "provider": result.get("provider"),
        "model": result.get("model"),
    }
