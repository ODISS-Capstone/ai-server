"""LLM 에이전트 검색 (T13) — OpenAI API 기반 웹 검색/추론."""
import logging
from typing import Any, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

SEARCH_SYSTEM_PROMPT = (
    "당신은 의약품 및 건강 관련 정보를 검색하는 전문 에이전트입니다.\n"
    "사용자의 질문에 대해 정확하고 신뢰할 수 있는 의약 정보를 제공합니다.\n"
    "불확실한 정보는 '확인이 필요합니다'라고 명시합니다."
)


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

    messages = [{"role": "system", "content": SEARCH_SYSTEM_PROMPT}]
    if context:
        messages.append(
            {"role": "user", "content": f"참고 정보:\n{context}\n\n질문: {query}"}
        )
    else:
        messages.append({"role": "user", "content": query})

    try:
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
            data = resp.json()

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
