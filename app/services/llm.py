"""프롬프트 조립 및 내부/외부 LLM 호출."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

from app.core.config import settings
from app.services.llm_queue import run_with_engine_queue
from app.services.prompt_registry import DEFAULT_PROMPTS, get_prompt_registry
from app.services.tool_registry import ToolRegistry, get_tool_registry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_RECOGNITION = DEFAULT_PROMPTS["main_answer"]["system"]


def build_user_prompt(query_text: str, llm_doc: str) -> str:
    """질의와 LLM용 문서를 합쳐 유저 메시지 생성."""
    return get_prompt_registry().render_user(
        "main_answer",
        query_text=query_text,
        llm_doc=llm_doc,
    )


async def call_internal_llm(
    query_text: str,
    llm_doc: str,
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_tools: bool = False,
    max_tool_rounds: int = 3,
    tool_registry: Optional[ToolRegistry] = None,
) -> str:
    """내부 LLM(Qwen, EXAONE 등) 호출. 미설정 시 목업 응답.

    `use_tools=True`이면 OpenAI-compatible tool calling 루프를 수행하며,
    12개 공공데이터 tool을 LLM이 호출하게 한다.
    """
    url = api_url or settings.internal_llm_api_url
    key = api_key or settings.internal_llm_api_key
    if not url or not key:
        return "(내부 LLM 미설정) 녹용은 일반적으로 고혈압 약과 함께 드셔도 되는 경우가 많습니다. 다만 개인에 따라 다를 수 있으니, 약사나 의사에게 한 번 여쭤보시는 것이 좋습니다. 정확한 판단은 의사·약사 상담이 필요합니다."

    messages = get_prompt_registry().render_messages(
        "main_answer",
        query_text=query_text,
        llm_doc=llm_doc,
    )

    if not use_tools:
        return await _post_chat_once(url, key, messages)

    registry = tool_registry or get_tool_registry()
    tools = registry.get_tool_schemas()
    if not tools:
        return await _post_chat_once(url, key, messages)

    return await run_chat_with_tools(
        messages=messages,
        api_url=url,
        api_key=key,
        tool_registry=registry,
        max_tool_rounds=max_tool_rounds,
    )


async def call_external_llm(
    censored_payload: str,
    *,
    provider: str = "openai",
) -> str:
    """외부 프론티어 모델(Gemini, GPT-4, Claude) 호출. 검열된 페이로드만 전달."""
    if provider == "openai" and settings.openai_api_key:
        url = "https://api.openai.com/v1/chat/completions"
        messages = get_prompt_registry().render_messages(
            "external_review",
            censored_payload=censored_payload,
        )

        async def post_external() -> str:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": messages,
                        "max_tokens": 512,
                    },
                )
                r.raise_for_status()
                data = r.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

        return await run_with_engine_queue("external", post_external)
    # 목업
    return censored_payload[:200] + "\n\n(외부 LLM 미설정 또는 동일 응답)"


async def run_chat_with_tools(
    *,
    messages: list[dict[str, Any]],
    api_url: str,
    api_key: str,
    tool_registry: ToolRegistry,
    model: str = "qwen",
    max_tokens: int = 512,
    max_tool_rounds: int = 3,
    engine: str = "internal",
) -> str:
    """Run OpenAI-compatible chat completion with tool-calling loop.

    Each round posts the current messages plus tool schemas. If the response
    contains `tool_calls`, this executes them via `ToolRegistry.dispatch` and
    feeds the results back as `role="tool"` messages.
    """
    conversation: list[dict[str, Any]] = list(messages)
    tools = tool_registry.get_tool_schemas()

    for _round in range(max_tool_rounds + 1):
        async def post_round(current_messages=list(conversation)) -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": current_messages,
                        "tools": tools,
                        "tool_choice": "auto",
                        "max_tokens": max_tokens,
                    },
                )
                r.raise_for_status()
                return r.json()

        data = await run_with_engine_queue(engine, post_round)
        message = data.get("choices", [{}])[0].get("message", {}) or {}
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            return message.get("content") or ""

        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": tool_calls,
        }
        conversation.append(assistant_entry)

        for call in tool_calls:
            tool_result = await _execute_tool_call(call, tool_registry)
            conversation.append(tool_result)

    logger.warning(
        "Tool-calling loop exhausted max_tool_rounds=%d without final answer",
        max_tool_rounds,
    )
    return ""


async def _execute_tool_call(
    tool_call: dict[str, Any],
    tool_registry: ToolRegistry,
) -> dict[str, Any]:
    call_id = tool_call.get("id", "")
    function_block = tool_call.get("function", {}) or {}
    tool_name = function_block.get("name", "")
    arguments = function_block.get("arguments", {})

    try:
        result = await tool_registry.dispatch(tool_name, arguments)
    except Exception as exc:
        logger.error("Tool dispatch crashed for %s: %s", tool_name, exc)
        result = {
            "success": False,
            "message": f"Tool dispatch crashed: {exc}",
            "items": [],
        }

    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool_name,
        "content": json.dumps(result, ensure_ascii=False),
    }


async def _post_chat_once(
    url: str,
    key: str,
    messages: list[dict[str, Any]],
    *,
    model: str = "qwen",
    max_tokens: int = 512,
) -> str:
    async def post_internal() -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
            )
            r.raise_for_status()
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

    return await run_with_engine_queue("internal", post_internal)
