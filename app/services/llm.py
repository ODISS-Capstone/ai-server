"""프롬프트 조립 및 내부/외부 LLM 호출."""
from __future__ import annotations

import json
import logging
import re
from time import perf_counter
from typing import Any, Optional

import httpx

from app.core.config import settings
from app.core.safety import ensure_disclaimer
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


async def check_internal_llm_health(
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Check whether the configured OpenAI-compatible internal LLM is reachable."""
    url = api_url or settings.internal_llm_api_url
    selected_model = model or settings.internal_llm_model
    if not url:
        logger.warning("[InternalLLMHealth] not_configured")
        return {
            "status": "not_configured",
            "configured": False,
            "model": selected_model,
            "url": None,
            "message": "INTERNAL_LLM_API_URL is not set",
        }

    key = api_key if api_key is not None else settings.internal_llm_api_key
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    started = perf_counter()
    logger.info("[InternalLLMHealth] check_start model=%s url=%s", selected_model, url)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                url,
                headers=headers,
                json={
                    "model": selected_model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 8,
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            data = response.json()

        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        elapsed_ms = (perf_counter() - started) * 1000
        logger.info(
            "[InternalLLMHealth] check_ok model=%s status_code=%d answer_chars=%d elapsed_ms=%.1f",
            selected_model,
            response.status_code,
            len(answer),
            elapsed_ms,
        )
        return {
            "status": "ok",
            "configured": True,
            "model": selected_model,
            "url": url,
            "status_code": response.status_code,
            "elapsed_ms": round(elapsed_ms, 1),
            "answer_preview": answer[:80],
        }
    except httpx.HTTPStatusError as exc:
        elapsed_ms = (perf_counter() - started) * 1000
        logger.error(
            "[InternalLLMHealth] check_http_error model=%s status_code=%d elapsed_ms=%.1f",
            selected_model,
            exc.response.status_code,
            elapsed_ms,
        )
        return {
            "status": "error",
            "configured": True,
            "model": selected_model,
            "url": url,
            "status_code": exc.response.status_code,
            "elapsed_ms": round(elapsed_ms, 1),
            "message": exc.response.text[:300],
        }
    except Exception as exc:
        elapsed_ms = (perf_counter() - started) * 1000
        logger.error(
            "[InternalLLMHealth] check_failed model=%s error=%s elapsed_ms=%.1f",
            selected_model,
            exc,
            elapsed_ms,
        )
        return {
            "status": "error",
            "configured": True,
            "model": selected_model,
            "url": url,
            "elapsed_ms": round(elapsed_ms, 1),
            "message": str(exc),
        }


async def call_internal_llm(
    query_text: str,
    llm_doc: str,
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_tools: bool = False,
    max_tool_rounds: int = 3,
    tool_registry: Optional[ToolRegistry] = None,
    model: Optional[str] = None,
) -> str:
    """내부 LLM(Qwen, EXAONE 등) 호출. 미설정 시 목업 응답.

    `use_tools=True`이면 OpenAI-compatible tool calling 루프를 수행하며,
    12개 공공데이터 tool을 LLM이 호출하게 한다.
    """
    url = api_url or settings.internal_llm_api_url
    key = api_key if api_key is not None else settings.internal_llm_api_key
    if not url:
        logger.warning("[InternalLLM] not_configured fallback=true query_chars=%d", len(query_text or ""))
        return _safe_internal_fallback(query_text, llm_doc)

    messages = get_prompt_registry().render_messages(
        "main_answer",
        query_text=query_text,
        llm_doc=llm_doc,
    )

    if not use_tools:
        return await _post_chat_once(
            url,
            key,
            messages,
            model=model or settings.internal_llm_model,
        )

    registry = tool_registry or get_tool_registry()
    tools = registry.get_tool_schemas()
    if not tools:
        return await _post_chat_once(
            url,
            key,
            messages,
            model=model or settings.internal_llm_model,
        )

    return await run_chat_with_tools(
        messages=messages,
        api_url=url,
        api_key=key,
        tool_registry=registry,
        max_tool_rounds=max_tool_rounds,
        model=model or settings.internal_llm_model,
    )


async def call_local_delivery_llm(
    *,
    original_query: str,
    reviewed_message: str,
    user_profile: Optional[dict[str, Any]] = None,
    conversation_context: str = "",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    require_disclaimer: bool = True,
) -> str:
    """로컬 모델이 GPT Judge 검토문을 최종 사용자 발화로 변환."""
    source = reviewed_message.strip()
    if not source:
        return ensure_disclaimer(
            "확인된 정보가 부족해 바로 답변드리기 어렵습니다.",
            required=require_disclaimer,
        )

    url = api_url or settings.internal_llm_api_url
    key = api_key if api_key is not None else settings.internal_llm_api_key
    if not url:
        logger.warning("[DeliveryLLM] not_configured fallback=true original_query_chars=%d", len(original_query or ""))
        return ensure_disclaimer(source, required=require_disclaimer)

    messages = get_prompt_registry().render_messages(
        "local_delivery",
        original_query=original_query,
        reviewed_message=source,
        user_profile=json.dumps(user_profile or {}, ensure_ascii=False),
        conversation_context=conversation_context or "(없음)",
    )
    try:
        answer = await _post_chat_once(
            url,
            key,
            messages,
            model=model or settings.internal_llm_model,
            max_tokens=256,
            timeout_seconds=settings.local_delivery_llm_timeout_seconds,
            chat_template_kwargs={"enable_thinking": False},
        )
    except Exception as exc:  # noqa: BLE001 - delivery polish must never block final answer
        logger.warning("[DeliveryLLM] failed_fast fallback=true error=%s", exc)
        return ensure_disclaimer(source, required=require_disclaimer)
    return ensure_disclaimer(answer or source, required=require_disclaimer)


async def judge_identity_conflict(
    *,
    current_text: str,
    patient_profile: dict[str, Any],
    recent_history: str = "",
    current_time: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Use the internal LLM to decide whether a speaker conflicts with profile."""
    url = api_url or settings.internal_llm_api_url
    key = api_key if api_key is not None else settings.internal_llm_api_key
    fallback = _heuristic_identity_conflict(current_text, patient_profile)
    if not url:
        return {
            "conflict": fallback,
            "source": "heuristic_no_internal_llm",
            "raw": "",
        }

    messages = get_prompt_registry().render_messages(
        "identity_conflict_judge",
        current_time=current_time or "",
        patient_profile=json.dumps(patient_profile or {}, ensure_ascii=False),
        recent_history=recent_history[:1200] or "(없음)",
        current_text=current_text or "",
    )

    try:
        answer = await _post_chat_once(
            url,
            key,
            messages,
            model=model or settings.internal_llm_model,
            max_tokens=32,
        )
    except Exception as exc:  # noqa: BLE001 - identity gate should not block on LLM outage
        logger.warning("[IdentityJudge] internal_llm_failed fallback=%s error=%s", fallback, exc)
        return {
            "conflict": fallback,
            "source": "heuristic_after_internal_llm_error",
            "raw": repr(exc),
        }

    parsed = _parse_identity_judge_answer(answer)
    if parsed is None:
        parsed = fallback
    return {
        "conflict": parsed,
        "source": "internal_qwen",
        "raw": answer[:200],
        "heuristic_conflict": fallback,
    }


async def extract_identity_profile_with_llm(
    *,
    current_text: str,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Extract user or managed medication-subject identity fields using Qwen."""
    fallback = _heuristic_identity_extract(current_text)
    url = api_url or settings.internal_llm_api_url
    key = api_key if api_key is not None else settings.internal_llm_api_key
    if not url:
        return {
            "profile": fallback,
            "source": "heuristic_no_internal_llm",
            "raw": "",
        }

    messages = get_prompt_registry().render_messages(
        "identity_profile_extract",
        current_text=current_text or "",
    )
    try:
        answer = await _post_chat_once(
            url,
            key,
            messages,
            model=model or settings.internal_llm_model,
            max_tokens=160,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[IdentityExtract] internal_llm_failed fallback=%s error=%s", fallback, exc)
        return {
            "profile": fallback,
            "source": "heuristic_after_internal_llm_error",
            "raw": repr(exc),
        }

    parsed = _parse_identity_profile_answer(answer)
    if not parsed:
        parsed = fallback
    return {
        "profile": parsed,
        "source": "internal_qwen",
        "raw": answer[:300],
        "heuristic_profile": fallback,
    }


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
            async with httpx.AsyncClient(timeout=settings.openai_timeout_seconds) as client:
                r = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"},
                    json={
                        "model": settings.openai_model,
                        "messages": messages,
                        "max_tokens": 512,
                    },
                )
                r.raise_for_status()
                data = r.json()
                answer = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                return _strip_reasoning_tags(answer)

        return await run_with_engine_queue("external", post_external)
    # 목업
    return "외부 LLM이 설정되지 않아 추가 검토를 생략했습니다."


async def run_chat_with_tools(
    *,
    messages: list[dict[str, Any]],
    api_url: str,
    api_key: Optional[str],
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
            async with httpx.AsyncClient(timeout=settings.internal_llm_timeout_seconds) as client:
                r = await client.post(
                    api_url,
                    headers=_json_headers(api_key),
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
            return _strip_reasoning_tags(message.get("content") or "")

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


def _safe_internal_fallback(query_text: str, llm_doc: str) -> str:
    """내부 LLM이 꺼져 있을 때 특정 약효를 추측하지 않는 안전 fallback."""
    context_hint = llm_doc.strip()
    if context_hint:
        return ensure_disclaimer(
            f"'{query_text}'에 대해 확인된 기록은 있지만, 로컬 답변 모델이 아직 설정되지 않아 "
            "자세한 판단을 바로 드리기 어렵습니다. 복용 중인 약과 처방전을 가지고 약사나 의사에게 확인하세요."
        )
    return ensure_disclaimer(
        f"'{query_text}'에 대해 확인된 정보가 부족합니다. 약 이름, 처방전, 복용 중인 약 정보를 확인한 뒤 "
        "약사나 의사에게 상담하세요."
    )


def _parse_identity_judge_answer(answer: str) -> Optional[bool]:
    stripped = (answer or "").strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict) and isinstance(payload.get("conflict"), bool):
            return payload["conflict"]
    except json.JSONDecodeError:
        pass
    first = next((line.strip().upper() for line in stripped.splitlines() if line.strip()), "")
    if first.startswith("TRUE"):
        return True
    if first.startswith("FALSE"):
        return False
    if "TRUE" in first and "FALSE" not in first:
        return True
    if "FALSE" in first and "TRUE" not in first:
        return False
    return None


def _parse_identity_profile_answer(answer: str) -> dict[str, Any]:
    raw = (answer or "").strip()
    if not raw:
        return {}
    match = None
    if "{" in raw and "}" in raw:
        match = raw[raw.find("{"): raw.rfind("}") + 1]
    try:
        payload = json.loads(match or raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    profile: dict[str, Any] = {}
    name = str(payload.get("name") or "").strip()
    age = str(payload.get("age") or "").strip()
    gender = str(payload.get("gender") or "").strip()
    conditions = payload.get("conditions") or []
    if name:
        profile["name"] = name
    if age:
        profile["age"] = age
    if gender:
        profile["gender"] = gender
    if isinstance(conditions, list):
        normalized_conditions = [str(item).strip() for item in conditions if str(item).strip()]
        if normalized_conditions:
            profile["conditions"] = normalized_conditions
    return profile


def _heuristic_identity_extract(text: str) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    if not text:
        return profile

    name_patterns = [
        r"(?:제\s*이름은|이름은)\s*([가-힣]{2,5}?)(?:이고|고|입니다|이에요|예요|,|\s|$)",
        r"(?:저는|나는)\s*([가-힣]{2,5}?)(?:이고|고|입니다|이에요|예요|,|\s|$)",
        r"^\s*([가-힣]{2,5})\s*(?:남자|남성|여자|여성)",
        r"^\s*([가-힣]{2,5})\s*,?\s*\d{1,3}\s*(?:살|세)",
        (
            r"(?:대상자|아버지|어머니|엄마|아빠|남편|아내|배우자)"
            r"(?:\s*이름은|\s*성함은|\s*는|\s*가|\s*께서는)?\s*"
            r"([가-힣]{2,5}?)(?:이고|고|입니다|이에요|예요|,|\s+\d{1,3}\s*(?:살|세)|\s|$)"
        ),
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = match.group(1)
        if _looks_like_non_name_identity_candidate(candidate):
            continue
        profile["name"] = candidate
        break
    age_match = re.search(r"(\d{1,3})\s*(?:살|세)", text)
    if age_match:
        profile["age"] = age_match.group(1)
    if "남자" in text or "남성" in text:
        profile["gender"] = "남성"
    elif "여자" in text or "여성" in text:
        profile["gender"] = "여성"
    conditions = [
        token
        for token in ("고혈압", "당뇨", "천식", "신장질환", "간질환", "심장질환")
        if token in text
    ]
    if conditions:
        profile["conditions"] = conditions
    return profile


def _looks_like_non_name_identity_candidate(value: str) -> bool:
    return value in {
        "고혈압",
        "당뇨",
        "천식",
        "신장질환",
        "간질환",
        "심장질환",
        "임신",
        "딸",
        "아들",
        "보호자",
        "가족",
        "엄마",
        "아빠",
        "아버지",
        "어머니",
    }


def _heuristic_identity_conflict(text: str, profile: dict[str, Any]) -> bool:
    """Conservative fallback used only when the internal LLM is unavailable."""
    if not text or not profile:
        return False
    name = str(profile.get("name") or "").strip()
    age = str(profile.get("age") or "").strip()
    gender = str(profile.get("gender") or "").strip()
    if name and name in text:
        return False
    if age and (f"{age}살" in text or f"{age}세" in text):
        return False
    if gender and gender in text:
        return False
    if name and any(marker in text for marker in ("제 이름은", "저는", "나는")):
        if name not in text:
            return True
    if age and ("살" in text or "세" in text) and f"{age}살" not in text and f"{age}세" not in text:
        return True
    if gender == "남성" and any(token in text for token in ("여자", "여성")):
        return True
    if gender == "여성" and any(token in text for token in ("남자", "남성")):
        return True
    return False


def _json_headers(api_key: Optional[str]) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _strip_reasoning_tags(content: str) -> str:
    """Remove completed Qwen-style reasoning blocks from user-facing text."""
    if "<think" not in content.lower():
        return content
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>\s*", "", content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think\b[^>]*>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


async def _post_chat_once(
    url: str,
    key: Optional[str],
    messages: list[dict[str, Any]],
    *,
    model: str = "qwen",
    max_tokens: int = 512,
    timeout_seconds: Optional[float] = None,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
) -> str:
    async def post_internal() -> str:
        started = perf_counter()
        logger.info(
            "[InternalLLM] request_start model=%s url=%s messages=%d max_tokens=%d",
            model,
            url,
            len(messages),
            max_tokens,
        )
        async with httpx.AsyncClient(timeout=timeout_seconds or settings.internal_llm_timeout_seconds) as client:
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if chat_template_kwargs:
                payload["chat_template_kwargs"] = chat_template_kwargs
            r = await client.post(
                url,
                headers=_json_headers(key),
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            answer = _strip_reasoning_tags(
                data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            )
            logger.info(
                "[InternalLLM] request_done model=%s answer_chars=%d elapsed_ms=%.1f",
                model,
                len(answer),
                (perf_counter() - started) * 1000,
            )
            return answer

    return await run_with_engine_queue("internal", post_internal)
