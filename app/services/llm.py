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
            temperature=settings.internal_llm_delivery_temperature,
            timeout_seconds=settings.local_delivery_llm_timeout_seconds,
            chat_template_kwargs={"enable_thinking": False},
        )
    except Exception as exc:  # noqa: BLE001 - delivery polish must never block final answer
        logger.warning("[DeliveryLLM] failed_fast fallback=true error=%s", exc)
        return ensure_disclaimer(source, required=require_disclaimer)
    return ensure_disclaimer(answer or source, required=require_disclaimer)


async def recover_medical_followup_with_llm(
    *,
    current_text: str,
    conversation_context: str = "",
    user_profile: Optional[dict[str, Any]] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Use the local LLM to rescue short follow-ups that deterministic routing would suppress."""
    url = api_url or settings.internal_llm_api_url
    key = api_key if api_key is not None else settings.internal_llm_api_key
    if not url:
        return {"is_medical_followup": False, "response": "", "source": "local_llm_not_configured"}

    messages = [
        {
            "role": "system",
            "content": (
                "너는 한국어 복약 상담 대화 라우터다. 사용자의 현재 발화가 직전 의료/복약 상담에 이어지는 "
                "짧은 후속 발화인지 판단하고, 맞으면 안전하고 간결한 답변을 만든다. "
                "어르신 발화는 '그거', '그럼?', '먹어 말어?', '어디다 대?', '뭐라고?'처럼 맥락을 생략할 수 있으므로 최근 대화와 함께 해석한다. "
                "단순 잡담, 호출어, 의미 없는 추임새, 의료 맥락 없는 발화면 false로 둔다. "
                "위험 복용, 임의 증량, 보조식품 병용, 복용 여부 혼동은 보수적으로 안전 안내한다. "
                "response는 반드시 자연스러운 한국어 한글 문장으로만 작성하고, 중국어/일본어/영어 표현을 섞지 않는다. "
                "반드시 JSON만 출력한다: "
                "{\"is_medical_followup\": true|false, \"intent\": \"...\", \"response\": \"...\"}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"[사용자 프로필]\n{json.dumps(user_profile or {}, ensure_ascii=False)}\n\n"
                f"[최근 대화/메모리]\n{conversation_context[:1600] or '(없음)'}\n\n"
                f"[현재 발화]\n{current_text}"
            ),
        },
    ]
    try:
        answer = await _post_chat_once(
            url,
            key,
            messages,
            model=model or settings.internal_llm_model,
            max_tokens=160,
            temperature=settings.internal_llm_route_temperature,
            timeout_seconds=settings.local_delivery_llm_timeout_seconds,
            chat_template_kwargs={"enable_thinking": False},
        )
    except Exception as exc:  # noqa: BLE001 - suppression fallback should stay non-blocking
        logger.warning("[FollowupRecoveryLLM] failed fallback=false error=%s", exc)
        return {"is_medical_followup": False, "response": "", "source": "local_llm_error", "raw": repr(exc)}

    parsed = _parse_followup_recovery_answer(answer)
    parsed["source"] = "local_llm"
    parsed["raw"] = answer[:300]
    if parsed.get("is_medical_followup") and parsed.get("response"):
        parsed["response"] = ensure_disclaimer(str(parsed["response"]), required=True)
    return parsed


async def classify_reasoning_route_with_llm(
    *,
    current_text: str,
    conversation_context: str = "",
    user_profile: Optional[dict[str, Any]] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Use the local LLM as the primary intent/router classifier."""
    url = api_url or settings.internal_llm_api_url
    key = api_key if api_key is not None else settings.internal_llm_api_key
    if not url:
        return {"source": "local_llm_not_configured", "usable": False}

    messages = [
        {
            "role": "system",
            "content": (
                "너는 ODISS 복약 상담 서버의 라우팅 엔진이다. 사용자의 현재 발화를 보고 하나의 route를 고른다. "
                "문자열 키워드에 매달리지 말고 의미와 최근 메모리 맥락을 함께 판단한다.\n"
                "어르신 사용자는 직전 답변을 못 듣거나 맥락을 생략해 '그럼?', '그거 먹어?', '어디다 대?', '뭐라고?'처럼 말할 수 있다. "
                "이 경우 최근 대화가 복약/OCR/안전 안내라면 그 맥락의 후속 발화로 해석한다.\n"
                "가능한 route_label:\n"
                "- ocr_capture: 사용자가 사진/카메라/처방전/약봉투 촬영 또는 OCR 등록을 요청함. "
                "약 이름 없이 '새 약 받아왔어', '약 타왔어', '새 처방 받았어'처럼 말하면 새 약봉투/처방전 확인이 필요한 것으로 보고 이 route를 고름\n"
                "- ocr_result: OCR 결과 텍스트를 처리/저장/검토해야 함\n"
                "- drug_identification: 알약/낱알/이 약이 무엇인지 식별 요청이며 촬영이 필요할 수 있음\n"
                "- meal_medication_prep: 밥을 먹고 난 뒤 어떤 약을 먹을지 나중에 알려달라는 준비 요청\n"
                "- after_meal_medication: 사용자가 이미 밥을 먹고 와서 지금 어떤 식후 약을 먹을지 묻는 요청\n"
                "- medication_record: 사용자가 '약 먹었어', '혈압약 먹었어'처럼 복용 완료를 말하거나 기록해달라는 요청. "
                "다만 '먹어도 돼?', '괜찮아?', '두 번', '못 먹었어'처럼 안전 판단이 필요한 질문이면 medication_safety_query로 둠\n"
                "- medication_taken_recall: '아까 약 먹었나?', '먹었는지 모르겠어', '헷갈려'처럼 방금/아까/오늘 약을 먹었는지 확인 요청\n"
                "- medication_safety_query: 약/건기식/용량/같이 먹기/위험성/중복복용 등 안전 질문\n"
                "- supplement_query: 건강기능식품/한약재/영양제 관련 질문\n"
                "- profile_recall: 내가 누구인지/내 프로필이 무엇인지 묻는 요청\n"
                "- lifestyle_memory: 생활습관/일상 기억 저장 또는 회상\n"
                "- non_actionable_ack: 네, 알겠습니다, 응, 좋아요처럼 직전 질문에 답하는 confirmation이 아니고 새 작업도 아닌 단순 확인/맞장구\n"
                "- noise_fragment: 스읍, 음, 나중에 밤처럼 의미가 불완전한 STT 조각, 숨소리, 추임새, 잘린 문장\n"
                "- smalltalk: 감사/인사/짧은 정서 표현\n"
                "- emergency: 응급상황\n"
                "- unknown: 의료/복약 맥락이 불명확하거나 무시해야 함\n"
                "non_actionable_ack/noise_fragment/unknown은 보통 사용자에게 답하지 않고 무시한다. "
                "단, 직전 시스템 질문에 대한 명확한 예/아니오 답변이면 해당 업무 route를 고른다.\n"
                "출력 JSON 필드: "
                "{\"route_label\":\"...\", \"mode\":\"MEMORY_ONLY|TOOL_FIRST|FRONTIER_FIRST|ASK_USER_CLARIFY\", "
                "\"intent\":\"medication_query|supplement_query|drug_identification|smalltalk|emergency|unknown\", "
                "\"task_types\":[\"request_ocr\"|\"dur_check\"|\"supplement_lookup\"|\"search_history\"|\"hira_lookup\"|\"dur_product_info\"], "
                "\"rationale\":\"짧은 이유\"}. JSON만 출력한다."
            ),
        },
        {
            "role": "user",
            "content": (
                f"[사용자 프로필]\n{json.dumps(user_profile or {}, ensure_ascii=False)}\n\n"
                f"[최근 메모리/대화]\n{conversation_context[:1800] or '(없음)'}\n\n"
                f"[현재 발화]\n{current_text}"
            ),
        },
    ]
    try:
        answer = await _post_chat_once(
            url,
            key,
            messages,
            model=model or settings.internal_llm_model,
            max_tokens=192,
            temperature=settings.internal_llm_route_temperature,
            timeout_seconds=settings.local_delivery_llm_timeout_seconds,
            chat_template_kwargs={"enable_thinking": False},
        )
    except Exception as exc:  # noqa: BLE001 - routing must fall back to deterministic code
        logger.warning("[RouteLLM] failed fallback=true error=%s", exc)
        return {"source": "local_llm_error", "usable": False, "raw": repr(exc)}

    parsed = _parse_route_classifier_answer(answer)
    parsed["source"] = "local_llm"
    parsed["raw"] = answer[:400]
    return parsed


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
    if not url:
        return {
            "conflict": False,
            "source": "local_llm_not_configured",
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
            temperature=settings.internal_llm_route_temperature,
            chat_template_kwargs={"enable_thinking": False},
        )
    except Exception as exc:  # noqa: BLE001 - identity gate should not block on LLM outage
        logger.warning("[IdentityJudge] local_llm_failed conflict=false error=%s", exc)
        return {
            "conflict": False,
            "source": "local_llm_error",
            "raw": repr(exc),
        }

    parsed = _parse_identity_judge_answer(answer)
    if parsed is None:
        parsed = False
    return {
        "conflict": parsed,
        "source": "local_llm",
        "raw": answer[:200],
    }


async def judge_pending_identity_reply_with_llm(
    *,
    current_text: str,
    patient_profile: dict[str, Any],
    pending_action: str,
    extracted_profile: Optional[dict[str, Any]] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Classify replies to pending identity verification without keyword heuristics."""
    url = api_url or settings.internal_llm_api_url
    key = api_key if api_key is not None else settings.internal_llm_api_key
    if not url:
        return {"decision": "unclear", "profile": extracted_profile or {}, "source": "local_llm_not_configured"}

    messages = [
        {
            "role": "system",
            "content": (
                "너는 복약 상담 시스템의 신원 확인 답변 판정기다. 현재 발화가 저장된 환자 본인 확인 질문에 대한 "
                "답인지 판단한다. '맞아' 같은 긍정어가 있어도 다른 이름/나이/성별이 함께 나오면 단순 same_person으로 보지 말고 "
                "provided_identity 또는 different_person으로 판정한다. 숨소리, 감탄사, 잘린 STT, 의미 없는 소리는 noise로 판정한다. "
                "신원 확인 질문 뒤의 '아니', '아니야', '내가 아니야', '다른 사람이야'는 noise가 아니라 rejected로 판정한다. "
                "신원 확인 질문 뒤의 '네', '맞아', '본인 맞아'처럼 다른 신원 정보가 없는 명확한 긍정은 same_person으로 판정한다. "
                "가능한 decision: same_person, different_person, provided_identity, rejected, noise, unclear. "
                "JSON만 출력한다: {\"decision\":\"...\", \"profile\": {\"name\":\"\", \"age\":\"\", \"gender\":\"\", \"conditions\":[]}, \"rationale\":\"...\"}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"[pending_action]\n{pending_action}\n\n"
                f"[저장된 환자 프로필]\n{json.dumps(patient_profile or {}, ensure_ascii=False)}\n\n"
                f"[이미 추출된 프로필 후보]\n{json.dumps(extracted_profile or {}, ensure_ascii=False)}\n\n"
                f"[현재 발화]\n{current_text}"
            ),
        },
    ]
    try:
        answer = await _post_chat_once(
            url,
            key,
            messages,
            model=model or settings.internal_llm_model,
            max_tokens=120,
            temperature=settings.internal_llm_route_temperature,
            chat_template_kwargs={"enable_thinking": False},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[PendingIdentityJudge] local_llm_failed fallback=unclear error=%s", exc)
        return {"decision": "unclear", "profile": extracted_profile or {}, "source": "local_llm_error", "raw": repr(exc)}

    parsed = _parse_pending_identity_reply_answer(answer)
    parsed["source"] = "local_llm"
    parsed["raw"] = answer[:300]
    return parsed


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
            temperature=settings.internal_llm_route_temperature,
            chat_template_kwargs={"enable_thinking": False},
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


async def extract_ocr_medication_candidates_with_llm(
    raw_text: str,
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Extract OCR medication candidates with a frontier LLM, not the dialogue model."""
    if not raw_text.strip():
        return {"medications": [], "clarification_question": "", "source": "empty_ocr_text"}

    messages = [
        {
            "role": "system",
            "content": (
                "너는 한국 처방전 OCR 텍스트에서 처방 의약품 후보를 추출하는 엔진이다. "
                "반드시 JSON만 출력한다. 추측은 하지 말고, OCR 원문에 보이는 약명 후보만 사용한다. "
                "의약품명은 보통 '정', '캡슐', '시럽', '장용정' 또는 mg 표기를 포함한다. "
                "용법/처방 목적/증상이 불명확하면 clarification_question에 사용자에게 물어볼 짧은 질문을 넣어라."
            ),
        },
        {
            "role": "user",
            "content": (
                "다음 OCR 원문에서 약명 후보를 JSON으로 추출해.\n"
                "형식: {\"medications\":[{\"name\":\"\",\"dosage\":\"\",\"frequency\":\"\",\"timing\":\"\",\"purpose_or_symptom\":\"\"}],"
                "\"clarification_question\":\"\"}\n\n"
                f"OCR 원문:\n{raw_text[:3000]}"
            ),
        },
    ]
    answer = ""
    try:
        if settings.openai_api_key:
            async def post_openai() -> str:
                async with httpx.AsyncClient(timeout=max(settings.openai_timeout_seconds, 12.0)) as client:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {settings.openai_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model or settings.openai_model,
                            "messages": messages,
                            "max_tokens": 320,
                            "temperature": 0,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

            answer = await run_with_engine_queue("external", post_openai)
            source = "frontier_openai"
        elif settings.google_ai_api_key:
            async def post_gemini() -> str:
                gemini_model = model or "gemini-2.5-flash"
                async with httpx.AsyncClient(timeout=max(settings.openai_timeout_seconds, 12.0)) as client:
                    response = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent",
                        headers={"Content-Type": "application/json"},
                        params={"key": settings.google_ai_api_key},
                        json={
                            "contents": [
                                {
                                    "role": "user",
                                    "parts": [
                                        {"text": messages[0]["content"] + "\n\n" + messages[1]["content"]}
                                    ],
                                }
                            ],
                            "generationConfig": {
                                "temperature": 0,
                                "maxOutputTokens": 320,
                            },
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    candidates = data.get("candidates") or []
                    parts = (
                        candidates[0]
                        .get("content", {})
                        .get("parts", [])
                        if candidates
                        else []
                    )
                    return "\n".join(str(part.get("text") or "") for part in parts)

            answer = await run_with_engine_queue("external", post_gemini)
            source = "frontier_gemini"
        else:
            return {"medications": [], "clarification_question": "", "source": "no_frontier_llm"}
    except Exception as exc:  # noqa: BLE001 - OCR parsing must fall back quickly
        logger.warning("[OCRMedicationExtract] frontier_llm_failed error=%s", exc)
        return {"medications": [], "clarification_question": "", "source": "frontier_llm_error", "raw": repr(exc)}

    parsed = _parse_ocr_medication_candidates_answer(answer)
    parsed["source"] = source
    parsed["raw"] = answer[:300]
    return parsed


async def refine_ocr_medication_candidates_with_context(
    *,
    raw_text: str,
    current_medications: list[dict[str, Any]],
    user_text: str,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Use a frontier LLM to correct OCR medication candidates with user symptom context."""
    if not (raw_text or current_medications or user_text):
        return {"medications": [], "clarification_question": "", "source": "empty_ocr_context"}
    current = json.dumps(current_medications or [], ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": (
                "너는 한국 처방전 OCR 약명 보정 엔진이다. OCR 원문, 현재 약명 후보, 사용자의 증상/처방 이유를 함께 보고 "
                "약명 후보를 재평가한다. 예를 들어 사용자가 통풍 때문에 받은 약이라고 말했는데 OCR 후보가 진통제/항히스타민제처럼 "
                "맥락과 맞지 않으면, 한국에서 통풍 치료에 실제 쓰이는 약명 후보(예: 페브릭정 등)를 제안할 수 있다. "
                "단, 확신이 낮으면 기존 후보를 유지하고 clarification_question에 확인 질문을 넣는다. 반드시 JSON만 출력한다."
            ),
        },
        {
            "role": "user",
            "content": (
                "형식: {\"medications\":[{\"name\":\"\",\"dosage\":\"\",\"frequency\":\"\",\"timing\":\"\","
                "\"purpose_or_symptom\":\"\",\"correction_reason\":\"\"}],\"clarification_question\":\"\"}\n\n"
                f"[OCR 원문]\n{(raw_text or '')[:2500]}\n\n"
                f"[현재 약명 후보]\n{current[:1500]}\n\n"
                f"[사용자 증상/처방 이유]\n{user_text[:1000]}"
            ),
        },
    ]
    try:
        if settings.openai_api_key:
            async def post_openai() -> str:
                async with httpx.AsyncClient(timeout=max(settings.openai_timeout_seconds, 12.0)) as client:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {settings.openai_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model or settings.openai_model,
                            "messages": messages,
                            "max_tokens": 360,
                            "temperature": 0,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

            answer = await run_with_engine_queue("external", post_openai)
            source = "frontier_openai_context_refine"
        elif settings.google_ai_api_key:
            async def post_gemini() -> str:
                gemini_model = model or "gemini-2.5-flash"
                async with httpx.AsyncClient(timeout=max(settings.openai_timeout_seconds, 12.0)) as client:
                    response = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent",
                        headers={"Content-Type": "application/json"},
                        params={"key": settings.google_ai_api_key},
                        json={
                            "contents": [
                                {
                                    "role": "user",
                                    "parts": [
                                        {"text": messages[0]["content"] + "\n\n" + messages[1]["content"]}
                                    ],
                                }
                            ],
                            "generationConfig": {"temperature": 0, "maxOutputTokens": 360},
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    candidates = data.get("candidates") or []
                    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
                    return "\n".join(str(part.get("text") or "") for part in parts)

            answer = await run_with_engine_queue("external", post_gemini)
            source = "frontier_gemini_context_refine"
        else:
            return {"medications": [], "clarification_question": "", "source": "no_frontier_llm"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("[OCRMedicationRefine] frontier_llm_failed error=%s", exc)
        return {"medications": [], "clarification_question": "", "source": "frontier_llm_error", "raw": repr(exc)}

    parsed = _parse_ocr_medication_candidates_answer(answer)
    parsed["source"] = source
    parsed["raw"] = answer[:300]
    return parsed


async def judge_prior_conversation_turn(
    *,
    current_text: str,
    stored_profile: dict[str, Any],
    extracted_profile: dict[str, Any] | None = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """LLM judges STT reply to 'have we talked before?' — avoids scripted loops."""
    url = api_url or settings.internal_llm_api_url
    key = api_key if api_key is not None else settings.internal_llm_api_key
    fallback = _heuristic_prior_conversation_decision(
        current_text,
        stored_profile,
        extracted_profile or {},
    )
    if not url:
        return {
            **fallback,
            "source": "heuristic_no_internal_llm",
            "raw": "",
        }

    messages = get_prompt_registry().render_messages(
        "prior_conversation_judge",
        stored_profile=json.dumps(stored_profile or {}, ensure_ascii=False),
        extracted_profile=json.dumps(extracted_profile or {}, ensure_ascii=False),
        current_text=current_text or "",
    )
    try:
        answer = await _post_chat_once(
            url,
            key,
            messages,
            model=model or settings.internal_llm_model,
            max_tokens=120,
            temperature=settings.internal_llm_route_temperature,
            chat_template_kwargs={"enable_thinking": False},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[PriorConversationJudge] internal_llm_failed fallback=%s error=%s",
            fallback.get("decision"),
            exc,
        )
        return {
            **fallback,
            "source": "heuristic_after_internal_llm_error",
            "raw": repr(exc),
        }

    parsed = _parse_prior_conversation_answer(answer)
    if not parsed:
        return {
            **fallback,
            "source": "heuristic_after_parse_error",
            "raw": (answer or "")[:300],
        }
    merged_profile = {**fallback.get("profile", {}), **parsed.get("profile", {})}
    return {
        "decision": parsed.get("decision") or fallback.get("decision"),
        "profile": merged_profile,
        "source": "internal_qwen",
        "raw": (answer or "")[:300],
        "heuristic_decision": fallback.get("decision"),
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
    temperature: Optional[float] = None,
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
                        "temperature": settings.internal_llm_temperature if temperature is None else temperature,
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


def _parse_pending_identity_reply_answer(answer: str) -> dict[str, Any]:
    stripped = _strip_reasoning_tags(answer or "").strip()
    if not stripped:
        return {"decision": "unclear", "profile": {}, "rationale": ""}
    match = stripped
    if "{" in stripped and "}" in stripped:
        match = stripped[stripped.find("{"): stripped.rfind("}") + 1]
    try:
        payload = json.loads(match)
    except json.JSONDecodeError:
        return {"decision": "unclear", "profile": {}, "rationale": ""}
    if not isinstance(payload, dict):
        return {"decision": "unclear", "profile": {}, "rationale": ""}
    decision = str(payload.get("decision") or "").strip().lower()
    if decision not in {"same_person", "different_person", "provided_identity", "rejected", "noise", "unclear"}:
        decision = "unclear"
    raw_profile = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}
    return {
        "decision": decision,
        "profile": _parse_identity_profile_answer(json.dumps(raw_profile or {}, ensure_ascii=False)),
        "rationale": str(payload.get("rationale") or "").strip(),
    }


def _parse_followup_recovery_answer(answer: str) -> dict[str, Any]:
    stripped = _strip_reasoning_tags(answer or "").strip()
    if not stripped:
        return {"is_medical_followup": False, "intent": "", "response": ""}
    match = stripped
    if "{" in stripped and "}" in stripped:
        match = stripped[stripped.find("{"): stripped.rfind("}") + 1]
    try:
        payload = json.loads(match)
    except json.JSONDecodeError:
        return {"is_medical_followup": False, "intent": "", "response": ""}
    if not isinstance(payload, dict):
        return {"is_medical_followup": False, "intent": "", "response": ""}
    return {
        "is_medical_followup": bool(payload.get("is_medical_followup")),
        "intent": str(payload.get("intent") or "").strip(),
        "response": str(payload.get("response") or "").strip(),
    }


def _parse_route_classifier_answer(answer: str) -> dict[str, Any]:
    stripped = _strip_reasoning_tags(answer or "").strip()
    if not stripped:
        return {"usable": False}
    match = stripped
    if "{" in stripped and "}" in stripped:
        match = stripped[stripped.find("{"): stripped.rfind("}") + 1]
    try:
        payload = json.loads(match)
    except json.JSONDecodeError:
        return {"usable": False}
    if not isinstance(payload, dict):
        return {"usable": False}

    route_label = str(payload.get("route_label") or "").strip()
    mode = str(payload.get("mode") or "").strip().upper()
    intent = str(payload.get("intent") or "").strip()
    raw_tasks = payload.get("task_types") or []
    task_types = [str(task).strip() for task in raw_tasks if str(task).strip()] if isinstance(raw_tasks, list) else []
    allowed_routes = {
        "ocr_capture",
        "ocr_result",
        "drug_identification",
        "meal_medication_prep",
        "after_meal_medication",
        "medication_record",
        "medication_taken_recall",
        "medication_safety_query",
        "supplement_query",
        "profile_recall",
        "lifestyle_memory",
        "non_actionable_ack",
        "noise_fragment",
        "smalltalk",
        "emergency",
        "unknown",
    }
    allowed_modes = {"MEMORY_ONLY", "TOOL_FIRST", "FRONTIER_FIRST", "ASK_USER_CLARIFY"}
    allowed_intents = {
        "medication_query",
        "supplement_query",
        "drug_identification",
        "smalltalk",
        "emergency",
        "unknown",
    }
    allowed_tasks = {
        "request_ocr",
        "dur_check",
        "supplement_lookup",
        "search_history",
        "hira_lookup",
        "dur_product_info",
        "llm_judge_verify",
    }
    if route_label not in allowed_routes or mode not in allowed_modes or intent not in allowed_intents:
        return {"usable": False}
    return {
        "usable": True,
        "route_label": route_label,
        "mode": mode,
        "intent": intent,
        "task_types": [task for task in task_types if task in allowed_tasks],
        "rationale": str(payload.get("rationale") or "").strip(),
    }


def _parse_prior_conversation_answer(answer: str) -> Optional[dict[str, Any]]:
    raw = (answer or "").strip()
    if not raw:
        return None
    match = None
    if "{" in raw and "}" in raw:
        match = raw[raw.find("{"): raw.rfind("}") + 1]
    try:
        payload = json.loads(match or raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    decision = str(payload.get("decision") or "").strip().lower()
    allowed = {"new_user", "returning_match", "provide_identity", "unclear"}
    if decision not in allowed:
        return None
    nested = payload.get("profile")
    if isinstance(nested, dict):
        profile = _parse_identity_profile_answer(json.dumps(nested, ensure_ascii=False))
    else:
        profile = {}
    return {"decision": decision, "profile": profile}


def _heuristic_prior_conversation_decision(
    text: str,
    stored_profile: dict[str, Any],
    extracted_profile: dict[str, Any],
) -> dict[str, Any]:
    """Fallback when internal LLM is off; never loops on the same prompt."""
    merged = {**extracted_profile, **{k: v for k, v in _heuristic_identity_extract(text).items() if v}}
    name = str(merged.get("name") or "").strip()
    age = str(merged.get("age") or "").strip()
    gender = str(merged.get("gender") or "").strip()
    has_core = bool(name and (age or gender))
    stored_name = str(stored_profile.get("name") or "").strip()

    stripped = (text or "").strip().lower()
    negative = any(
        token in stripped
        for token in ("아니", "없", "처음", "모르", "첫", "본 적 없", "대화한 적 없")
    )
    affirmative = any(token in stripped for token in ("네", "예", "맞아", "맞습니다", "응", "그래", "본인", "있어"))

    if has_core:
        return {"decision": "provide_identity", "profile": merged}
    if stored_name and name and name == stored_name:
        return {"decision": "returning_match", "profile": merged}
    if negative:
        return {"decision": "new_user", "profile": merged}
    if affirmative and not name:
        return {"decision": "new_user", "profile": merged}
    if affirmative and name:
        return {"decision": "returning_match", "profile": merged}
    return {"decision": "unclear", "profile": merged}


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
    name = _normalize_korean_person_name(str(payload.get("name") or "").strip())
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


def _normalize_korean_person_name(name: str) -> str:
    if len(name) >= 3 and name.endswith(("이가", "이는", "이야")):
        return name[:-2]
    if len(name) >= 4 and name[-1:] in {"가", "은", "는", "야"}:
        return name[:-1]
    return name


def _parse_ocr_medication_candidates_answer(answer: str) -> dict[str, Any]:
    raw = (answer or "").strip()
    if not raw:
        return {"medications": [], "clarification_question": ""}
    match = None
    if "{" in raw and "}" in raw:
        match = raw[raw.find("{"): raw.rfind("}") + 1]
    try:
        payload = json.loads(match or raw)
    except json.JSONDecodeError:
        return {"medications": [], "clarification_question": ""}
    if not isinstance(payload, dict):
        return {"medications": [], "clarification_question": ""}
    medications: list[dict[str, str]] = []
    for item in payload.get("medications") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        medications.append(
            {
                "name": name,
                "dosage": str(item.get("dosage") or "").strip(),
                "frequency": str(item.get("frequency") or "").strip(),
                "timing": str(item.get("timing") or "").strip(),
                "purpose_or_symptom": str(item.get("purpose_or_symptom") or "").strip(),
            }
        )
    return {
        "medications": medications[:8],
        "clarification_question": str(payload.get("clarification_question") or "").strip(),
    }


def _heuristic_identity_extract(text: str) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    if not text:
        return profile

    name_patterns = [
        r"(?:제\s*이름은|이름은)\s*([가-힣]{2,5}?)(?:이고|고|입니다|이에요|예요|,|\s|$)",
        r"(?:저는|나는|난)\s*([가-힣]{2,5}?)(?:이고|고|입니다|이에요|예요|,|\s|$)",
        r"^\s*([가-힣]{2,5})\s*(?:남자|남성|여자|여성)",
        r"^\s*([가-힣]{2,5})\s*,?\s*(?:\d{1,3}|[가-힣]{2,8})\s*(?:살|세)",
        r"(?:^|\s)([가-힣]{2,5})\s*,?\s*(?:\d{1,3}|[가-힣]{2,8})\s*(?:살|세)\s*(?:남자|남성|여자|여성)?",
        r"(?:^|\s)([가-힣]{2,5})\s*(?:남자|남성|여자|여성)\s*,?\s*\d{1,3}\s*(?:살|세)?",
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
    else:
        korean_age_match = re.search(r"([가-힣]{2,8})\s*(?:살|세)", text)
        if korean_age_match:
            age = _parse_korean_age(korean_age_match.group(1))
            if age:
                profile["age"] = str(age)
    if "남자" in text or "남성" in text:
        profile["gender"] = "남성"
    elif "여자" in text or "여성" in text:
        profile["gender"] = "여성"
    conditions = [
        token
        for token in ("고혈압", "당뇨", "천식", "통풍", "신장질환", "간질환", "심장질환")
        if token in text
    ]
    if conditions:
        profile["conditions"] = conditions
    return profile


def _parse_korean_age(text: str) -> int:
    compact = re.sub(r"\s+", "", text or "")
    direct = {
        "스무": 20,
        "스물": 20,
        "서른": 30,
        "마흔": 40,
        "쉰": 50,
        "예순": 60,
        "일흔": 70,
        "여든": 80,
        "아흔": 90,
    }
    if compact in direct:
        return direct[compact]
    tens = {
        "스물": 20,
        "서른": 30,
        "마흔": 40,
        "쉰": 50,
        "예순": 60,
        "일흔": 70,
        "여든": 80,
        "아흔": 90,
    }
    ones = {
        "한": 1,
        "하나": 1,
        "두": 2,
        "둘": 2,
        "세": 3,
        "셋": 3,
        "네": 4,
        "넷": 4,
        "다섯": 5,
        "여섯": 6,
        "일곱": 7,
        "여덟": 8,
        "아홉": 9,
    }
    for ten_text, ten_value in tens.items():
        if compact.startswith(ten_text):
            rest = compact[len(ten_text):]
            if not rest:
                return ten_value
            if rest in ones:
                return ten_value + ones[rest]
    return 0


def _looks_like_non_name_identity_candidate(value: str) -> bool:
    return value in {
        "고혈압",
        "당뇨",
        "천식",
        "신장질환",
        "간질환",
        "심장질환",
        "임신",
        "남자고",
        "여자고",
        "남성이고",
        "여성이고",
        "딸",
        "아들",
        "보호자",
        "가족",
        "엄마",
        "아빠",
        "아버지",
        "어머니",
    }


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
    temperature: Optional[float] = None,
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
                "temperature": settings.internal_llm_temperature if temperature is None else temperature,
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
