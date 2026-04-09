"""프롬프트 조립 및 내부/외부 LLM 호출."""
from typing import Optional

import httpx

from app.core.config import settings

SYSTEM_PROMPT_RECOGNITION = """당신은 복약 상담을 보조하는 AI입니다.
- 사용자의 질문에 대해 현재 복용 약물 정보와 DUR 주의사항을 바탕으로 답변합니다.
- 처방을 대체하거나 진단을 내리지 않습니다.
- 답변 끝에 "정확한 판단은 의사·약사 상담이 필요합니다"를 포함하세요.
- 짧고 읽기 쉬운 문장으로, 고령 사용자도 이해하기 쉽게 작성하세요."""


def build_user_prompt(query_text: str, llm_doc: str) -> str:
    """질의와 LLM용 문서를 합쳐 유저 메시지 생성."""
    return f"""다음은 사용자 질문과 현재 복용 약물·주의사항 요약입니다.

[복용 약물 및 주의사항]
{llm_doc}

[사용자 질문]
{query_text}

위 정보만 사용해 친절하고 안전하게 답변해 주세요."""


async def call_internal_llm(
    query_text: str,
    llm_doc: str,
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """내부 LLM(Qwen, EXAONE 등) 호출. 미설정 시 목업 응답."""
    url = api_url or settings.internal_llm_api_url
    key = api_key or settings.internal_llm_api_key
    if not url or not key:
        return "(내부 LLM 미설정) 녹용은 일반적으로 고혈압 약과 함께 드셔도 되는 경우가 많습니다. 다만 개인에 따라 다를 수 있으니, 약사나 의사에게 한 번 여쭤보시는 것이 좋습니다. 정확한 판단은 의사·약사 상담이 필요합니다."
    user = build_user_prompt(query_text, llm_doc)
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "qwen",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_RECOGNITION},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 512,
            },
        )
        r.raise_for_status()
        data = r.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""


async def call_external_llm(
    censored_payload: str,
    *,
    provider: str = "openai",
) -> str:
    """외부 프론티어 모델(Gemini, GPT-4, Claude) 호출. 검열된 페이로드만 전달."""
    if provider == "openai" and settings.openai_api_key:
        url = "https://api.openai.com/v1/chat/completions"
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                url,
                headers={"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_RECOGNITION},
                        {"role": "user", "content": censored_payload},
                    ],
                    "max_tokens": 512,
                },
            )
            r.raise_for_status()
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    # 목업
    return censored_payload[:200] + "\n\n(외부 LLM 미설정 또는 동일 응답)"
