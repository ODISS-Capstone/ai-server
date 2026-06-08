"""서버 측 OCR: 이미지에서 처방전/약봉투 텍스트 추출 및 구조화."""
import base64
import json
import logging
import re
from typing import Optional

import httpx

from app.core.config import settings
from app.schemas.ocr import MedicationItem, OcrResponse

logger = logging.getLogger(__name__)


def _parse_medications_from_text(raw_text: str) -> list[MedicationItem]:
    """추출 텍스트에서 약품명·용량·복용법 패턴을 파싱해 MedicationItem 리스트 생성."""
    items: list[MedicationItem] = []
    lines = [s.strip() for s in raw_text.splitlines() if s.strip()]
    # 예: "삼진디아제팜정 2mg (0.5정, 1일 3회, 식후 30분 복용)"
    for line in lines:
        # 용량 패턴 (숫자 + mg, mL 등)
        strength = None
        strength_match = re.search(r"(\d+(?:\.\d+)?\s*(?:mg|mL|정|mg/g)?)", line, re.I)
        if strength_match:
            strength = strength_match.group(1).strip()
        # 복용량 (0.5정, 1.0정 등)
        dosage = None
        dosage_match = re.search(r"\(?\s*(\d+(?:\.\d+)?\s*정)", line)
        if dosage_match:
            dosage = dosage_match.group(1).strip()
        # 1일 N회
        frequency = None
        freq_match = re.search(r"1\s*일\s*\d+\s*회", line)
        if freq_match:
            frequency = freq_match.group(0)
        # 식후/식전 등
        timing = None
        timing_match = re.search(r"(?:식후|식전|아침|점심|저녁)[^\s,)]*", line)
        if timing_match:
            timing = timing_match.group(0).strip()
        # 약품명: 앞쪽 한글/영문 조합 (용량 앞까지)
        name = line
        if strength:
            name = line.split(strength)[0].strip().rstrip("()")
        elif dosage:
            name = line.split(dosage)[0].strip().rstrip("()")
        if len(name) > 100:
            name = name[:100]
        items.append(
            MedicationItem(
                name=name or "알 수 없음",
                strength=strength,
                dosage=dosage,
                frequency=frequency,
                timing=timing,
                raw_line=line,
            )
        )
    return items


async def run_ocr_image(image_bytes: bytes, content_type: str = "image/jpeg") -> OcrResponse:
    """
    이미지 바이트를 서버 OCR로 처리한다.

    우선순위:
    1. DeepSeek OCR 설정이 있으면 기존 DeepSeek 경로 사용
    2. GEMINI_API_KEY가 있으면 Gemini Vision OCR 사용
    3. 둘 다 없으면 실패 응답(샘플 반환 금지)
    """
    api_url = settings.deepseek_ocr_api_url
    api_key = settings.deepseek_ocr_api_key

    if not api_url or not api_key:
        return await _run_gemini_ocr_image(image_bytes, content_type)

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:{content_type};base64,{b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 이미지는 처방전 또는 약 봉투입니다. 이미지에 보이는 모든 약품 정보를 텍스트로 추출해 주세요. 각 약마다 한 줄씩, 형식: 약품명 용량 (복용량, 빈도, 시점)"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ]

    try:
        async with httpx.AsyncClient(timeout=settings.ocr_api_timeout_seconds) as client:
            resp = await client.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "deepseek-chat", "messages": messages, "max_tokens": 512},
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return OcrResponse(raw_text="", medications=[], success=False, message="No response from OCR API")
            raw_text = choices[0].get("message", {}).get("content", "") or ""
    except Exception as exc:  # noqa: BLE001 - callers can ask for recapture/fallback
        return OcrResponse(
            raw_text="",
            medications=[],
            success=False,
            message=f"OCR API timeout or error: {exc}",
        )

    medications = _parse_medications_from_text(raw_text)
    return OcrResponse(raw_text=raw_text, medications=medications)


async def _run_gemini_ocr_image(image_bytes: bytes, content_type: str) -> OcrResponse:
    api_key = settings.gemini_api_key
    if not api_key:
        return OcrResponse(
            raw_text="",
            medications=[],
            success=False,
            message="OCR API 미설정: GEMINI_API_KEY 또는 DeepSeek OCR 설정이 필요합니다.",
        )

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    prompt = (
        "이 이미지는 처방전, 약봉투, 약 설명서 또는 복약 관련 사진입니다. "
        "이미지에 보이는 약품명, 용량, 복용량, 복용 빈도, 복용 시점을 가능한 한 모두 한국어로 추출해 주세요. "
        "각 약은 한 줄로 출력하고, 확실하지 않은 글자는 추측하지 말고 '불확실'이라고 표시하세요. "
        "설명 문장이나 마크다운 없이 OCR 원문 중심으로만 답하세요."
    )
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": content_type or "image/jpeg",
                            "data": b64,
                        },
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 1024,
        },
    }
    models = [
        settings.gemini_ocr_model,
        *[
            model.strip()
            for model in settings.gemini_ocr_fallback_models.split(",")
            if model.strip()
        ],
    ]

    async with httpx.AsyncClient(timeout=settings.gemini_ocr_timeout_seconds) as client:
        for model in dict.fromkeys(models):
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={api_key}"
            )
            try:
                resp = await client.post(url, json=body)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[GeminiOCR] request failed model=%s error=%s", model, exc)
                continue
            if resp.status_code >= 400:
                logger.warning("[GeminiOCR] request failed model=%s status=%s body=%s", model, resp.status_code, resp.text[:500])
                continue
            data = resp.json()
            raw_text = _extract_gemini_text(data)
            medications = _parse_medications_from_text(raw_text)
            return OcrResponse(raw_text=raw_text, medications=medications, success=bool(raw_text))

    return OcrResponse(raw_text="", medications=[], success=False, message="Gemini OCR 처리 실패")


def _extract_gemini_text(payload: dict) -> str:
    texts: list[str] = []
    for candidate in payload.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = str(part.get("text") or "").strip()
            if text:
                texts.append(text)
    return "\n".join(texts).strip()
