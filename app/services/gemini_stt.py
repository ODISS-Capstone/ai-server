"""Gemini 기반 서버 측 음성 전사.

모바일 APK에 Gemini API 키를 넣지 않기 위해, 단말은 짧은 음성 파일만
ai-server로 업로드하고 서버가 Gemini REST API를 호출한다.
"""
from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiSttError(RuntimeError):
    """Gemini STT 호출 실패."""


def _extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    texts: list[str] = []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = str(part.get("text") or "").strip()
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


async def transcribe_audio_with_gemini(
    audio_bytes: bytes,
    mime_type: str,
    language: str = "ko-KR",
) -> str:
    if not settings.gemini_api_key:
        raise GeminiSttError("GEMINI_API_KEY is not configured")
    if not audio_bytes:
        return ""

    prompt = (
        "You are a speech-to-text engine for an elderly Korean medication assistant. "
        f"Transcribe the audio in {language}. Return only the spoken text, without "
        "translation, explanation, timestamps, markdown, or punctuation embellishment. "
        "If no speech is present, return an empty string."
    )
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type or "audio/mp4",
                            "data": base64.b64encode(audio_bytes).decode("ascii"),
                        },
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 128,
        },
    }

    models = [
        settings.gemini_stt_model,
        *[
            model.strip()
            for model in settings.gemini_stt_fallback_models.split(",")
            if model.strip()
        ],
    ]
    async with httpx.AsyncClient(timeout=settings.gemini_stt_timeout_seconds) as client:
        last_error = ""
        for model in dict.fromkeys(models):
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={settings.gemini_api_key}"
            )
            response = await client.post(url, json=body)
            if response.status_code < 400:
                return _extract_text(response.json())
            last_error = f"{model}: {response.status_code}"
            logger.warning(
                "[GeminiSTT] request failed model=%s status=%s body=%s",
                model,
                response.status_code,
                response.text[:500],
            )

    raise GeminiSttError(f"Gemini STT failed: {last_error or 'unknown error'}")
