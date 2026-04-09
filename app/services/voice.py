"""STT: 음성 파일 → 질문 텍스트 (Clova STT 또는 Whisper)."""
from app.core.config import settings
from app.schemas.voice import SttResponse


async def run_stt(audio_bytes: bytes, content_type: str = "audio/wav") -> SttResponse:
    """
    음성 바이트를 텍스트로 변환.
    Clova STT 미설정 시 목업 반환.
    """
    if not settings.clova_stt_client_id or not settings.clova_stt_client_secret:
        return SttResponse(
            text="(STT API 미설정) 고혈압 약 이거 먹고 있는데, 혹시 녹용 먹으면 안 될까?",
            message="(목업: Clova STT 연동 필요)",
        )
    # TODO: Clova STT API 호출
    # https://api.ncloud-docs.com/docs/ai-application-service-clova-speech-recognition
    return SttResponse(text="(Clova STT 연동 예정)", message="Not implemented")
