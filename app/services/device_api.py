"""단말기 전송용 서버 API: 스마트 스피커 등으로 최종 답변 전달."""
from typing import Any, Optional


async def send_to_device(
    device_id: str,
    text: str,
    tts_requested: bool = True,
    meta: Optional[dict[str, Any]] = None,
) -> bool:
    """
    최종 답변 텍스트를 지정 단말(스마트 스피커 등)로 전송.
    실제 구현 시 WebSocket 또는 디바이스별 Push API 호출.
    """
    # 목업: 로컬에서 저장만 하거나 외부 API 호출
    _ = device_id, text, tts_requested, meta
    return True
