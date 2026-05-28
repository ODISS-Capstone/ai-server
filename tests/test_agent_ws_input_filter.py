"""Server-side STT input filtering for low-latency WebSocket turns."""
from __future__ import annotations

from app.api.routes.agent_ws import _is_incomplete_or_noise_utterance


def test_incomplete_or_noise_utterances_are_filtered_before_pipeline() -> None:
    assert _is_incomplete_or_noise_utterance("어 나 이거 그") is True
    assert _is_incomplete_or_noise_utterance("딸깍") is True
    assert _is_incomplete_or_noise_utterance("어 어") is True
    assert _is_incomplete_or_noise_utterance("흐음") is True
    assert _is_incomplete_or_noise_utterance("네 어 그") is True
    assert _is_incomplete_or_noise_utterance("음 나 이거") is True
    assert _is_incomplete_or_noise_utterance("이 오디오는") is True
    assert _is_incomplete_or_noise_utterance("이 오는") is True
    assert _is_incomplete_or_noise_utterance("이 오디오는 한국어 음성입니다.") is True


def test_actionable_utterances_are_not_filtered() -> None:
    assert _is_incomplete_or_noise_utterance("네") is False
    assert _is_incomplete_or_noise_utterance("어딨어") is False
    assert _is_incomplete_or_noise_utterance("어디 있어") is False
    assert _is_incomplete_or_noise_utterance("뉴스 알려줘") is False
    assert _is_incomplete_or_noise_utterance("다시 말해줘") is False
    assert _is_incomplete_or_noise_utterance("처방전 사진 찍을 건데") is False
    assert _is_incomplete_or_noise_utterance("처방전 찍자") is False
    assert _is_incomplete_or_noise_utterance("어 서번 전 사진 찍을 건데 찍어 줄 수 있어?") is False
    assert _is_incomplete_or_noise_utterance("김영수 72세 남성") is False
    assert _is_incomplete_or_noise_utterance("약 어디 있어") is False
