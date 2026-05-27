"""Wake-word must not be treated as a medication name."""
from __future__ import annotations

from app.engines.memory import MemoryEngine
from app.engines.reasoning import ReasoningEngine
from app.engines.llm_judge import LLMJudgeEngine
from app.services.medication_extraction import (
    filter_drug_name_candidates,
    is_ocr_capture_request_text,
    is_wake_word_only,
    is_non_medication_token,
    strip_wake_words,
)


def test_strip_wake_words_removes_odiss():
    assert strip_wake_words("오디스, 내가 누구인지 말해봐.") == "내가 누구인지 말해봐"
    assert strip_wake_words("오티스, 혈압약 먹어도 돼?") == "혈압약 먹어도 돼"
    assert strip_wake_words("오 디 스, 혈압약 먹어도 돼?") == "혈압약 먹어도 돼"
    assert strip_wake_words("보디스, 혈압약 먹어도 돼?") == "혈압약 먹어도 돼"
    assert strip_wake_words("오디스") == ""
    assert strip_wake_words("오디 혈압약 먹어도 돼?") == "혈압약 먹어도 돼"
    assert strip_wake_words("야, 혈압약 먹어도 돼?") == "혈압약 먹어도 돼"
    assert strip_wake_words("먹어야 하는 약") == "먹어야 하는 약"
    assert strip_wake_words("오디오가 안 들려") == "오디오가 안 들려"
    assert is_non_medication_token("오디스")
    assert not is_wake_word_only("")
    assert not is_wake_word_only("?")
    assert is_wake_word_only("오디스?")
    assert is_wake_word_only("보리스")
    assert is_wake_word_only("오티스")
    assert is_wake_word_only("오지스?")
    assert is_wake_word_only("오 디 스")
    assert is_wake_word_only("오디")
    assert is_wake_word_only("오티즈")
    assert is_wake_word_only("보디스")
    assert is_wake_word_only("야")
    assert is_wake_word_only("들려?")
    assert is_wake_word_only("내 말 들려?")
    assert not is_wake_word_only("오디스 혈압약 먹어도 돼?")


def test_ocr_capture_request_detects_new_medication_package_language():
    assert is_ocr_capture_request_text("오디스 나 새약 받아왔어")
    assert is_ocr_capture_request_text("병원에서 약 타왔는데")
    assert is_ocr_capture_request_text("새 처방 받았어")
    assert is_ocr_capture_request_text("오늘 처방 나왔어")
    assert is_ocr_capture_request_text("약봉투를 카메라 앞에 보여줄게")
    assert not is_ocr_capture_request_text("사진에서 읽힌 약 이름 결과가 뭐야")


def test_reasoning_does_not_use_wake_word_as_drug_name():
    engine = ReasoningEngine(MemoryEngine(), LLMJudgeEngine())
    names = engine._extract_drug_names("오디스", {})
    assert names == []

    names = engine._extract_drug_names("오디스. 혈압약 먹어도 되나요?", {})
    assert "오디스" not in names


def test_memory_query_medications_ignore_wake_word():
    memory = MemoryEngine()
    assert memory._extract_query_medications("오디스") == []
    assert memory._extract_query_medications("오디스, 아스피린 먹어도 되나요?") == ["아스피린"]


def test_filter_drug_name_candidates():
    assert filter_drug_name_candidates(["오디스", "타이레놀정"]) == ["타이레놀정"]
