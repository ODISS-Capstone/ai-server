"""Conversation pipeline tests for Judge review + local delivery."""
import asyncio

from app.core.config import settings
from app.engines.llm_judge import LLMJudgeEngine
from app.services.llm import call_local_delivery_llm
from app.services.prompt_registry import PromptRegistry


def test_prompt_registry_has_judge_and_local_delivery_prompts(tmp_path):
    registry = PromptRegistry(path=tmp_path / "missing.json")

    review_messages = registry.render_messages(
        "judge_final_review",
        original_query="같이 먹어도 돼?",
        core_message="출혈 위험이 있어 확인이 필요합니다.",
        additional_context_block="",
    )
    delivery_messages = registry.render_messages(
        "local_delivery",
        original_query="같이 먹어도 돼?",
        user_profile="{}",
        conversation_context="(없음)",
        reviewed_message="출혈 위험이 있어 확인이 필요합니다.",
    )

    assert "최종 검토" in review_messages[0]["content"]
    assert "로컬 대화 모델" in delivery_messages[0]["content"]


def test_judge_final_review_falls_back_without_openai_key(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None)
    engine = LLMJudgeEngine()

    result = asyncio.run(
        engine.review_final_answer(
            "와파린과 아스피린은 출혈 위험 확인이 필요합니다.",
            "이 두 약 같이 먹어도 돼?",
        )
    )

    assert result["reviewed"] is False
    assert result["reviewed_text"] == "와파린과 아스피린은 출혈 위험 확인이 필요합니다."


def test_local_delivery_falls_back_to_reviewed_message_without_internal_llm(monkeypatch):
    monkeypatch.setattr(settings, "internal_llm_api_url", None)
    monkeypatch.setattr(settings, "internal_llm_api_key", None)

    answer = asyncio.run(
        call_local_delivery_llm(
            original_query="이 두 약 같이 먹어도 돼?",
            reviewed_message="와파린과 아스피린은 함께 먹으면 출혈 위험이 커질 수 있습니다.",
        )
    )

    assert "와파린" in answer
    assert "아스피린" in answer
    assert "의사·약사 상담" in answer
