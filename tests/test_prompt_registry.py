"""Prompt registry rendering tests."""
import json

import pytest

from app.services.prompt_registry import PromptRegistry


def test_prompt_registry_renders_json_prompt(tmp_path):
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(
        json.dumps(
            {
                "prompts": {
                    "custom": {
                        "system": "System {name}",
                        "user": "Question {query}",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    registry = PromptRegistry(path=prompt_file, defaults={})
    messages = registry.render_messages("custom", name="ODISS", query="약 먹어도 돼?")

    assert messages == [
        {"role": "system", "content": "System ODISS"},
        {"role": "user", "content": "Question 약 먹어도 돼?"},
    ]


def test_prompt_registry_uses_defaults_when_file_is_missing(tmp_path):
    registry = PromptRegistry(path=tmp_path / "missing.json")

    user_prompt = registry.render_user(
        "main_answer",
        query_text="녹용 먹어도 돼?",
        llm_doc="혈압약 복용 중",
    )

    assert "[복용 약물 및 주의사항]" in user_prompt
    assert "혈압약 복용 중" in user_prompt


def test_default_prompts_include_demo_conversation_policy(tmp_path):
    registry = PromptRegistry(path=tmp_path / "missing.json")

    main_system = registry.render_system("main_answer")
    external_system = registry.render_system("external_review")
    judge_system = registry.render_system("judge_verify")
    review_system = registry.render_system("judge_final_review")
    delivery_system = registry.render_system("local_delivery")
    conflict_system = registry.render_system("identity_conflict_judge")

    assert "가입/프로필 회상" in main_system
    assert "나이와 무관하게" in main_system
    assert "만성질환자, 보호자, 복약 사용자라는 이유만으로 고령자라고 추정하지 않습니다" in main_system
    assert "사용자가 고령자라고 확인된 경우가 아니라면 '어르신'이라고 부르지 않습니다" in main_system
    assert "비의료 대화에는 상담 문구를 붙이지 않습니다" in main_system
    assert "데모 인물의 이름, 나이, 성별, 약 이름" in external_system
    assert "사용자와 복약 관리 대상자의 안전" in judge_system
    assert "저장된 사용자/복약 관리 대상자 프로필" in conflict_system
    assert "데모 인물의 이름, 나이, 약 이름" in review_system
    assert "<이름>님" in delivery_system
    assert "어르신" in delivery_system
    assert "검토 완료 문장에 '어르신'이 들어 있어도" in delivery_system
    assert "65세 이상" in delivery_system
    assert "데모 인물의 이름, 나이, 약 이름" in delivery_system
    assert "사용자님" in delivery_system
    identity_system = registry.render_system("identity_profile_extract")
    assert "대신 관리한다고 명시한 대상자" in identity_system
    assert "보호자 정보와 대상자 정보를 섞지 말고" in identity_system
    assert "김영수" not in delivery_system
    assert "데모 대화처럼" not in delivery_system
    assert "<think>" in delivery_system


def test_configured_prompts_keep_conversation_guardrails():
    registry = PromptRegistry()
    rendered = "\n\n".join(
        registry.render_system(key)
        for key in (
            "main_answer",
            "external_review",
            "judge_verify",
            "judge_final_review",
            "local_delivery",
            "identity_conflict_judge",
            "identity_profile_extract",
            "prior_conversation_judge",
        )
    )

    for forbidden in ("김영수", "김영식", "72세 김", "데모 대화처럼"):
        assert forbidden not in rendered
    for outdated in ("저장된 환자 프로필", "환자 신원정보", "환자 안전"):
        assert outdated not in rendered
    for required in (
        "나이와 무관하게",
        "고령자로 보지 않습니다",
        "사용자가 고령자라고 확인된 경우가 아니라면",
        "데모 인물의 이름, 나이, 성별, 약 이름",
        "사용자와 복약 관리 대상자의 안전",
        "저장된 사용자/복약 관리 대상자 프로필",
        "검토 완료 문장에 '어르신'이 들어 있어도",
        "보호자 정보와 대상자 정보를 섞지 말고",
        "의사·약사 상담 문구는",
        "<think>",
    ):
        assert required in rendered


def test_tool_prompt_does_not_overgeneralize_geriatric_caution():
    with open("app/prompts/llm_tools.json", "r", encoding="utf-8") as f:
        tools = json.load(f)["tools"]

    geriatric = next(
        tool["function"]["description"]
        for tool in tools
        if tool["function"]["name"] == "Tool_Check_DUR_Geriatric_Caution"
    )

    assert "65세 이상 사용자 또는 복약 관리 대상자" in geriatric
    assert "확인될 때만" in geriatric
    assert "노인 환자" not in geriatric


def test_prompt_registry_requires_template_variables(tmp_path):
    registry = PromptRegistry(
        path=tmp_path / "missing.json",
        defaults={"custom": {"system": "System", "user": "{required}"}},
    )

    with pytest.raises(KeyError, match="required"):
        registry.render_user("custom")
