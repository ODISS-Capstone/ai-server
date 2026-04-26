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


def test_prompt_registry_requires_template_variables(tmp_path):
    registry = PromptRegistry(
        path=tmp_path / "missing.json",
        defaults={"custom": {"system": "System", "user": "{required}"}},
    )

    with pytest.raises(KeyError, match="required"):
        registry.render_user("custom")
