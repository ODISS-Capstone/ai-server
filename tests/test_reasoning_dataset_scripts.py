"""Reasoning fine-tuning dataset script tests."""
import json
from pathlib import Path

import pytest

from scripts.generate_reasoning_dataset import (
    build_generation_prompt,
    load_tool_names,
    validate_sample,
)
from scripts.train_qwen_reasoning_lora import load_jsonl, render_chat_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_PATH = PROJECT_ROOT / "app" / "prompts" / "llm_tools.json"
SEED_DATASET = PROJECT_ROOT / "data" / "fine_tuning" / "qwen_reasoning_samples.jsonl"


def test_seed_dataset_records_are_valid_tool_call_samples():
    tool_names = load_tool_names(TOOLS_PATH)
    rows = load_jsonl(SEED_DATASET)

    assert len(rows) >= 5
    for row in rows:
        validate_sample(row, tool_names)
        assert row["messages"][0]["role"] == "system"
        assert any(message.get("tool_calls") for message in row["messages"])
        assistant_messages = [m for m in row["messages"] if m.get("role") == "assistant"]
        assert all(m["content"].strip().startswith("<think>") for m in assistant_messages)
        assert all("</think>" in m["content"] for m in assistant_messages)
        assert row["messages"][-1]["role"] == "assistant"
        assert "정확한 판단은 의사·약사 상담이 필요합니다" in row["messages"][-1]["content"]
        assert row["metadata"]["format"] == "qwen3.5_think_tool_calling"


def test_generation_prompt_mentions_available_tools():
    tool_names = load_tool_names(TOOLS_PATH)
    prompt = json.loads(build_generation_prompt(0, tool_names))

    assert "available_tools" in prompt
    assert "Tool_Check_DUR_Combination_Contraindication" in prompt["available_tools"]
    assert prompt["scenario_seed"]["target_tools"]
    assert "Every assistant content must start" in " ".join(prompt["constraints"])


def test_validate_sample_rejects_unknown_tool():
    sample = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "tool_call_id": "x", "name": "Tool_X", "content": "{}"},
            {"role": "assistant", "content": "answer"},
        ],
        "expected_tools": ["Tool_Not_Real"],
    }

    with pytest.raises(ValueError, match="unknown expected_tools"):
        validate_sample(sample, ["Tool_Real"])


def test_render_chat_text_uses_tokenizer_template():
    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is False
            return "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)

    text = render_chat_text(
        FakeTokenizer(),
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        },
    )

    assert "system: sys" in text
    assert "assistant: answer" in text
