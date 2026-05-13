"""Tests for task-family dataset split and fit gates."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_engine_datasets import (
    evaluate_delivery,
    evaluate_memory,
    evaluate_reasoning_alignment,
    evaluate_router,
    load_jsonl,
)
from scripts.split_reasoning_dataset import _strip_think, split_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_split_dataset_creates_router_memory_delivery_contract_sets(tmp_path):
    source = tmp_path / "reasoning.jsonl"
    sample = {
        "messages": [
            {"role": "system", "content": "tool_calls, function.arguments, tool_call_id"},
            {"role": "user", "content": "와파린이랑 아스피린 같이 먹어도 돼?"},
            {
                "role": "assistant",
                "content": "<think>의도 분석</think>",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "Tool_Check_DUR_Combination_Contraindication",
                            "arguments": "{\"item_name\":\"아스피린 장용정\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "name": "Tool_Check_DUR_Combination_Contraindication",
                "content": "{\"success\":true,\"items\":[{\"ITEM_NAME\":\"아스피린\"}]}",
            },
            {
                "role": "assistant",
                "content": (
                    "<think>결과 요약</think>\n"
                    "출혈 위험이 있을 수 있습니다. 정확한 판단은 의사·약사 상담이 필요합니다."
                ),
            },
        ],
        "expected_tools": ["Tool_Check_DUR_Combination_Contraindication"],
        "metadata": {"intent": "medication_query", "source": "seed"},
    }
    _write_jsonl(source, [sample])

    stats = split_dataset([source], tmp_path)
    assert stats["input_rows"] == 1
    assert stats["router_rows"] == 1
    assert stats["memory_rows"] == 1
    assert stats["delivery_rows"] == 1

    router_rows = load_jsonl(tmp_path / "qwen_router_samples.jsonl")
    memory_rows = load_jsonl(tmp_path / "qwen_memory_samples.jsonl")
    delivery_rows = load_jsonl(tmp_path / "qwen_delivery_samples.jsonl")

    assert evaluate_router(router_rows)["violations"] == []
    assert evaluate_memory(memory_rows)["violations"] == []
    assert evaluate_delivery(delivery_rows)["violations"] == []

    # Original monolithic sample should be flagged as runtime-mismatch training data.
    mismatch = evaluate_reasoning_alignment([sample])
    assert mismatch["runtime_mismatches"]


def test_split_dataset_strips_attributed_and_trailing_think_blocks():
    content = 'Answer. <think data-source="qwen">internal</think>\nNext.<THINK>unfinished'

    assert _strip_think(content) == "Answer. Next."


def test_dataset_evaluator_rejects_attributed_think_tags():
    rows = [
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": '<think data-source="qwen">internal</think>\nfinal',
                }
            ]
        }
    ]

    assert evaluate_delivery(rows)["violations"]
