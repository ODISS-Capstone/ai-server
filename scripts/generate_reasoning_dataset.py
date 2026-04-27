"""Generate Qwen reasoning-engine SFT data with the OpenAI Chat Completions API.

The output format is one JSON object per line:

{
  "messages": [...OpenAI-style chat messages including optional tool_calls...],
  "expected_tools": ["Tool_Check_DUR_..."],
  "metadata": {"intent": "...", "source": "openai_synthetic"}
}

Usage:
    OPENAI_API_KEY=... python scripts/generate_reasoning_dataset.py \
        --count 50 \
        --output data/fine_tuning/qwen_reasoning_synthetic.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOLS_PATH = PROJECT_ROOT / "app" / "prompts" / "llm_tools.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "fine_tuning" / "qwen_reasoning_synthetic.jsonl"

REASONING_SYSTEM_PROMPT = (
    "당신은 ODISS의 Qwen reasoning-engine fine-tuning 데이터 생성기입니다. "
    "복약 상담 상황에서 어떤 공공데이터 tool을 호출해야 하는지와, "
    "tool 결과를 근거로 최종 답변하는 OpenAI chat-format SFT 샘플을 만드세요. "
    "추론 및 API 호출 지시는 assistant 출력이 아니라 system 메시지에 넣으세요. "
    "반드시 JSON 객체 하나만 출력하세요."
)

SCENARIO_SEEDS = [
    {
        "intent": "medication_query",
        "patient": "72세, 와파린 복용 중",
        "query": "새로 받은 진통제랑 같이 먹어도 되는지 묻는 상황",
        "target_tools": [
            "Tool_Check_DUR_Combination_Contraindication",
            "Tool_Check_DUR_Geriatric_Caution",
        ],
    },
    {
        "intent": "drug_identification",
        "patient": "약 이름을 모르는 사용자",
        "query": "알약 색상, 모양, 각인만 보고 식별하려는 상황",
        "target_tools": ["Tool_Get_Drug_Identification"],
    },
    {
        "intent": "supplement_query",
        "patient": "혈압약 복용 중인 68세 사용자",
        "query": "홍삼, 오메가3, 루테인 등 건강기능식품을 같이 먹어도 되는지 묻는 상황",
        "target_tools": [
            "Tool_Get_Health_Supplement_Detail",
            "Tool_Search_Health_Supplement_List",
        ],
    },
    {
        "intent": "duration_or_dosage",
        "patient": "수면제 또는 진통제를 오래 복용 중인 고령 사용자",
        "query": "복용 기간 또는 용량이 안전한지 묻는 상황",
        "target_tools": [
            "Tool_Check_DUR_Duration_Caution",
            "Tool_Check_DUR_Dosage_Caution",
            "Tool_Check_DUR_Geriatric_Caution",
        ],
    },
    {
        "intent": "pregnancy_or_age_specific",
        "patient": "임신 가능성이 있거나 소아 보호자인 사용자",
        "query": "특정 연령/임신 상태에서 복용 가능한지 묻는 상황",
        "target_tools": [
            "Tool_Check_DUR_Age_Specific_Contraindication",
            "Tool_Check_DUR_Pregnancy_Contraindication",
        ],
    },
]


def load_tool_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [tool["function"]["name"] for tool in data.get("tools", [])]


def build_generation_prompt(index: int, tool_names: list[str]) -> str:
    seed = SCENARIO_SEEDS[index % len(SCENARIO_SEEDS)]
    return json.dumps(
        {
            "task": "Create one Korean SFT sample for Qwen tool-calling reasoning.",
            "required_output_schema": {
                "messages": [
                    {
                        "role": "system",
                        "content": "ODISS reasoning engine system prompt with tool-use instructions",
                    },
                    {"role": "user", "content": "string"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_001",
                                "type": "function",
                                "function": {
                                    "name": "one of available_tools",
                                    "arguments": "JSON string",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_001",
                        "name": "same as tool call",
                        "content": "JSON string with success/items fields",
                    },
                    {
                        "role": "assistant",
                        "content": "final Korean answer",
                    },
                ],
                "expected_tools": ["tool names used"],
                "metadata": {
                    "intent": seed["intent"],
                    "source": "openai_synthetic",
                    "risk": "low|medium|high",
                    "api_family": "dur|hira|health_supplement|mixed",
                    "format": "qwen3.5_system_tool_calling",
                },
            },
            "constraints": [
                "Use only tool names from available_tools.",
                "Use Korean for user and assistant text.",
                "Do not include <think> blocks in assistant messages.",
                "The first system message must instruct the model to reason internally, choose public-data API tools when needed, call tools before answering, and answer safely from tool results.",
                "When tool use is needed, the assistant tool-call message should have empty content and a valid tool_calls array.",
                "The final answer must be cautious and include '정확한 판단은 의사·약사 상담이 필요합니다.'",
                "Make tool result content realistic but synthetic; do not include real personal data.",
                "Return only valid JSON, no markdown.",
            ],
            "scenario_seed": seed,
            "available_tools": tool_names,
        },
        ensure_ascii=False,
    )


async def generate_one(
    *,
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    index: int,
    tool_names: list[str],
) -> dict[str, Any]:
    response = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": build_generation_prompt(index, tool_names)},
            ],
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        },
        timeout=90.0,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    sample = json.loads(content)
    validate_sample(sample, tool_names)
    return sample


def validate_sample(sample: dict[str, Any], tool_names: list[str]) -> None:
    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 5:
        raise ValueError("sample.messages must contain at least 5 chat turns")

    expected_tools = sample.get("expected_tools")
    if not isinstance(expected_tools, list) or not expected_tools:
        raise ValueError("sample.expected_tools must be a non-empty list")

    unknown = set(expected_tools) - set(tool_names)
    if unknown:
        raise ValueError(f"unknown expected_tools: {sorted(unknown)}")

    final_message = messages[-1]
    if final_message.get("role") != "assistant" or not final_message.get("content"):
        raise ValueError("last message must be a non-empty assistant answer")

    system_message = messages[0]
    if system_message.get("role") != "system" or not system_message.get("content"):
        raise ValueError("first message must be a non-empty system prompt")
    system_content = system_message["content"]
    if "tool" not in system_content.lower() and "API" not in system_content:
        raise ValueError("system prompt must include tool/API usage instructions")

    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_messages:
        raise ValueError("sample must contain assistant messages")
    for idx, message in enumerate(assistant_messages, start=1):
        content = message.get("content", "")
        if "<think>" in content or "</think>" in content:
            raise ValueError(f"assistant message {idx} must not include <think> blocks")


async def generate_dataset(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tool_names = load_tool_names(Path(args.tools_path))

    async with httpx.AsyncClient() as client:
        with output_path.open("a", encoding="utf-8") as f:
            for idx in range(args.count):
                sample = await generate_one(
                    client=client,
                    api_key=api_key,
                    model=args.model,
                    index=idx,
                    tool_names=tool_names,
                )
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f.flush()
                print(f"wrote sample {idx + 1}/{args.count} -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--tools-path", default=str(DEFAULT_TOOLS_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(generate_dataset(parse_args()))
