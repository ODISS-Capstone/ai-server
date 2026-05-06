"""Generate Qwen reasoning-engine SFT data with the OpenAI Chat Completions API.

The output format is one JSON object per line:

{
  "messages": [...OpenAI-style chat messages including optional tool_calls...],
  "expected_tools": ["Tool_Check_DUR_..."],
  "metadata": {"intent": "...", "source": "openai_synthetic"}
}

In-context exemplars: the script reads validated samples from
``--exemplars-path`` (default: ``data/fine_tuning/qwen_reasoning_samples.jsonl``)
and embeds 1–3 of them inside the generation prompt so GPT can imitate the
exact JSON shape, longCOT pattern, and Korean tone.

Usage::

    # 1. fill .env with OPENAI_API_KEY
    OPENAI_API_KEY=sk-...
    OPENAI_DATASET_MODEL=gpt-5.5        # optional override

    # 2. generate
    python scripts/generate_reasoning_dataset.py \
        --count 100 \
        --output data/fine_tuning/qwen_reasoning_synthetic.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOLS_PATH = PROJECT_ROOT / "app" / "prompts" / "llm_tools.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "fine_tuning" / "qwen_reasoning_synthetic.jsonl"
DEFAULT_EXEMPLARS_PATH = PROJECT_ROOT / "data" / "fine_tuning" / "qwen_reasoning_samples.jsonl"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: only sets vars that aren't already in os.environ."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_ROOT / ".env")

REASONING_SYSTEM_PROMPT = (
    "당신은 ODISS의 Qwen reasoning-engine fine-tuning 데이터 생성기입니다. "
    "복약 상담 상황에서 어떤 공공데이터 tool을 호출해야 하는지와, "
    "tool 결과를 근거로 최종 답변하는 OpenAI chat-format SFT 샘플을 만드세요. "
    "런타임 system prompt에는 tool call API 양식을 넣고, 훈련용 assistant 메시지에는 "
    "<think>...</think> longCOT 블록을 넣으세요. "
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


def load_exemplars(path: Path) -> list[dict[str, Any]]:
    """Read validated SFT samples used as in-context examples."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _pick_exemplars(
    exemplars: list[dict[str, Any]],
    *,
    intent: str,
    k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Pick up to ``k`` exemplars, preferring matching intent."""
    if not exemplars or k <= 0:
        return []
    same_intent = [
        e for e in exemplars if e.get("metadata", {}).get("intent") == intent
    ]
    others = [
        e for e in exemplars if e.get("metadata", {}).get("intent") != intent
    ]
    rng.shuffle(same_intent)
    rng.shuffle(others)
    picked = (same_intent + others)[:k]
    return picked


def _supports_temperature(model: str) -> bool:
    """Return whether the Chat Completions request should include temperature.

    GPT-5 reasoning-family models reject legacy sampling controls such as
    ``temperature`` on this endpoint.  Keep the CLI option for older models,
    but omit the field automatically for GPT-5+ slugs.
    """
    normalized = model.lower().strip()
    return not normalized.startswith("gpt-5")


def build_generation_prompt(
    index: int,
    tool_names: list[str],
    *,
    exemplars: list[dict[str, Any]] | None = None,
    rng: random.Random | None = None,
    exemplar_count: int = 2,
) -> str:
    seed = SCENARIO_SEEDS[index % len(SCENARIO_SEEDS)]
    rng = rng or random.Random(index)
    chosen = _pick_exemplars(
        exemplars or [], intent=seed["intent"], k=exemplar_count, rng=rng
    )
    return json.dumps(
        {
            "task": "Create one Korean SFT sample for Qwen tool-calling reasoning.",
            "in_context_examples": chosen,
            "required_output_schema": {
                "messages": [
                    {
                        "role": "system",
                        "content": "ODISS reasoning engine system prompt with tool-use instructions",
                    },
                    {"role": "user", "content": "string"},
                    {
                        "role": "assistant",
                        "content": "<think>\n1. 의도: ...\n2. 입력 근거: ...\n3. 위험 후보: ...\n4. 필요한 API: ...\n5. tool 인자 결정: ...\n6. 답변 보류: ...\n</think>",
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
                        "content": "<think>\n1. tool 결과 요약: ...\n2. 근거 평가: ...\n3. 답변 전략: ...\n4. 안전 문구: ...\n</think>\nfinal Korean answer",
                    },
                ],
                "expected_tools": ["tool names used"],
                "metadata": {
                    "intent": seed["intent"],
                    "source": "openai_synthetic",
                    "risk": "low|medium|high",
                    "api_family": "dur|hira|health_supplement|mixed",
                    "format": "qwen3.5_longcot_tool_calling",
                },
            },
            "constraints": [
                "Use only tool names from available_tools.",
                "Use Korean for user and assistant text.",
                "Mirror the exact JSON shape, message ordering, longCOT structure, and Korean tone of in_context_examples; treat them as canonical reference samples.",
                "Do NOT copy the user query, drug names, or numeric facts from in_context_examples. Invent a new realistic scenario consistent with scenario_seed.",
                "The first system message must instruct the model to reason internally, choose public-data API tools when needed, call tools before answering, and answer safely from tool results.",
                "The first system message must explicitly include the tool call API format: assistant content is empty when calling tools, tool_calls array is used, each call has id/type/function.name/function.arguments, function.arguments is a JSON string, and tool results arrive as role='tool' with tool_call_id/name/content.",
                "Assistant messages must include longCOT <think>...</think> blocks for training data.",
                "When tool use is needed, the assistant message should include a longCOT <think> block plus a valid tool_calls array.",
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
    exemplars: list[dict[str, Any]],
    rng: random.Random,
    exemplar_count: int,
    temperature: float,
) -> dict[str, Any]:
    user_prompt = build_generation_prompt(
        index, tool_names,
        exemplars=exemplars, rng=rng, exemplar_count=exemplar_count,
    )
    request_body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    if _supports_temperature(model):
        request_body["temperature"] = temperature

    response = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=request_body,
        timeout=120.0,
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
    required_system_terms = ["tool_calls", "function.arguments", "tool_call_id"]
    missing_terms = [term for term in required_system_terms if term not in system_content]
    if missing_terms:
        raise ValueError(
            "system prompt must include tool call API format terms: "
            + ", ".join(missing_terms)
        )

    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_messages:
        raise ValueError("sample must contain assistant messages")
    for idx, message in enumerate(assistant_messages, start=1):
        content = message.get("content", "")
        if not _has_think_block(content):
            raise ValueError(f"assistant message {idx} must include <think>...</think>")


def _has_think_block(content: str) -> bool:
    stripped = content.strip()
    return stripped.startswith("<think>") and "</think>" in stripped


async def generate_dataset(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required. Set it in ai-server/.env (preferred) "
            "or export OPENAI_API_KEY=... before running, or pass --api-key."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tool_names = load_tool_names(Path(args.tools_path))
    exemplars = load_exemplars(Path(args.exemplars_path))
    rng = random.Random(args.seed)

    if args.dry_run:
        sample_prompt = build_generation_prompt(
            0, tool_names,
            exemplars=exemplars, rng=rng,
            exemplar_count=args.exemplars,
        )
        print("# dry-run: example prompt that would be sent to OpenAI")
        print(f"# model: {args.model}")
        print(sample_prompt[:4000])
        print("...")
        print(f"# exemplars loaded: {len(exemplars)}")
        return

    failed = 0
    async with httpx.AsyncClient() as client:
        with output_path.open("a", encoding="utf-8") as f:
            for idx in range(args.count):
                try:
                    sample = await generate_one(
                        client=client,
                        api_key=api_key,
                        model=args.model,
                        index=idx,
                        tool_names=tool_names,
                        exemplars=exemplars,
                        rng=rng,
                        exemplar_count=args.exemplars,
                        temperature=args.temperature,
                    )
                except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                    failed += 1
                    detail = ""
                    if isinstance(exc, httpx.HTTPStatusError):
                        detail = f" | response={exc.response.text[:1000]}"
                    print(
                        f"[skip {idx + 1}/{args.count}] "
                        f"{type(exc).__name__}: {exc}{detail}"
                    )
                    if failed >= args.max_failures:
                        raise RuntimeError(
                            f"aborted after {failed} consecutive/total failures"
                        ) from exc
                    continue
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f.flush()
                print(f"wrote sample {idx + 1}/{args.count} -> {output_path}")
    print(f"done. wrote {args.count - failed} samples (skipped {failed})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20,
                        help="how many samples to generate")
    parser.add_argument("--model",
                        default=os.getenv("OPENAI_DATASET_MODEL", "gpt-5.5"),
                        help="OpenAI model id (env: OPENAI_DATASET_MODEL)")
    parser.add_argument("--api-key", default=None,
                        help="overrides OPENAI_API_KEY from env/.env")
    parser.add_argument("--tools-path", default=str(DEFAULT_TOOLS_PATH))
    parser.add_argument("--exemplars-path", default=str(DEFAULT_EXEMPLARS_PATH),
                        help="JSONL file with hand-written reference samples for in-context learning")
    parser.add_argument("--exemplars", type=int, default=2,
                        help="number of in-context examples per generation request (0 disables)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260506,
                        help="RNG seed for exemplar shuffling")
    parser.add_argument("--max-failures", type=int, default=10,
                        help="abort run after this many failed generations")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--dry-run", action="store_true",
                        help="print the first generation prompt without calling OpenAI")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(generate_dataset(parse_args()))
