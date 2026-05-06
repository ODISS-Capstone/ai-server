"""Split monolithic ODISS reasoning data into task-family datasets.

Input samples are OpenAI-style chat JSONL records (the existing
``qwen_reasoning_samples.jsonl`` / ``qwen_reasoning_synthetic.jsonl`` format).
This script derives three specialized datasets:

1. router   : route decision (`tool_first` / `frontier_first` / ...)
2. memory   : OCR normalization + evidence selection summary
3. delivery : elder-facing response rendering (no ``<think>``)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    PROJECT_ROOT / "data" / "fine_tuning" / "qwen_reasoning_samples.jsonl",
    PROJECT_ROOT / "data" / "fine_tuning" / "qwen_reasoning_synthetic.jsonl",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "fine_tuning"


def _strip_think(content: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", content or "", flags=re.DOTALL).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _collect_medication_candidates(sample: dict[str, Any]) -> list[str]:
    meds: list[str] = []
    for message in sample.get("messages", []):
        if message.get("role") == "assistant":
            for call in message.get("tool_calls", []) or []:
                function = call.get("function", {})
                args_raw = function.get("arguments", "{}")
                try:
                    args = json.loads(args_raw)
                except Exception:
                    args = {}
                item_name = args.get("item_name") or args.get("product_name")
                if item_name:
                    meds.append(str(item_name).strip())
        elif message.get("role") == "tool":
            content = str(message.get("content", ""))
            for key in ("ITEM_NAME", "itemName", "name"):
                for m in re.findall(rf'"{key}"\s*:\s*"([^"]+)"', content):
                    meds.append(m.strip())
    normalized: list[str] = []
    for med in meds:
        if med and med not in normalized:
            normalized.append(med)
    return normalized[:6]


def _first_user_text(sample: dict[str, Any]) -> str:
    for message in sample.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def _last_assistant_text(sample: dict[str, Any]) -> str:
    for message in reversed(sample.get("messages", [])):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


def _build_router_record(sample: dict[str, Any], sample_id: str) -> dict[str, Any]:
    expected_tools = sample.get("expected_tools", []) or []
    metadata = sample.get("metadata", {}) or {}
    intent = str(metadata.get("intent", "unknown"))
    route_mode = "tool_first" if expected_tools else "frontier_first"
    if intent == "smalltalk":
        route_mode = "memory_only"

    target = {
        "route_mode": route_mode,
        "intent": intent,
        "required_tasks": expected_tools,
        "rationale": "tool evidence needed before user-facing answer"
        if expected_tools
        else "no deterministic tool requirement",
    }
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "당신은 ODISS Reasoning Router입니다. 사용자 입력과 현재 맥락을 읽고 "
                    "route_mode(tool_first/frontier_first/memory_only/ask_user_clarify), "
                    "intent, required_tasks를 JSON으로만 출력하세요."
                ),
            },
            {"role": "user", "content": _first_user_text(sample)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "metadata": {
            "task_family": "router",
            "source_sample_id": sample_id,
            "format": "odiss_router_json",
            "source": metadata.get("source", "unknown"),
        },
    }


def _build_memory_record(sample: dict[str, Any], sample_id: str) -> dict[str, Any]:
    metadata = sample.get("metadata", {}) or {}
    meds = _collect_medication_candidates(sample)
    target = {
        "normalized_query": " ".join(_first_user_text(sample).split()),
        "normalized_medications": meds,
        "dur_searchable": bool(meds),
        "selected_artifacts": [
            {"category": "structured_memory", "reason": "medication context"},
            {"category": "prescriptions", "reason": "latest OCR prescription trace"},
        ],
        "summary": "핵심 복약 이력만 선택해 요약하고, 근거가 부족하면 fallback 검색을 사용한다.",
    }
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "당신은 ODISS Memory Engine입니다. OCR 오탈자 정규화, DUR 검색 가능성 판단, "
                    "필요한 메모리 아티팩트만 선택한 뒤 JSON으로 요약하세요."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": _first_user_text(sample),
                        "ocr_candidates": meds,
                        "expected_tools": sample.get("expected_tools", []),
                    },
                    ensure_ascii=False,
                ),
            },
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "metadata": {
            "task_family": "memory",
            "source_sample_id": sample_id,
            "format": "odiss_memory_json",
            "source": metadata.get("source", "unknown"),
        },
    }


def _build_delivery_record(sample: dict[str, Any], sample_id: str) -> dict[str, Any]:
    metadata = sample.get("metadata", {}) or {}
    final_text = _strip_think(_last_assistant_text(sample))
    if "정확한 판단은 의사·약사 상담이 필요합니다" not in final_text:
        final_text = (
            f"{final_text.rstrip('.')}."
            " 정확한 판단은 의사·약사 상담이 필요합니다."
        ).strip()
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "당신은 ODISS 대화 엔진의 최종 발화 생성기입니다. "
                    "주어진 사실 요약을 어르신이 이해하기 쉬운 짧은 존댓말로 바꾸세요. "
                    "런타임 출력에는 <think> 토큰을 절대 포함하지 마세요."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": _first_user_text(sample),
                        "reasoning_summary": final_text,
                        "intent": (metadata.get("intent") or "unknown"),
                    },
                    ensure_ascii=False,
                ),
            },
            {"role": "assistant", "content": final_text},
        ],
        "metadata": {
            "task_family": "delivery",
            "source_sample_id": sample_id,
            "format": "odiss_delivery_plaintext",
            "source": metadata.get("source", "unknown"),
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_dataset(inputs: list[Path], output_dir: Path) -> dict[str, int]:
    all_rows: list[dict[str, Any]] = []
    for path in inputs:
        all_rows.extend(_read_jsonl(path))

    router_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    delivery_rows: list[dict[str, Any]] = []

    for idx, sample in enumerate(all_rows):
        sample_id = f"sample-{idx:06d}"
        router_rows.append(_build_router_record(sample, sample_id))
        memory_rows.append(_build_memory_record(sample, sample_id))
        delivery_rows.append(_build_delivery_record(sample, sample_id))

    _write_jsonl(output_dir / "qwen_router_samples.jsonl", router_rows)
    _write_jsonl(output_dir / "qwen_memory_samples.jsonl", memory_rows)
    _write_jsonl(output_dir / "qwen_delivery_samples.jsonl", delivery_rows)

    return {
        "input_rows": len(all_rows),
        "router_rows": len(router_rows),
        "memory_rows": len(memory_rows),
        "delivery_rows": len(delivery_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[str(path) for path in DEFAULT_INPUTS],
        help="one or more source JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory where split datasets are written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = split_dataset(
        inputs=[Path(p) for p in args.inputs],
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
