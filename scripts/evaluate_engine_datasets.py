"""Evaluate dataset/runtime fit for ODISS engine-contract datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ALLOWED_ROUTE_MODES = {
    "tool_first",
    "frontier_first",
    "memory_only",
    "ask_user_clarify",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _assistant_content(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages", [])):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def _has_tool_calls(row: dict[str, Any]) -> bool:
    for message in row.get("messages", []):
        if message.get("tool_calls"):
            return True
    return False


def evaluate_router(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    for idx, row in enumerate(rows):
        content = _assistant_content(row)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            violations.append(f"row {idx}: assistant content is not JSON")
            continue
        route_mode = parsed.get("route_mode")
        if route_mode not in ALLOWED_ROUTE_MODES:
            violations.append(f"row {idx}: unknown route_mode={route_mode}")
        if _has_tool_calls(row):
            violations.append(f"row {idx}: router sample must not contain tool_calls")
        if "<think>" in content:
            violations.append(f"row {idx}: router sample must not contain <think>")
    return {
        "task_family": "router",
        "total_rows": len(rows),
        "violations": violations,
        "pass_rate": (len(rows) - len(violations)) / len(rows) if rows else 0.0,
    }


def evaluate_memory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    for idx, row in enumerate(rows):
        content = _assistant_content(row)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            violations.append(f"row {idx}: assistant content is not JSON")
            continue
        if not isinstance(parsed.get("normalized_medications", []), list):
            violations.append(f"row {idx}: normalized_medications must be a list")
        if not isinstance(parsed.get("dur_searchable"), bool):
            violations.append(f"row {idx}: dur_searchable must be bool")
        if not parsed.get("summary"):
            violations.append(f"row {idx}: summary must be non-empty")
        if _has_tool_calls(row):
            violations.append(f"row {idx}: memory sample must not contain tool_calls")
        if "<think>" in content:
            violations.append(f"row {idx}: memory sample must not contain <think>")
    return {
        "task_family": "memory",
        "total_rows": len(rows),
        "violations": violations,
        "pass_rate": (len(rows) - len(violations)) / len(rows) if rows else 0.0,
    }


def evaluate_delivery(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    for idx, row in enumerate(rows):
        content = _assistant_content(row)
        if not content.strip():
            violations.append(f"row {idx}: empty assistant content")
            continue
        if "<think>" in content or "</think>" in content:
            violations.append(f"row {idx}: delivery sample must not contain <think>")
        if "정확한 판단은 의사·약사 상담이 필요합니다" not in content:
            violations.append(f"row {idx}: missing safety disclaimer")
        if _has_tool_calls(row):
            violations.append(f"row {idx}: delivery sample must not contain tool_calls")
    return {
        "task_family": "delivery",
        "total_rows": len(rows),
        "violations": violations,
        "pass_rate": (len(rows) - len(violations)) / len(rows) if rows else 0.0,
    }


def evaluate_reasoning_alignment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect samples that still teach monolithic runtime-incompatible behavior."""
    mismatches: list[str] = []
    mismatch_rows = 0
    for idx, row in enumerate(rows):
        assistant_messages = [
            m for m in row.get("messages", []) if m.get("role") == "assistant"
        ]
        row_mismatch = False
        if any("<think>" in str(m.get("content", "")) for m in assistant_messages):
            mismatches.append(f"row {idx}: contains <think> (training-only artifact)")
            row_mismatch = True
        if _has_tool_calls(row):
            mismatches.append(f"row {idx}: contains monolithic tool_calls")
            row_mismatch = True
        if row_mismatch:
            mismatch_rows += 1
    return {
        "task_family": "reasoning",
        "total_rows": len(rows),
        "runtime_mismatches": mismatches,
        "mismatch_rate": mismatch_rows / len(rows) if rows else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="path to JSONL dataset")
    parser.add_argument(
        "--task-family",
        choices=["router", "memory", "delivery", "reasoning"],
        required=True,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when violations/mismatches are found",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.dataset))

    if args.task_family == "router":
        result = evaluate_router(rows)
        bad = bool(result["violations"])
    elif args.task_family == "memory":
        result = evaluate_memory(rows)
        bad = bool(result["violations"])
    elif args.task_family == "delivery":
        result = evaluate_delivery(rows)
        bad = bool(result["violations"])
    else:
        result = evaluate_reasoning_alignment(rows)
        bad = bool(result["runtime_mismatches"])

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.strict and bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
