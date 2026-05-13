"""Live backend validation harness for ODISS with a running Qwen/vLLM server.

This script intentionally exercises runtime paths instead of unit-test mocks:
- direct OpenAI-compatible vLLM calls
- ai-server health endpoints
- EngineOrchestrator turns with local delivery LLM enabled
- seeded HTTP /query/ask scenario

It writes JSONL case results and a compact Markdown report under
``reports/odiss_backend_validation``.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import websockets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database.md_store import md_store
from app.engines.conversation import ConversationEngine
from app.engines.llm_judge import LLMJudgeEngine
from app.engines.memory import MemoryEngine
from app.engines.reasoning import ReasoningEngine
from app.services.engine_orchestrator import EngineOrchestrator


DEFAULT_CASES = [
    {
        "id": "smalltalk_greeting",
        "text": "안녕하세요.",
        "expected_mode": "memory_only",
        "require_disclaimer": False,
    },
    {
        "id": "clarify_empty",
        "text": "",
        "expected_mode": "ask_user_clarify",
        "require_disclaimer": False,
    },
    {
        "id": "emergency_signal",
        "text": "어르신이 호흡이 힘들고 의식이 없어요. 119 불러야 하나요?",
        "expected_mode": "frontier_first",
    },
    {
        "id": "medication_interaction",
        "text": "타이레놀과 이부프로펜을 같이 먹어도 되나요?",
        "expected_mode": "tool_first",
    },
]


@dataclass
class ValidationContext:
    vllm_url: str
    backend_url: str
    model: str
    output_dir: Path
    run_id: str
    scenario: str | None = None
    strict: bool = False
    scenarios: list[dict[str, Any]] | None = None


USECASE_SCENARIOS = [
    {
        "id": "elder_prescription_followup",
        "speaker_id": "validation_elder_prescription",
        "seed_medications": ["타이레놀정", "이부프로펜정", "알마겔정"],
        "steps": [
            {
                "text": "이 약들 같이 먹어도 되나요?",
                "expected_mode": "tool_first",
                "expected_intent": "medication_query",
                "expected_terms": ["타이레놀", "이부프로펜"],
            },
            {
                "text": "그중 위장약은 뭐예요?",
                "expected_mode": "tool_first",
                "expected_intent": "medication_query",
                "expected_terms": ["알마겔"],
            },
            {
                "text": "아까 말한 약 다시 쉽게 설명해줘",
                "expected_mode": "memory_only",
                "expected_terms": ["타이레놀", "이부프로펜", "알마겔"],
            },
        ],
    },
    {
        "id": "smalltalk_to_medical",
        "speaker_id": "validation_smalltalk_medical",
        "seed_medications": ["아스피린정"],
        "steps": [
            {
                "text": "안녕하세요.",
                "expected_mode": "memory_only",
                "expected_intent": "smalltalk",
            },
            {
                "text": "아스피린정 먹을 때 주의할 점 알려줘",
                "expected_mode": "tool_first",
                "expected_intent": "medication_query",
                "expected_terms": ["아스피린"],
            },
            {
                "text": "고마워요.",
                "expected_mode": "memory_only",
                "expected_intent": "smalltalk",
                "require_disclaimer": False,
            },
        ],
    },
    {
        "id": "emergency_mid_conversation",
        "speaker_id": "validation_emergency_flow",
        "seed_medications": ["혈압약"],
        "steps": [
            {
                "text": "혈압약은 보통 언제 먹나요?",
                "expected_mode": "tool_first",
                "expected_intent": "medication_query",
                "expected_terms": ["혈압약"],
            },
            {
                "text": "지금 의식이 없고 호흡이 힘들어요. 119 불러야 하나요?",
                "expected_mode": "tool_first",
                "expected_intent": "emergency",
                "expected_terms": ["119"],
            },
            {
                "text": "조금 괜찮아졌는데 그래도 병원 가야 해요?",
                "expected_mode": "memory_only",
                "expected_terms": ["119", "응급"],
            },
        ],
    },
    {
        "id": "ambiguous_to_clarified",
        "speaker_id": "validation_clarify_flow",
        "seed_medications": ["타이레놀정"],
        "steps": [
            {
                "text": "",
                "expected_mode": "ask_user_clarify",
                "expected_intent": "unknown",
                "require_disclaimer": False,
            },
            {
                "text": "타이레놀정 먹어도 되는지 궁금해요",
                "expected_mode": "tool_first",
                "expected_intent": "medication_query",
                "expected_terms": ["타이레놀"],
            },
        ],
    },
]


def extract_json_from_markdown(text: str) -> Any:
    """Read the first fenced JSON block from a scenario note."""
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Markdown scenario notes must include a fenced ```json block.")
    return json.loads(match.group(1))


def load_scenarios_from_file(path: Path) -> list[dict[str, Any]]:
    """Load user-authored scenarios from .json or Markdown notes."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw) if path.suffix.lower() == ".json" else extract_json_from_markdown(raw)
    if isinstance(data, dict) and "scenarios" in data:
        scenarios = data["scenarios"]
    elif isinstance(data, dict):
        scenarios = [data]
    elif isinstance(data, list):
        scenarios = data
    else:
        raise ValueError("Scenario file must contain a scenario object, a list, or {'scenarios': [...]}.")
    return [normalize_scenario(scenario, index) for index, scenario in enumerate(scenarios, start=1)]


def normalize_scenario(scenario: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(scenario, dict):
        raise ValueError(f"Scenario #{index} must be an object.")
    scenario_id = scenario.get("id") or f"custom_scenario_{index}"
    steps = scenario.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Scenario '{scenario_id}' must define a non-empty steps list.")

    normalized_steps = []
    for step_index, step in enumerate(steps, start=1):
        if isinstance(step, str):
            step = {"text": step}
        if not isinstance(step, dict):
            raise ValueError(f"Scenario '{scenario_id}' step #{step_index} must be a string or object.")
        if "text" not in step:
            raise ValueError(f"Scenario '{scenario_id}' step #{step_index} is missing 'text'.")
        normalized_steps.append(
            {
                "id": step.get("id", f"step_{step_index}"),
                "text": str(step.get("text", "")),
                "expected_mode": step.get("expected_mode", "memory_only"),
                "expected_intent": step.get("expected_intent"),
                "expected_terms": list(step.get("expected_terms") or []),
                "forbidden_terms": list(step.get("forbidden_terms") or []),
                "trace_expectations": dict(step.get("trace_expectations") or {}),
                "expected_tool_calls": list(step.get("expected_tool_calls") or []),
                "must_not_call_tools": list(step.get("must_not_call_tools") or []),
                "expected_memory_reads": list(step.get("expected_memory_reads") or []),
                "expected_memory_writes": list(step.get("expected_memory_writes") or []),
                "expected_external_apis": list(step.get("expected_external_apis") or []),
                "expected_response_type": step.get("expected_response_type"),
                "expect_identity_gate": step.get("expect_identity_gate", False),
                "force_last_seen_minutes_ago": step.get("force_last_seen_minutes_ago"),
                "require_disclaimer": step.get("require_disclaimer", True),
                "include_judge": step.get("include_judge", False),
                "include_delivery_llm": step.get("include_delivery_llm", True),
                "allow_frontier_memory_fallback": step.get("allow_frontier_memory_fallback", False),
            }
        )

    return {
        "id": str(scenario_id),
        "speaker_id": str(scenario.get("speaker_id") or f"custom_{scenario_id}"),
        "runner": scenario.get("runner", "orchestrator"),
        "seed_medications": list(scenario.get("seed_medications") or []),
        "trace_expectations": dict(scenario.get("trace_expectations") or {}),
        "steps": normalized_steps,
    }


def scenarios_for_context(ctx: ValidationContext) -> list[dict[str, Any]]:
    return ctx.scenarios if ctx.scenarios is not None else USECASE_SCENARIOS


def post_json(url: str, payload: dict[str, Any], timeout: float = 90.0) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def quality_flags(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    has_think_tag = bool(re.search(r"</?think\b", stripped, flags=re.IGNORECASE))
    disclaimer_phrase = "정확한 판단은 의사·약사 상담이 필요합니다"
    disclaimer_variants = (
        disclaimer_phrase,
        "정확한 판단은 의사·약사와 상담이 필요합니다",
        "정확한 정보는 의사나 약사와 상담해 주세요",
        "정확한 조언은 의사나 약사와 상담",
    )
    disclaimer_count = sum(stripped.count(phrase) for phrase in disclaimer_variants)
    return {
        "non_empty": bool(stripped),
        "no_think_tags": not has_think_tag,
        "has_safety_disclaimer": any(phrase in stripped for phrase in disclaimer_variants),
        "safety_disclaimer_count": disclaimer_count,
        "has_duplicate_disclaimer": disclaimer_count > 1,
        "korean_friendly": any(token in stripped for token in ("사용자님", "님", "약", "드", "말씀")),
    }


def passed_quality(flags: dict[str, Any], *, require_disclaimer: bool = True) -> bool:
    required = ["non_empty", "no_think_tags", "korean_friendly"]
    if require_disclaimer:
        required.append("has_safety_disclaimer")
    return all(bool(flags.get(key)) for key in required)


def result_record(
    *,
    layer: str,
    case_id: str,
    status: str,
    elapsed_ms: float,
    payload: dict[str, Any],
    error: str | None = None,
    run_id: str | None = None,
    scenario_id: str | None = None,
    step_index: int | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "scenario_id": scenario_id,
        "step_index": step_index,
        "layer": layer,
        "case_id": case_id,
        "status": status,
        "elapsed_ms": round(elapsed_ms, 1),
        "payload": payload,
        "error": error,
    }


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def text_snapshot(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": len(text.encode("utf-8")),
        "chars": len(text),
        "sha16": hash_text(text),
        "preview": text[:300],
    }


def directory_snapshot(path: Path) -> dict[str, Any]:
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.exists() else []
    return {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "files": [str(p) for p in files[:20]],
    }


def latest_permanent_entries(category: str, limit: int = 5) -> list[str]:
    base = PROJECT_ROOT / "data" / "md_database" / "permanent" / category
    if not base.exists():
        return []
    files = sorted(base.glob("*/*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files[:limit]]


def memory_snapshot(speaker_id: str | None) -> dict[str, Any]:
    md_base = PROJECT_ROOT / "data" / "md_database"
    flash = md_base / "flash"
    safe_speaker = safe_segment(speaker_id or "")
    speaker_root = md_base / "structured_memory" / "speakers" / safe_speaker if speaker_id else None
    patient_root = md_base / "permanent" / "patients" / safe_speaker if speaker_id else None
    return {
        "flash": {
            "context_memory": text_snapshot(flash / "context_memory.md"),
            "current_requirement": text_snapshot(flash / "current_requirement.md"),
            "prescription_log": text_snapshot(flash / "prescription_log.md"),
        },
        "permanent_latest": {
            "medication_log": latest_permanent_entries("medication_log"),
            "prescriptions": latest_permanent_entries("prescriptions"),
            "dur_linkage": latest_permanent_entries("dur_linkage"),
            "ocr_history": latest_permanent_entries("ocr_history"),
        },
        "speaker": {
            "id": speaker_id,
            "patient_history": text_snapshot(patient_root / "history.md") if patient_root else {},
            "patient_profile": text_snapshot(patient_root / "profile.md") if patient_root else {},
            "structured": directory_snapshot(speaker_root) if speaker_root else {},
        },
    }


def diff_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    changed_flash = []
    for key, before_snap in before.get("flash", {}).items():
        after_snap = after.get("flash", {}).get(key, {})
        if before_snap.get("sha16") != after_snap.get("sha16"):
            changed_flash.append(key)

    before_latest = before.get("permanent_latest", {})
    after_latest = after.get("permanent_latest", {})
    permanent_growth = {
        key: max(0, len(after_latest.get(key, [])) - len(before_latest.get(key, [])))
        for key in set(before_latest) | set(after_latest)
    }

    before_speaker = before.get("speaker", {})
    after_speaker = after.get("speaker", {})
    return {
        "changed_flash": sorted(changed_flash),
        "permanent_growth": permanent_growth,
        "patient_history_changed": (
            before_speaker.get("patient_history", {}).get("sha16")
            != after_speaker.get("patient_history", {}).get("sha16")
        ),
        "structured_file_delta": (
            after_speaker.get("structured", {}).get("file_count", 0)
            - before_speaker.get("structured", {}).get("file_count", 0)
        ),
        "structured_bytes_delta": (
            after_speaker.get("structured", {}).get("total_bytes", 0)
            - before_speaker.get("structured", {}).get("total_bytes", 0)
        ),
    }


def safe_segment(value: str) -> str:
    import re

    cleaned = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", value.strip())
    return cleaned.strip("._") or "unknown"


def terms_present(text: str, terms: list[str]) -> dict[str, bool]:
    return {term: term in (text or "") for term in terms}


def recall_sources_from_turn(turn: Any) -> dict[str, Any]:
    return {
        "context": {
            "has_context_memory": bool(turn.context.get("context_memory")),
            "has_prescription_log": bool(turn.context.get("prescription_log")),
            "user_profile": turn.context.get("user_profile") or {},
            "memory_prompt_chars": len(turn.context.get("memory_prompt") or ""),
            "relevant_memories": len(turn.context.get("relevant_memories") or []),
            "memory_briefs": turn.context.get("memory_briefs") or [],
        },
        "evidence": {
            "normalized_medications": turn.evidence.normalized_medications,
            "dur_searchable": turn.evidence.dur_searchable,
            "used_frontier_fallback": turn.evidence.used_frontier_fallback,
            "artifact_refs": [ref.model_dump() for ref in turn.evidence.artifact_refs],
            "summary_preview": turn.evidence.summary[:300],
            "memory_prompt_chars": len(turn.evidence.memory_prompt or ""),
        },
        "execution_result_keys": sorted((turn.execution_results.get("task_results") or {}).keys()),
    }


def synthetic_engine_trace(
    *,
    turn: Any,
    include_judge: bool,
    include_delivery_llm: bool,
    memory_updated: bool,
) -> list[dict[str, Any]]:
    mode = turn.decision.mode.value
    branch_stage = {
        "tool_first": "Reasoning.execute_tasks",
        "memory_only": "Memory.search_history",
        "ask_user_clarify": "Reasoning.mark_clarify_required",
        "frontier_first": "Memory.frontier_fallback_or_skip",
    }.get(mode, "Reasoning.branch_unknown")
    stages = [
        ("Memory.initialize", "observed"),
        ("Conversation.receive_input", "observed"),
        ("Conversation.generate_filler", "observed"),
        ("Memory.load_context", "observed"),
        ("Reasoning.route_execution", "observed"),
        ("Memory.prepare_evidence_bundle", "observed"),
        (branch_stage, "observed"),
        ("Reasoning.core_message", "observed"),
        ("LLMJudge.review_final_answer", "observed" if include_judge else "skipped"),
        ("QwenDelivery.call_local_delivery_llm", "observed" if include_delivery_llm else "skipped"),
        ("Conversation.compose_from_contract", "observed"),
        ("Memory.update_and_compress", "observed" if memory_updated else "not_in_orchestrator"),
    ]
    return [
        {
            "order": index,
            "stage": stage,
            "status": status,
            "route_mode": mode if stage == "Reasoning.route_execution" else None,
            "intent": turn.decision.intent if stage == "Reasoning.route_execution" else None,
        }
        for index, (stage, status) in enumerate(stages, start=1)
    ]


def engine_trace_ok(trace: list[dict[str, Any]]) -> bool:
    expected_prefix = [
        "ME_Initialize",
        "CE_Input",
        "CE_Latency",
        "ME_Context",
        "RE_Intent",
        "ME_RAG",
    ]
    return [item["stage"] for item in trace[: len(expected_prefix)]] == expected_prefix


def _as_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return item
    return {}


def _normalize_trace_token(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", str(value or "").lower())


def _matches_trace_token(expected: str, actual: str) -> bool:
    expected_norm = _normalize_trace_token(expected)
    actual_norm = _normalize_trace_token(actual)
    if not expected_norm:
        return True
    if not actual_norm:
        return False
    return expected_norm in actual_norm or actual_norm in expected_norm


def _ordered_subsequence(expected: list[str], actual: list[str]) -> dict[str, Any]:
    missing: list[str] = []
    cursor = 0
    for expected_item in expected:
        found_at = None
        for idx in range(cursor, len(actual)):
            if _matches_trace_token(expected_item, actual[idx]):
                found_at = idx
                break
        if found_at is None:
            missing.append(expected_item)
        else:
            cursor = found_at + 1
    return {"ok": not missing, "missing": missing}


def _event_text(event: dict[str, Any]) -> str:
    metadata = event.get("metadata") or {}
    values = [
        event.get("stage"),
        event.get("component"),
        event.get("action"),
        event.get("operation"),
        event.get("logical_file"),
        event.get("category"),
        event.get("path"),
        event.get("tool_id"),
        event.get("tool_name"),
        event.get("external_api"),
        json.dumps(metadata, ensure_ascii=False, default=str),
    ]
    return " ".join(str(value) for value in values if value)


def _contains_expected(expected: str, events: list[dict[str, Any]]) -> bool:
    return any(_matches_trace_token(expected, _event_text(event)) for event in events)


def _merge_trace_expectations(*sources: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    list_keys = {
        "expected_tool_calls",
        "must_not_call_tools",
        "expected_memory_reads",
        "expected_memory_writes",
        "expected_external_apis",
        "must_not_wait_for",
    }
    for source in sources:
        if not source:
            continue
        for key, value in source.items():
            if key == "expected_engine_sequence":
                # A step-level sequence is a narrower contract for one turn.
                # Concatenating it with scenario-level sequences makes the
                # validator require two ordered traces in a single turn.
                merged[key] = list(value or [])
            elif key in list_keys:
                current = list(merged.get(key) or [])
                for item in value or []:
                    if item not in current:
                        current.append(item)
                merged[key] = current
            else:
                merged[key] = value
    return merged


def validate_trace_expectations(
    *,
    expectations: dict[str, Any],
    engine_trace: list[Any],
    memory_trace: list[Any],
    tool_trace: list[Any],
) -> dict[str, Any]:
    """Validate scenario trace metadata against observed structured traces."""
    engine_events = [_as_dict(item) for item in engine_trace]
    memory_events = [_as_dict(item) for item in memory_trace]
    tool_events = [_as_dict(item) for item in tool_trace]
    combined_events = engine_events + memory_events + tool_events
    combined_trace_text = [_event_text(event) for event in combined_events]

    expected_sequence = list(expectations.get("expected_engine_sequence") or [])
    sequence = _ordered_subsequence(expected_sequence, combined_trace_text)

    expected_tool_calls = list(expectations.get("expected_tool_calls") or [])
    must_not_call_tools = list(expectations.get("must_not_call_tools") or [])
    expected_external_apis = list(expectations.get("expected_external_apis") or [])
    expected_memory_reads = list(expectations.get("expected_memory_reads") or [])
    expected_memory_writes = list(expectations.get("expected_memory_writes") or [])

    expected_tools = {
        item: _contains_expected(item, tool_events)
        for item in expected_tool_calls
    }
    forbidden_tools = {
        item: _contains_expected(item, tool_events)
        for item in must_not_call_tools
    }
    external_apis = {
        item: _contains_expected(item, tool_events)
        for item in expected_external_apis
    }
    memory_reads = {
        item: any(
            str(event.get("operation")) in {"read", "search"}
            and _matches_trace_token(item, _event_text(event))
            for event in memory_events
        )
        for item in expected_memory_reads
    }
    memory_writes = {
        item: any(
            str(event.get("operation")) in {"write", "update"}
            and _matches_trace_token(item, _event_text(event))
            for event in memory_events
        )
        for item in expected_memory_writes
    }

    checks = {
        "engine_sequence_ok": sequence["ok"],
        "expected_tool_calls_ok": all(expected_tools.values()) if expected_tools else True,
        "must_not_call_tools_ok": not any(forbidden_tools.values()) if forbidden_tools else True,
        "expected_external_apis_ok": all(external_apis.values()) if external_apis else True,
        "expected_memory_reads_ok": all(memory_reads.values()) if memory_reads else True,
        "expected_memory_writes_ok": all(memory_writes.values()) if memory_writes else True,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "engine_sequence": sequence,
        "expected_tool_calls": expected_tools,
        "must_not_call_tools": forbidden_tools,
        "expected_external_apis": external_apis,
        "expected_memory_reads": memory_reads,
        "expected_memory_writes": memory_writes,
    }


async def seed_speaker_medication_context(speaker_id: str, med_names: list[str]) -> None:
    memory = MemoryEngine()
    await memory.initialize()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ocr_data = {"medications": [{"name": name} for name in med_names]}
    dur_results = [
        {
            "name": name,
            "contraindications": [],
            "precautions": [{"note": "validation seed"}],
        }
        for name in med_names
    ]
    await memory.log_ocr_result(ocr_data, confidence=1.0)
    await memory.sync_ocr_dur(ocr_data, dur_results, speaker_id=speaker_id)


async def validate_vllm(ctx: ValidationContext) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case_id, prompt in [
        ("plain_ping", "ping에 한 단어로 답해줘."),
        ("no_think_ping", "/no_think\nping에 한 단어로 답해줘."),
    ]:
        started = time.perf_counter()
        try:
            data = post_json(
                f"{ctx.vllm_url}/v1/chat/completions",
                {
                    "model": ctx.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 64,
                    "temperature": 0,
                },
            )
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            flags = quality_flags(content)
            results.append(
                result_record(
                    run_id=ctx.run_id,
                    layer="vllm",
                    case_id=case_id,
                    status="ok",
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    payload={
                        "model": data.get("model"),
                        "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
                        "contains_think": not flags["no_think_tags"],
                        "content_preview": content[:240],
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - validation report should capture all failures
            results.append(
                result_record(
                    run_id=ctx.run_id,
                    layer="vllm",
                    case_id=case_id,
                    status="error",
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    payload={},
                    error=repr(exc),
                )
            )
    return results


async def validate_health(ctx: ValidationContext) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case_id, url in [
        ("backend_health", f"{ctx.backend_url}/health"),
        ("backend_llm_health", f"{ctx.backend_url}/health/llm"),
    ]:
        started = time.perf_counter()
        try:
            data = get_json(url, timeout=90)
            status = "ok" if data.get("status") in {"ok", None} else "error"
            results.append(
                result_record(
                    run_id=ctx.run_id,
                    layer="health",
                    case_id=case_id,
                    status=status,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    payload=data,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                result_record(
                    run_id=ctx.run_id,
                    layer="health",
                    case_id=case_id,
                    status="error",
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    payload={},
                    error=repr(exc),
                )
            )
    return results


async def validate_server_status(ctx: ValidationContext) -> list[dict[str, Any]]:
    started = time.perf_counter()
    payload: dict[str, Any] = {}
    status = "ok"
    error = None
    try:
        payload["vllm_models"] = get_json(f"{ctx.vllm_url}/v1/models", timeout=10)
        payload["backend_health"] = get_json(f"{ctx.backend_url}/health", timeout=10)
        payload["backend_llm_health"] = get_json(f"{ctx.backend_url}/health/llm", timeout=90)
        try:
            gpu_output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
            payload["gpu"] = gpu_output.strip().splitlines()
        except Exception as gpu_exc:  # noqa: BLE001
            payload["gpu_error"] = repr(gpu_exc)

        recent_reports = sorted(
            ctx.output_dir.glob("*.md"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ) if ctx.output_dir.exists() else []
        payload["recent_reports"] = [str(path) for path in recent_reports[:5]]
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = repr(exc)

    return [
        result_record(
            run_id=ctx.run_id,
            layer="status",
            case_id="server_status",
            status=status,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            payload=payload,
            error=error,
        )
    ]


async def validate_orchestrator(ctx: ValidationContext) -> list[dict[str, Any]]:
    memory = MemoryEngine()
    judge = LLMJudgeEngine()
    reasoning = ReasoningEngine(memory, judge)
    conversation = ConversationEngine()
    orchestrator = EngineOrchestrator(
        memory_engine=memory,
        reasoning_engine=reasoning,
        conversation_engine=conversation,
        llm_judge=judge,
    )

    results: list[dict[str, Any]] = []
    for case in DEFAULT_CASES:
        started = time.perf_counter()
        try:
            turn = await orchestrator.run_turn(
                text=case["text"],
                speaker_id="validation_backend",
                include_judge=False,
                include_delivery_llm=True,
                allow_frontier_memory_fallback=False,
            )
            final_text = turn.conversation.response_text
            flags = quality_flags(final_text)
            mode = str(turn.decision.mode.value)
            status = (
                "ok"
                if mode == case["expected_mode"]
                and passed_quality(flags, require_disclaimer=case.get("require_disclaimer", True))
                else "fail"
            )
            results.append(
                result_record(
                    run_id=ctx.run_id,
                    layer="orchestrator",
                    case_id=case["id"],
                    status=status,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    payload={
                        "input": case["text"],
                        "expected_mode": case["expected_mode"],
                        "actual_mode": mode,
                        "intent": turn.decision.intent,
                        "response_type": turn.conversation.response_type,
                        "quality": flags,
                        "final_preview": final_text[:500],
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                result_record(
                    run_id=ctx.run_id,
                    layer="orchestrator",
                    case_id=case["id"],
                    status="error",
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    payload={"input": case["text"]},
                    error=repr(exc),
                )
            )
    return results


async def run_orchestrator_step(
    *,
    scenario_id: str,
    step_index: int,
    speaker_id: str,
    scenario_trace_expectations: dict[str, Any],
    step: dict[str, Any],
    ctx: ValidationContext,
) -> dict[str, Any]:
    memory = MemoryEngine()
    judge = LLMJudgeEngine()
    reasoning = ReasoningEngine(memory, judge)
    conversation = ConversationEngine()
    orchestrator = EngineOrchestrator(
        memory_engine=memory,
        reasoning_engine=reasoning,
        conversation_engine=conversation,
        llm_judge=judge,
    )

    before = memory_snapshot(speaker_id)
    started = time.perf_counter()
    include_judge = bool(step.get("include_judge", False))
    include_delivery_llm = bool(step.get("include_delivery_llm", True))
    try:
        if step.get("force_last_seen_minutes_ago") is not None:
            await memory.force_identity_last_seen_minutes_ago(
                speaker_id,
                int(step["force_last_seen_minutes_ago"]),
            )
            before = memory_snapshot(speaker_id)

        if step.get("expect_identity_gate"):
            turn = await orchestrator.run_turn(
                text=step["text"],
                speaker_id=speaker_id,
                include_judge=False,
                include_delivery_llm=False,
                allow_frontier_memory_fallback=False,
                run_identity_gate=True,
            )
            identity_gate = turn.identity_gate
            final_text = turn.conversation.response_text
            flags = quality_flags(final_text)
            expected_terms = step.get("expected_terms", [])
            term_hits = terms_present(final_text, expected_terms)
            terms_ok = all(term_hits.values()) if expected_terms else True
            forbidden_hits = terms_present(final_text, step.get("forbidden_terms", []))
            forbidden_ok = not any(forbidden_hits.values())
            response_type_ok = (
                not step.get("expected_response_type")
                or turn.conversation.response_type == step.get("expected_response_type")
            )
            after = memory_snapshot(speaker_id)
            memory_diff = diff_snapshots(before, after)
            status = "ok" if all([
                not identity_gate.get("allowed", True),
                response_type_ok,
                terms_ok,
                forbidden_ok,
                flags["non_empty"],
                flags["no_think_tags"],
            ]) else "fail"
            return result_record(
                run_id=ctx.run_id,
                scenario_id=scenario_id,
                step_index=step_index,
                layer="scenario_identity_gate",
                case_id=step.get("id", f"step_{step_index}"),
                status=status,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                payload={
                    "speaker_id": speaker_id,
                    "input": step["text"],
                    "expected_response_type": step.get("expected_response_type"),
                    "actual_response_type": turn.conversation.response_type,
                    "identity_gate": {
                        "allowed": identity_gate.get("allowed"),
                        "reason": identity_gate.get("reason"),
                        "metadata": identity_gate.get("metadata") or {},
                    },
                    "engine_call_trace": [_as_dict(item) for item in turn.engine_trace],
                    "memory_trace": [_as_dict(item) for item in turn.memory_trace],
                    "quality": flags,
                    "term_hits": term_hits,
                    "forbidden_hits": forbidden_hits,
                    "memory_before": before,
                    "memory_after": after,
                    "memory_diff": memory_diff,
                    "checks": {
                        "gate_blocked": not identity_gate.get("allowed", True),
                        "response_type_ok": response_type_ok,
                        "terms_ok": terms_ok,
                        "forbidden_ok": forbidden_ok,
                        "non_empty": flags["non_empty"],
                        "no_think_tags": flags["no_think_tags"],
                    },
                    "final_preview": final_text[:700],
                },
            )

        turn = await orchestrator.run_turn(
            text=step["text"],
            speaker_id=speaker_id,
            include_judge=include_judge,
            include_delivery_llm=include_delivery_llm,
            allow_frontier_memory_fallback=bool(step.get("allow_frontier_memory_fallback", False)),
        )
        await memory.update_and_compress(
            {
                "query": step["text"],
                "answer": turn.conversation.response_text,
                "type": turn.decision.intent,
                "core_message": turn.core_message,
                "judge_review": turn.judge_review,
                "dur_results": turn.execution_results.get("task_results", {}).get("dur"),
            },
            speaker_id=speaker_id,
        )
        update_memory_trace = [
            {
                "operation": "write",
                "logical_file": "MedicationLog.md",
                "category": "medication_log",
                "path": "permanent/medication_log/*/*.md",
                "status": "observed",
                "metadata": {"source": "MemoryEngine.update_and_compress"},
            },
            {
                "operation": "write",
                "logical_file": "CurrentRequirement.md",
                "category": "current_requirement",
                "path": "flash/current_requirement.md",
                "status": "observed",
                "metadata": {"source": "MemoryEngine.update_and_compress"},
            },
            {
                "operation": "write",
                "logical_file": "ContextMemory.md",
                "category": "context_memory",
                "path": "flash/context_memory.md",
                "status": "observed",
                "metadata": {"source": "MemoryEngine.update_and_compress"},
            },
            {
                "operation": "write",
                "logical_file": "patients/{speaker_id}/history.md",
                "category": "patients",
                "path": f"permanent/patients/{speaker_id}/history.md",
                "status": "observed",
                "metadata": {"source": "MemoryEngine.update_and_compress"},
            },
        ]
        if turn.execution_results.get("task_results", {}).get("dur"):
            update_memory_trace.append(
                {
                    "operation": "write",
                    "logical_file": "DURLinkageHistory.md",
                    "category": "dur_linkage",
                    "path": "permanent/dur_linkage/*/*.md",
                    "status": "observed",
                    "metadata": {"source": "MemoryEngine.update_and_compress"},
                }
            )
        after = memory_snapshot(speaker_id)
        memory_diff = diff_snapshots(before, after)
        final_text = turn.conversation.response_text
        flags = quality_flags(final_text)
        trace = [_as_dict(item) for item in (turn.engine_trace or [])]
        if not trace:
            trace = synthetic_engine_trace(
                turn=turn,
                include_judge=include_judge,
                include_delivery_llm=include_delivery_llm,
                memory_updated=True,
            )
        else:
            trace.append(
                {
                    "stage": "ME_Update",
                    "component": "MemoryEngine",
                    "action": "update_and_compress",
                    "status": "observed",
                    "metadata": {"speaker_id": speaker_id},
                }
            )
        memory_trace = [_as_dict(item) for item in (turn.memory_trace or [])] + update_memory_trace
        tool_trace = [_as_dict(item) for item in (turn.tool_trace or [])]
        expected_terms = step.get("expected_terms", [])
        final_answer_term_hits = terms_present(final_text, expected_terms)
        model_stage_term_hits = terms_present(
            " ".join(
                [
                    turn.core_message,
                    turn.reviewed_message,
                    turn.delivery_message,
                ]
            ),
            expected_terms,
        )
        memory_term_hits = terms_present(
            " ".join(
                [
                    json.dumps(recall_sources_from_turn(turn), ensure_ascii=False),
                    json.dumps(before, ensure_ascii=False),
                    json.dumps(after, ensure_ascii=False),
                    json.dumps(memory_diff, ensure_ascii=False),
                ]
            ),
            expected_terms,
        )
        term_hits = {
            "final_answer": final_answer_term_hits,
            "model_stages": model_stage_term_hits,
            "memory": memory_term_hits,
        }
        mode_ok = turn.decision.mode.value == step.get("expected_mode")
        intent_ok = (
            not step.get("expected_intent")
            or turn.decision.intent == step.get("expected_intent")
        )
        final_answer_terms_ok = (
            all(final_answer_term_hits.values())
            if expected_terms
            else True
        )
        model_stage_terms_ok = (
            all(model_stage_term_hits.values())
            if expected_terms
            else True
        )
        memory_terms_ok = (
            all(memory_term_hits.values())
            if expected_terms
            else True
        )
        forbidden_hits = terms_present(final_text, step.get("forbidden_terms", []))
        forbidden_ok = not any(forbidden_hits.values())
        response_type_ok = (
            not step.get("expected_response_type")
            or turn.conversation.response_type == step.get("expected_response_type")
        )
        quality_ok = passed_quality(
            flags,
            require_disclaimer=step.get("require_disclaimer", True),
        )
        memory_ok = bool(memory_diff["changed_flash"]) and memory_diff["patient_history_changed"]
        trace_ok = engine_trace_ok(trace)
        step_trace_metadata = step.get("trace_expectations") or {}
        trace_expectations = _merge_trace_expectations(
            None if step_trace_metadata else scenario_trace_expectations,
            step_trace_metadata,
            {
                "expected_tool_calls": step.get("expected_tool_calls", []),
                "must_not_call_tools": step.get("must_not_call_tools", []),
                "expected_memory_reads": step.get("expected_memory_reads", []),
                "expected_memory_writes": step.get("expected_memory_writes", []),
                "expected_external_apis": step.get("expected_external_apis", []),
            },
        )
        trace_checks = validate_trace_expectations(
            expectations=trace_expectations,
            engine_trace=trace,
            memory_trace=memory_trace,
            tool_trace=tool_trace,
        )
        status = "ok" if all([
            mode_ok,
            intent_ok,
            final_answer_terms_ok,
            forbidden_ok,
            response_type_ok,
            quality_ok,
            memory_ok,
            trace_ok,
            trace_checks["ok"],
        ]) else "fail"
        return result_record(
            run_id=ctx.run_id,
            scenario_id=scenario_id,
            step_index=step_index,
            layer="scenario_orchestrator",
            case_id=step.get("id", f"step_{step_index}"),
            status=status,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            payload={
                "speaker_id": speaker_id,
                "input": step["text"],
                "expected_mode": step.get("expected_mode"),
                "actual_mode": turn.decision.mode.value,
                "expected_intent": step.get("expected_intent"),
                "actual_intent": turn.decision.intent,
                "response_type": turn.conversation.response_type,
                "quality": flags,
                "term_hits": term_hits,
                "forbidden_hits": forbidden_hits,
                "recall_sources": recall_sources_from_turn(turn),
                "memory_before": before,
                "memory_after": after,
                "memory_diff": memory_diff,
                "engine_call_trace": trace,
                "memory_trace": memory_trace,
                "tool_trace": tool_trace,
                "trace_expectations": trace_expectations,
                "trace_checks": trace_checks,
                "checks": {
                    "mode_ok": mode_ok,
                    "intent_ok": intent_ok,
                    "final_answer_terms_ok": final_answer_terms_ok,
                    "model_stage_terms_ok": model_stage_terms_ok,
                    "memory_terms_ok": memory_terms_ok,
                    "forbidden_ok": forbidden_ok,
                    "response_type_ok": response_type_ok,
                    "quality_ok": quality_ok,
                    "memory_ok": memory_ok,
                    "trace_ok": trace_ok,
                    "trace_expectations_ok": trace_checks["ok"],
                },
                "final_preview": final_text[:700],
            },
        )
    except Exception as exc:  # noqa: BLE001
        after = memory_snapshot(speaker_id)
        return result_record(
            run_id=ctx.run_id,
            scenario_id=scenario_id,
            step_index=step_index,
            layer="scenario_orchestrator",
            case_id=step.get("id", f"step_{step_index}"),
            status="error",
            elapsed_ms=(time.perf_counter() - started) * 1000,
            payload={
                "speaker_id": speaker_id,
                "input": step.get("text", ""),
                "memory_before": before,
                "memory_after": after,
                "memory_diff": diff_snapshots(before, after),
            },
            error=repr(exc),
        )


async def validate_usecase_scenarios(ctx: ValidationContext) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    scenarios = [
        scenario for scenario in scenarios_for_context(ctx)
        if scenario.get("runner", "orchestrator") != "websocket"
        and ctx.scenario in {None, scenario["id"]}
    ]
    for scenario in scenarios:
        speaker_id = f"{scenario['speaker_id']}_{ctx.run_id[:8]}"
        await seed_speaker_medication_context(
            speaker_id,
            scenario.get("seed_medications", []),
        )
        seed_after = memory_snapshot(speaker_id)
        results.append(
            result_record(
                run_id=ctx.run_id,
                scenario_id=scenario["id"],
                step_index=0,
                layer="scenario_seed",
                case_id="seed_medication_context",
                status="ok",
                elapsed_ms=0.0,
                payload={
                    "speaker_id": speaker_id,
                    "seed_medications": scenario.get("seed_medications", []),
                    "memory_after": seed_after,
                    "write_paths": {
                        "ocr_history": bool(seed_after["permanent_latest"]["ocr_history"]),
                        "prescriptions": bool(seed_after["permanent_latest"]["prescriptions"]),
                        "prescription_log": seed_after["flash"]["prescription_log"]["exists"],
                        "structured_speaker_files": seed_after["speaker"]["structured"].get("file_count", 0),
                    },
                },
            )
        )
        for index, step in enumerate(scenario["steps"], start=1):
            results.append(
                await run_orchestrator_step(
                    scenario_id=scenario["id"],
                    step_index=index,
                    speaker_id=speaker_id,
                    scenario_trace_expectations=scenario.get("trace_expectations", {}),
                    step=step,
                    ctx=ctx,
                )
            )
    return results


async def validate_websocket_dialogue(ctx: ValidationContext) -> list[dict[str, Any]]:
    scenario_pool = scenarios_for_context(ctx)
    if ctx.scenarios is None:
        scenario = next(
            (item for item in scenario_pool if item["id"] == "smalltalk_to_medical"),
            scenario_pool[0],
        )
    else:
        websocket_scenarios = [
            item for item in scenario_pool
            if item.get("runner") == "websocket"
        ]
        if not websocket_scenarios:
            return []
        scenario = websocket_scenarios[0]
    if ctx.scenario and ctx.scenario != scenario["id"]:
        return []

    speaker_id = f"ws_{scenario['speaker_id']}_{ctx.run_id[:8]}"
    await seed_speaker_medication_context(speaker_id, scenario.get("seed_medications", []))
    ws_url = ctx.backend_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/chat"
    results: list[dict[str, Any]] = []

    try:
        async with websockets.connect(ws_url, open_timeout=10) as websocket:
            for index, step in enumerate(scenario["steps"], start=1):
                if step.get("force_last_seen_minutes_ago") is not None:
                    force_memory = MemoryEngine()
                    await force_memory.force_identity_last_seen_minutes_ago(
                        speaker_id,
                        int(step["force_last_seen_minutes_ago"]),
                    )
                before = memory_snapshot(speaker_id)
                started = time.perf_counter()
                await websocket.send(
                    json.dumps(
                        {
                            "type": "stt_result",
                            "text": step["text"],
                            "speaker_id": speaker_id,
                        },
                        ensure_ascii=False,
                    )
                )
                messages: list[dict[str, Any]] = []
                for _ in range(6):
                    raw = await asyncio.wait_for(websocket.recv(), timeout=120)
                    message = json.loads(raw)
                    messages.append(message)
                    if message.get("type") in {"response", "error"}:
                        break
                response = next((msg for msg in messages if msg.get("type") == "response"), {})
                final_text = response.get("response_text") or response.get("text", "")
                after = memory_snapshot(speaker_id)
                for _ in range(10):
                    if diff_snapshots(before, after)["patient_history_changed"]:
                        break
                    await asyncio.sleep(0.2)
                    after = memory_snapshot(speaker_id)
                flags = quality_flags(final_text)
                memory_diff = diff_snapshots(before, after)
                actual_response_type = response.get("response_type")
                response_type_ok = (
                    not step.get("expected_response_type")
                    or actual_response_type == step.get("expected_response_type")
                )
                expected_terms = step.get("expected_terms", [])
                term_hits = terms_present(final_text, expected_terms)
                terms_ok = all(term_hits.values()) if expected_terms else True
                forbidden_hits = terms_present(final_text, step.get("forbidden_terms", []))
                forbidden_ok = not any(forbidden_hits.values())
                identity_gate_ok = (
                    not step.get("expect_identity_gate")
                    or bool(response.get("identity_gate"))
                )
                status = (
                    "ok"
                    if response
                    and passed_quality(flags, require_disclaimer=step.get("require_disclaimer", True))
                    and (memory_diff["patient_history_changed"] or step.get("expect_identity_gate"))
                    and response_type_ok
                    and terms_ok
                    and forbidden_ok
                    and identity_gate_ok
                    else "fail"
                )
                results.append(
                    result_record(
                        run_id=ctx.run_id,
                        scenario_id=f"{scenario['id']}_websocket",
                        step_index=index,
                        layer="websocket",
                        case_id=step.get("id", f"step_{index}"),
                        status=status,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        payload={
                            "speaker_id": speaker_id,
                            "input": step["text"],
                            "messages": messages,
                            "quality": flags,
                            "expected_response_type": step.get("expected_response_type"),
                            "actual_response_type": actual_response_type,
                            "term_hits": term_hits,
                            "forbidden_hits": forbidden_hits,
                            "memory_before": before,
                            "memory_after": after,
                            "memory_diff": memory_diff,
                            "engine_call_trace": [
                                {"order": 1, "stage": "WebSocket.receive_stt_result", "status": "observed"},
                                {"order": 2, "stage": "EngineOrchestrator.run_turn", "status": "observed"},
                                {"order": 3, "stage": "WebSocket.send_filler", "status": "observed" if any(msg.get("type") == "filler" for msg in messages) else "skipped"},
                                {"order": 4, "stage": "WebSocket.send_response", "status": "observed" if response else "missing"},
                                {"order": 5, "stage": "Memory.update_and_compress", "status": "observed" if memory_diff["patient_history_changed"] else "missing"},
                            ],
                            "checks": {
                                "response_type_ok": response_type_ok,
                                "terms_ok": terms_ok,
                                "forbidden_ok": forbidden_ok,
                                "identity_gate_ok": identity_gate_ok,
                                "memory_ok": bool(memory_diff["patient_history_changed"] or step.get("expect_identity_gate")),
                            },
                            "final_preview": final_text[:700],
                        },
                    )
                )
    except Exception as exc:  # noqa: BLE001
        results.append(
            result_record(
                run_id=ctx.run_id,
                scenario_id=f"{scenario['id']}_websocket",
                step_index=None,
                layer="websocket",
                case_id="dialogue_connection",
                status="error",
                elapsed_ms=0.0,
                payload={"speaker_id": speaker_id, "ws_url": ws_url},
                error=repr(exc),
            )
        )
    return results


async def seed_http_session() -> str:
    await md_store.initialize()
    session_id = f"validation-{uuid.uuid4()}"
    content = (
        "# 파이프라인 세션\n"
        f"> 세션 ID: {session_id}\n"
        f"> 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        "## 사용자 질문\n타이레놀과 이부프로펜을 같이 먹어도 되나요?\n\n"
        "## OCR 결과\n```json\n{\"medications\": [\"타이레놀\", \"이부프로펜\"]}\n```\n\n"
        "## DUR 결과\n```json\n{\"items\": []}\n```\n\n"
        "## LLM 문서\n"
        "- 확인된 약 후보: 타이레놀, 이부프로펜\n"
        "- 병용 여부는 개인 질환과 복용량에 따라 달라질 수 있어 약사 확인이 필요합니다.\n"
    )
    await md_store.save("medication_log", content)
    return session_id


async def validate_http(ctx: ValidationContext) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    session_id = await seed_http_session()
    started = time.perf_counter()
    try:
        data = post_json(
            f"{ctx.backend_url}/query/ask",
            {
                "session_id": session_id,
                "query_text": "타이레놀과 이부프로펜을 같이 먹어도 되나요?",
            },
            timeout=180,
        )
        final_text = data.get("answer_final", "")
        flags = quality_flags(final_text)
        results.append(
            result_record(
                run_id=ctx.run_id,
                layer="http",
                case_id="seeded_query_ask",
                status="ok" if passed_quality(flags) else "fail",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                payload={
                    "session_id": session_id,
                    "sent_to_mcp": data.get("sent_to_mcp"),
                    "sent_to_device": data.get("sent_to_device"),
                    "quality": flags,
                    "answer_final_preview": final_text[:500],
                    "answer_internal_preview": (data.get("answer_internal") or "")[:300],
                    "answer_verified_preview": (data.get("answer_verified") or "")[:300],
                },
            )
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:800]
        results.append(
            result_record(
                run_id=ctx.run_id,
                layer="http",
                case_id="seeded_query_ask",
                status="error",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                payload={"session_id": session_id, "http_status": exc.code, "body": body},
                error=repr(exc),
            )
        )
    except Exception as exc:  # noqa: BLE001
        results.append(
            result_record(
                run_id=ctx.run_id,
                layer="http",
                case_id="seeded_query_ask",
                status="error",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                payload={"session_id": session_id},
                error=repr(exc),
            )
        )

    results.append(
        result_record(
            run_id=ctx.run_id,
            layer="http",
            case_id="query_pipeline_real_image",
            status="skipped",
            elapsed_ms=0.0,
            payload={
                "reason": "No sample prescription image was provided to this harness. Use POST /query/pipeline with multipart image to run the OCR/DUR full path.",
            },
        )
    )
    return results


def write_outputs(ctx: ValidationContext, results: list[dict[str, Any]]) -> tuple[Path, Path]:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = ctx.output_dir / f"usecase_validation_{stamp}_{ctx.run_id[:8]}.jsonl"
    md_path = ctx.output_dir / f"usecase_validation_{stamp}_{ctx.run_id[:8]}.md"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    lines = [
        "# ODISS Backend Live Validation Report",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Run ID: `{ctx.run_id}`",
        f"- vLLM URL: `{ctx.vllm_url}`",
        f"- Backend URL: `{ctx.backend_url}`",
        f"- Model: `{ctx.model}`",
        f"- Scenario filter: `{ctx.scenario or 'all'}`",
        f"- Result counts: `{json.dumps(counts, ensure_ascii=False)}`",
        "",
        "| Scenario | Step | Layer | Case | Status | Elapsed ms | Notes |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for row in results:
        payload = row.get("payload") or {}
        note = (
            payload.get("final_preview")
            or payload.get("answer_final_preview")
            or payload.get("answer_preview")
            or payload.get("content_preview")
            or payload.get("reason")
            or row.get("error")
            or ""
        )
        note = str(note).replace("\n", " ")[:180]
        lines.append(
            f"| {row.get('scenario_id') or '-'} | {row.get('step_index') if row.get('step_index') is not None else '-'} | {row['layer']} | {row['case_id']} | {row['status']} | {row['elapsed_ms']} | {note} |"
        )
    lines.append("")
    lines.append(f"Raw JSONL: `{jsonl_path}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path, md_path


async def run(ctx: ValidationContext) -> int:
    results: list[dict[str, Any]] = []
    preflight_results = await validate_server_status(ctx)
    results.extend(preflight_results)
    if getattr(ctx, "preflight_only", False):
        jsonl_path, md_path = write_outputs(ctx, results)
        print(json.dumps({"jsonl": str(jsonl_path), "markdown": str(md_path)}, ensure_ascii=False))
        return 0 if all(row["status"] == "ok" for row in results) else 1

    if any(row["status"] != "ok" for row in preflight_results):
        results.append(
            result_record(
                run_id=ctx.run_id,
                layer="preflight",
                case_id="skip_runtime_scenarios",
                status="skipped",
                elapsed_ms=0.0,
                payload={
                    "reason": (
                        "vLLM or ai-server preflight failed. Start both servers, "
                        "then rerun validation."
                    )
                },
            )
        )
        jsonl_path, md_path = write_outputs(ctx, results)
        print(json.dumps({"jsonl": str(jsonl_path), "markdown": str(md_path)}, ensure_ascii=False))
        return 1 if ctx.strict else 0

    for validator in (validate_vllm, validate_health):
        results.extend(await validator(ctx))
    results.extend(await validate_orchestrator(ctx))
    results.extend(await validate_usecase_scenarios(ctx))
    results.extend(await validate_websocket_dialogue(ctx))
    results.extend(await validate_http(ctx))

    jsonl_path, md_path = write_outputs(ctx, results)
    print(json.dumps({"jsonl": str(jsonl_path), "markdown": str(md_path)}, ensure_ascii=False))

    hard_failures = [
        row for row in results
        if row["status"] in {"error", "fail"} and row["case_id"] != "query_pipeline_real_image"
    ]
    return 1 if ctx.strict and hard_failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", default="http://localhost:8001")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--scenario", help="Run only one scenario id. Works with built-in or --scenario-file scenarios.")
    parser.add_argument(
        "--scenario-file",
        type=Path,
        help="Load user-authored scenarios from a .json file or Markdown note with a fenced ```json block.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any non-skipped validation fails.")
    parser.add_argument("--preflight-only", action="store_true", help="Only check vLLM, ai-server, GPU, and recent reports.")
    parser.add_argument(
        "--output-dir",
        default="reports/odiss_backend_validation",
        help="Directory for JSONL and Markdown validation reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = load_scenarios_from_file(args.scenario_file) if args.scenario_file else None
    ctx = ValidationContext(
        vllm_url=args.vllm_url.rstrip("/"),
        backend_url=args.backend_url.rstrip("/"),
        model=args.model,
        output_dir=Path(args.output_dir),
        run_id=datetime.now().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8],
        scenario=args.scenario,
        strict=args.strict,
        scenarios=scenarios,
    )
    setattr(ctx, "preflight_only", args.preflight_only)
    raise SystemExit(asyncio.run(run(ctx)))


if __name__ == "__main__":
    main()
