"""Scenario trace-gate tests for validate_backend_live."""
from __future__ import annotations

import asyncio
from pathlib import Path

from scripts.validate_backend_live import (
    load_scenarios_from_file,
    normalize_scenario,
    quality_flags,
    validate_trace_expectations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_SCENARIOS = PROJECT_ROOT / "scripts" / "odiss_engine_call_trace_scenarios.json"


def run(coro):
    return asyncio.run(coro)


def test_quality_flags_rejects_attributed_think_tags() -> None:
    flags = quality_flags('<think data-source="qwen">internal</think>\n답변입니다.')

    assert flags["no_think_tags"] is False


def test_normalize_scenario_preserves_trace_metadata() -> None:
    scenario = {
        "id": "trace_case",
        "speaker_id": "speaker-1",
        "trace_expectations": {
            "expected_engine_sequence": ["CE_Input", "ME_Context"],
            "expected_tool_calls": ["T2.병용금기정보조회"],
        },
        "steps": [
            {
                "id": "step",
                "text": "와파린이랑 아스피린 같이 먹어도 돼?",
                "expected_mode": "tool_first",
                "trace_expectations": {
                    "expected_memory_reads": ["PrescriptionLog.md"],
                    "expected_memory_writes": ["MedicationLog.md"],
                },
                "expected_tool_calls": ["T3.노인주의정보조회"],
                "must_not_call_tools": ["T13.LLM에이전트검색"],
                "expected_external_apis": ["API_MFDS_DUR"],
            }
        ],
    }

    normalized = normalize_scenario(scenario, 1)
    step = normalized["steps"][0]

    assert normalized["trace_expectations"]["expected_tool_calls"] == [
        "T2.병용금기정보조회"
    ]
    assert step["trace_expectations"]["expected_memory_reads"] == ["PrescriptionLog.md"]
    assert step["expected_tool_calls"] == ["T3.노인주의정보조회"]
    assert step["must_not_call_tools"] == ["T13.LLM에이전트검색"]
    assert step["expected_external_apis"] == ["API_MFDS_DUR"]


def test_trace_validator_accepts_expected_calls_and_memory_events() -> None:
    result = validate_trace_expectations(
        expectations={
            "expected_engine_sequence": ["CE_Input", "ME_Context", "RE_Intent", "ME_Update"],
            "expected_tool_calls": ["T2.병용금기정보조회"],
            "must_not_call_tools": ["T13.LLM에이전트검색"],
            "expected_external_apis": ["API_MFDS_DUR"],
            "expected_memory_reads": ["PrescriptionLog.md"],
            "expected_memory_writes": ["MedicationLog.md"],
        },
        engine_trace=[
            {"stage": "CE_Input", "component": "ConversationEngine", "action": "receive_input"},
            {"stage": "ME_Context", "component": "MemoryEngine", "action": "load_context"},
            {"stage": "RE_Intent", "component": "ReasoningEngine", "action": "route_execution"},
            {"stage": "ME_Update", "component": "MemoryEngine", "action": "update_and_compress"},
        ],
        memory_trace=[
            {"operation": "read", "logical_file": "PrescriptionLog.md", "category": "prescription_log"},
            {"operation": "write", "logical_file": "MedicationLog.md", "category": "medication_log"},
        ],
        tool_trace=[
            {
                "tool_id": "T2.병용금기정보조회",
                "tool_name": "combination_contraindication",
                "external_api": "API_MFDS_DUR",
            }
        ],
    )

    assert result["ok"] is True
    assert result["checks"]["engine_sequence_ok"] is True
    assert result["expected_tool_calls"]["T2.병용금기정보조회"] is True


def test_trace_validator_reports_missing_and_forbidden_calls() -> None:
    result = validate_trace_expectations(
        expectations={
            "expected_engine_sequence": ["CE_Input", "DUR_Tool.T4.DUR품목정보조회"],
            "expected_tool_calls": ["T4.DUR품목정보조회"],
            "must_not_call_tools": ["T13.LLM에이전트검색"],
        },
        engine_trace=[
            {"stage": "CE_Input", "component": "ConversationEngine", "action": "receive_input"},
        ],
        memory_trace=[],
        tool_trace=[
            {
                "tool_id": "T13.LLM에이전트검색",
                "tool_name": "llm_search",
                "external_api": "FrontierLLM",
            }
        ],
    )

    assert result["ok"] is False
    assert result["engine_sequence"]["missing"] == ["DUR_Tool.T4.DUR품목정보조회"]
    assert result["expected_tool_calls"]["T4.DUR품목정보조회"] is False
    assert result["must_not_call_tools"]["T13.LLM에이전트검색"] is True


def test_authored_trace_scenarios_load_with_trace_expectations() -> None:
    scenarios = load_scenarios_from_file(TRACE_SCENARIOS)

    assert scenarios
    first = scenarios[0]
    assert first["trace_expectations"]["expected_route"] == "memory_only"
    assert first["steps"][0]["trace_expectations"]["expected_tool_calls"] == []
    assert "forbidden_terms" in first["steps"][0]
